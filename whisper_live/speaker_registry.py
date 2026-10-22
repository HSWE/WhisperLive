"""Private, server-side storage of precomputed speaker embeddings.

Use `python -m whisper_live.speaker_registry --dir PATH enroll ID AUDIO.wav`.
Use `python -m whisper_live.speaker_registry --dir PATH batch data/speaker_enrollment.json`.
Registration is an offline admin operation; no registration HTTP endpoint is exposed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"
_FORMAT_VERSION = 1
_SPEAKER_ID = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")


class SpeakerRegistry:
    def __init__(self, directory: str | Path, model_name: str = DEFAULT_MODEL):
        self.directory = Path(directory).expanduser()
        self.model_name = model_name

    def _path(self, speaker_id: str) -> Path:
        if not isinstance(speaker_id, str) or not _SPEAKER_ID.fullmatch(speaker_id):
            raise ValueError(f"Invalid speaker ID: {speaker_id!r}")
        return self.directory / f"{speaker_id}.npz"

    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.size == 0 or not np.all(np.isfinite(vector)):
            raise ValueError("Speaker embedding must be nonempty and finite")
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm < 1e-8:
            raise ValueError("Speaker embedding has zero or invalid norm")
        return vector / norm

    def save(self, speaker_id: str, embedding: np.ndarray) -> Path:
        dest = self._path(speaker_id)
        vector = self._normalize(embedding)
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)

        fd, tmp_name = tempfile.mkstemp(
            dir=self.directory, prefix=f".{speaker_id}.", suffix=".npz"
        )
        try:
            with os.fdopen(fd, "wb") as target:
                np.savez_compressed(
                    target,
                    format_version=np.array(_FORMAT_VERSION),
                    embedding_model=np.array(self.model_name),
                    embedding=vector,
                )
                target.flush()
                os.fsync(target.fileno())
            os.replace(tmp_name, dest)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        return dest

    def load(self, speaker_id: str) -> np.ndarray:
        path = self._path(speaker_id)
        with np.load(path, allow_pickle=False) as data:
            version = int(data["format_version"].item())
            model = str(data["embedding_model"].item())
            embedding = data["embedding"]
        if version != _FORMAT_VERSION:
            raise ValueError(f"Unsupported registry format for {speaker_id}: {version}")
        if model != self.model_name:
            raise ValueError(
                f"Speaker {speaker_id} enrolled with {model!r}, "
                f"but current model is {self.model_name!r}. Re-enroll."
            )
        return self._normalize(embedding)

    def list_ids(self) -> list[str]:
        if not self.directory.exists():
            return []
        return sorted(
            entry.stem
            for entry in self.directory.glob("*.npz")
            if _SPEAKER_ID.fullmatch(entry.stem) and entry.is_file()
        )

    def delete(self, speaker_id: str) -> None:
        self._path(speaker_id).unlink()


def read_batch_manifest(
    manifest_path: str | Path, registry: SpeakerRegistry
) -> dict[str, list[Path]]:
    """Read {speaker_id: [wav, ...]}, resolving paths relative to the JSON."""
    manifest = Path(manifest_path).expanduser().resolve()

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key!r}")
            result[key] = value
        return result

    raw = json.loads(
        manifest.read_text(encoding="utf-8"), object_pairs_hook=unique_keys
    )
    if not isinstance(raw, dict) or not raw:
        raise ValueError(
            "Manifest must be a nonempty {speaker_id: [audio_paths]} object"
        )

    plans = {}
    for speaker_id, paths in raw.items():
        registry._path(speaker_id)  # validate ID before model initialization
        if (
            not isinstance(paths, list)
            or not paths
            or not all(isinstance(p, str) and p.strip() for p in paths)
        ):
            raise ValueError(
                f"Speaker {speaker_id!r} needs a nonempty list of audio paths"
            )
        resolved = [
            (
                Path(p).expanduser()
                if Path(p).expanduser().is_absolute()
                else manifest.parent / p
            ).resolve()
            for p in paths
        ]
        if len(set(resolved)) != len(resolved):
            raise ValueError(f"Speaker {speaker_id!r} has duplicate audio files")
        for path in resolved:
            if not path.is_file():
                raise FileNotFoundError(f"Missing audio for {speaker_id!r}: {path}")
        plans[speaker_id] = resolved
    return plans


def compute_reference_centroid(
    registry: SpeakerRegistry,
    diarizer,
    speaker_id: str,
    audio_files: list[Path],
    load_audio,
) -> np.ndarray:
    """One normalized embedding per recording, then a normalized mean."""
    registry._path(speaker_id)
    embeddings = []
    for audio_file in audio_files:
        audio_file = Path(audio_file)
        if not audio_file.is_file():
            raise FileNotFoundError(audio_file)
        audio = load_audio(str(audio_file))
        if len(audio) < 3 * 16000:
            raise ValueError(f"Reference must be at least 3s: {audio_file}")
        if float(np.sqrt(np.mean(audio**2))) < 0.001:
            raise ValueError(f"Reference is too quiet: {audio_file}")
        if not diarizer.enroll_speaker(speaker_id, audio):
            raise ValueError(f"Cannot compute embedding: {audio_file}")
        embeddings.append(registry._normalize(diarizer.speakers[speaker_id].copy()))
        print(f"Embedded {speaker_id}: {audio_file} ({len(audio) / 16000:.1f}s)")

    vectors = np.stack(embeddings)
    centroid = registry._normalize(np.mean(vectors, axis=0))
    scores = vectors @ centroid
    print(
        f"Reference cohesion {speaker_id}: samples={len(embeddings)} "
        f"min={scores.min():.3f} mean={scores.mean():.3f}"
    )
    return centroid


def main() -> None:
    parser = argparse.ArgumentParser(description="WhisperLive speaker registry")
    parser.add_argument(
        "--dir",
        default=os.environ.get("WHISPERLIVE_SPEAKER_REGISTRY_DIR"),
        help="Private directory for speaker embeddings",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("WHISPERLIVE_SPEAKER_MODEL", DEFAULT_MODEL),
        help="Exact pyannote embedding model ID",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    enroll = commands.add_parser(
        "enroll", help="Create or replace one speaker embedding"
    )
    enroll.add_argument("speaker_id")
    enroll.add_argument("audio_files", nargs="+", type=Path)
    batch = commands.add_parser("batch", help="Enroll speakers from a JSON manifest")
    batch.add_argument("manifest", type=Path)
    commands.add_parser("list", help="List registered speaker IDs")
    remove = commands.add_parser("delete", help="Delete speaker embedding")
    remove.add_argument("speaker_id")
    args = parser.parse_args()
    if not args.dir:
        parser.error("--dir or WHISPERLIVE_SPEAKER_REGISTRY_DIR is required")
    registry = SpeakerRegistry(args.dir, model_name=args.model)

    if args.command == "list":
        for speaker_id in registry.list_ids():
            print(speaker_id)
        return
    if args.command == "delete":
        registry.delete(args.speaker_id)
        print(f"Deleted {args.speaker_id}")
        return

    if args.command == "batch":
        plans = read_batch_manifest(args.manifest, registry)
    else:
        plans = {args.speaker_id: args.audio_files}

    # Import pyannote only for enrollment; load the heavy model once per batch.
    from whisper_live.diarization import SpeakerDiarizer, load_audio

    diarizer = SpeakerDiarizer(embedding_model=args.model)

    # Compute all references before writing any registry files. A bad recording
    # cannot leave an earlier speaker partially updated.
    prepared = {
        speaker_id: compute_reference_centroid(
            registry, diarizer, speaker_id, audio_files, load_audio
        )
        for speaker_id, audio_files in plans.items()
    }
    for speaker_id, centroid in prepared.items():
        path = registry.save(speaker_id, centroid)
        print(f"Registered {speaker_id} -> {path}")


if __name__ == "__main__":
    main()
