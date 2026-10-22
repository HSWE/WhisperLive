# =====================================
# Speaker Registry
# =====================================

PYTHON := .venv/bin/python

SPEAKER_MANIFEST ?= data/speaker_enrollment.json
SPEAKER_REGISTRY_DIR := $(CURDIR)/data/speaker_registry

export WHISPERLIVE_SPEAKER_REGISTRY_DIR := $(SPEAKER_REGISTRY_DIR)

.PHONY: speaker-enroll speaker-list speaker-delete

# Batch enroll speakers from JSON
speaker-enroll:
	@mkdir -p "$(SPEAKER_REGISTRY_DIR)"
	$(PYTHON) -m whisper_live.speaker_registry \
		batch "$(SPEAKER_MANIFEST)"

# List registered speakers
speaker-list:
	$(PYTHON) -m whisper_live.speaker_registry list

# Delete a registered speaker
speaker-delete:
	@test -n "$(SPEAKER_ID)" || \
		(echo "Usage: make speaker-delete SPEAKER_ID=Junas_001"; exit 1)
	$(PYTHON) -m whisper_live.speaker_registry \
		delete "$(SPEAKER_ID)"