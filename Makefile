# Last Updated: 2025-11-05
.ONESHELL:

.PHONY: lock
lock:
	# Delegate to ``tools/update_lock.py`` so the banner remains stable
	# when the dependency body does not change.
	python tools/update_lock.py
