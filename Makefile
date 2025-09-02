# Last Updated: 2025-09-01

.PHONY: lock
lock:
	pip-compile --quiet --generate-hashes --allow-unsafe --output-file requirements.lock requirements.in
	sed -i '1i# Last Updated: $(shell date -I)' requirements.lock

