# Last Updated: 2025-09-02

.PHONY: lock
lock:
	pip-compile --quiet --allow-unsafe --output-file requirements.lock requirements.in
	sed -i '1i# Last Updated: $(shell date -I)' requirements.lock

