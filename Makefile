# Last Updated: 2025-09-22

.PHONY: lock
lock:
	python -m piptools compile \
		--quiet \
		--allow-unsafe \
		--output-file requirements.lock \
		requirements.in
	sed -i '1i# Last Updated: $(shell date -I)' requirements.lock

