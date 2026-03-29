ROOT_MAKE := $(MAKE) -C ..

.PHONY: help python-setup python-cmdstan-install python-test python-typecheck benchmark benchmark-bg showcase

help:
	$(ROOT_MAKE) help

python-setup:
	$(ROOT_MAKE) python-setup

python-cmdstan-install:
	$(ROOT_MAKE) python-cmdstan-install

python-test:
	$(ROOT_MAKE) python-test

python-typecheck:
	$(ROOT_MAKE) python-typecheck

benchmark:
	$(ROOT_MAKE) benchmark

benchmark-bg:
	$(ROOT_MAKE) benchmark-bg

showcase:
	$(ROOT_MAKE) showcase
