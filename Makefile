.PHONY: all test experiment cuda run-cuda clean

all: cachesim/cachesim

cachesim/cachesim: cachesim/main.cpp cachesim/cache.cpp cachesim/cache.hpp
	$(MAKE) -C cachesim

test: all
	python3 -m pytest -q harness

experiment: all
	python3 harness/experiment.py

cuda:
	$(MAKE) -C cuda

run-cuda: cuda
	cd cuda && ./matmul && ./reduce

clean:
	$(MAKE) -C cachesim clean
	$(MAKE) -C cuda clean
