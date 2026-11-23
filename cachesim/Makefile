CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra

cachesim: main.cpp cache.cpp cache.hpp
	$(CXX) $(CXXFLAGS) -o $@ main.cpp cache.cpp

clean:
	rm -f cachesim
.PHONY: clean
