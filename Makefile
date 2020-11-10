.PHONY: all
all:
	mkdir -p build
	cmake -E chdir build cmake ..
	cmake --build build
