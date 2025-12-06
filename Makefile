all:
	nvcc main.cu -o hypercube -lGL -lSDL3

clean:
	rm -f hypercube
