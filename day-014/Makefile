NVCC = nvcc
NVCCFLAGS = -std=c++11 -O3
TARGET = stencil-part2
SOURCE = stencil-part2.cu

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

run:
	./$(TARGET)

.PHONY: all clean run
