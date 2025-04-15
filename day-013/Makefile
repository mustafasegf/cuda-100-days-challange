NVCC = nvcc
NVCCFLAGS = -std=c++11 -O3
TARGET = stencil
SOURCE = stencil.cu

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

run:
	./$(TARGET)

.PHONY: all clean run
