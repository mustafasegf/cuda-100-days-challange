NVCC = nvcc
NVCCFLAGS = -std=c++11 -O3
TARGET = vec-add
SOURCE = vec-add.cu

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

run:
	./vec-add

.PHONY: all clean run

