NVCC = nvcc
NVCCFLAGS = -std=c++11 -O3
TARGET = blur
SOURCE = blur.cu

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

run:
	./$(TARGET)

.PHONY: all clean run
