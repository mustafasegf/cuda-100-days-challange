NVCC = nvcc
NVCCFLAGS = -diag-suppress=1650  -std=c++11 -O3
TARGET = colorToGrayscale
SOURCE = colorToGrayscale.cu

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

run:
	./$(TARGET)

.PHONY: all clean run
