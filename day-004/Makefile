NVCC = nvcc
NVCCFLAGS = -std=c++11 -O3
TARGET = matmul
SOURCE = matmul.cu

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
