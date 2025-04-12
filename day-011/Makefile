NVCC = nvcc
NVCCFLAGS = -std=c++11 -O3
TARGET = transpose-shared-memory-bank
SOURCE = transpose-shared-memory-bank.cu

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

run:
	./$(TARGET)

.PHONY: all clean run
