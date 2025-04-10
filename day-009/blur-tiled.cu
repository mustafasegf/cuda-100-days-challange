#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#define CHANNELS 3
#define BLUR_SIZE 2

__global__ void blur_rgb_kernel(unsigned char *in, unsigned char *out,
                                int width, int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col >= width || row >= height)
    return;

  int red = 0, green = 0, blue = 0;
  int pixels = 0;

  for (int blurRow = -BLUR_SIZE; blurRow <= BLUR_SIZE; ++blurRow) {
    for (int blurCol = -BLUR_SIZE; blurCol <= BLUR_SIZE; ++blurCol) {
      int curRow = row + blurRow;
      int curCol = col + blurCol;

      if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
        int offset = (curRow * width + curCol) * CHANNELS;
        red += in[offset];
        green += in[offset + 1];
        blue += in[offset + 2];

        pixels++;
      }
    }
  }

  int outOffset = (row * width + col) * CHANNELS;
  out[outOffset] = red / pixels;
  out[outOffset + 1] = green / pixels;
  out[outOffset + 2] = blue / pixels;
}

__global__ void blur_rgb_tiled_kernel(unsigned char *in, unsigned char *out,
                                      int width, int height) {
  __shared__ unsigned char tile[(16 + 2 * BLUR_SIZE)][(16 + 2 * BLUR_SIZE)]
                               [CHANNELS];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = blockIdx.x * blockDim.x + tx;
  int row = blockIdx.y * blockDim.y + ty;

  for (int ch = 0; ch < CHANNELS; ch++) {
    int srcCol = col - BLUR_SIZE;
    int srcRow = row - BLUR_SIZE;

    int clampedCol = min(max(srcCol, 0), width - 1);
    int clampedRow = min(max(srcRow, 0), height - 1);

    int inOffset = (clampedRow * width + clampedCol) * CHANNELS + ch;
    tile[ty][tx][ch] = in[inOffset];
  }

  __syncthreads();

  if (col >= width || row >= height)
    return;

  int red = 0, green = 0, blue = 0;
  int pixels = 0;

  for (int dy = -BLUR_SIZE; dy <= BLUR_SIZE; dy++) {
    for (int dx = -BLUR_SIZE; dx <= BLUR_SIZE; dx++) {
      int x = tx + dx;
      int y = ty + dy;

      if (x >= 0 && x < blockDim.x + 2 * BLUR_SIZE && y >= 0 &&
          y < blockDim.y + 2 * BLUR_SIZE) {
        red += tile[y][x][0];
        green += tile[y][x][1];
        blue += tile[y][x][2];
        pixels++;
      }
    }
  }

  int outOffset = (row * width + col) * CHANNELS;
  out[outOffset + 0] = red / pixels;
  out[outOffset + 1] = green / pixels;
  out[outOffset + 2] = blue / pixels;
}

#pragma pack(push, 1)
typedef struct {
  uint16_t bfType;
  uint32_t bfSize;
  uint16_t bfReserved1;
  uint16_t bfReserved2;
  uint32_t bfOffBits;
} BMPHeader;

typedef struct {
  uint32_t biSize;
  int32_t biWidth;
  int32_t biHeight;
  uint16_t biPlanes;
  uint16_t biBitCount;
  uint32_t biCompression;
  uint32_t biSizeImage;
  int32_t biXPelsPerMeter;
  int32_t biYPelsPerMeter;
  uint32_t biClrUsed;
  uint32_t biClrImportant;
} DIBHeader;

typedef struct {
  uint8_t b, g, r;
} RGB;
#pragma pack(pop)

void saveRGB24BMP(const char *filename, const uint8_t *pixels, int width,
                  int height) {
  int rowSize = (3 * width + 3) & ~3;
  int imageSize = rowSize * height;

  BMPHeader bmp = {};
  DIBHeader dib = {};

  bmp.bfType = 0x4D42;                                   // 'BM'
  bmp.bfOffBits = sizeof(BMPHeader) + sizeof(DIBHeader); // No palette
  bmp.bfSize = bmp.bfOffBits + imageSize;
  bmp.bfReserved1 = 0;
  bmp.bfReserved2 = 0;

  dib.biSize = sizeof(DIBHeader);
  dib.biWidth = width;
  dib.biHeight = height;
  dib.biPlanes = 1;
  dib.biBitCount = 24;   // 24-bit RGB
  dib.biCompression = 0; // BI_RGB (no compression)
  dib.biSizeImage = imageSize;
  dib.biXPelsPerMeter = 2835;
  dib.biYPelsPerMeter = 2835;
  dib.biClrUsed = 0;
  dib.biClrImportant = 0;

  FILE *f = fopen(filename, "wb");
  if (!f)
    return;

  fwrite(&bmp, sizeof(bmp), 1, f);
  fwrite(&dib, sizeof(dib), 1, f);

  uint8_t *row = new uint8_t[rowSize];

  for (int y = height - 1; y >= 0; y--) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * 3;
      row[x * 3 + 0] = pixels[idx + 2]; // B
      row[x * 3 + 1] = pixels[idx + 1]; // G
      row[x * 3 + 2] = pixels[idx + 0]; // R
    }
    for (int x = width * 3; x < rowSize; x++) {
      row[x] = 0;
    }
    fwrite(row, 1, rowSize, f);
  }

  delete[] row;
  fclose(f);
}

bool load24BitBMP(const char *filename, uint8_t **out_pixels, int *width,
                  int *height) {
  FILE *f = fopen(filename, "rb");
  if (!f)
    return false;

  BMPHeader bmp;
  DIBHeader dib;

  fread(&bmp, sizeof(bmp), 1, f);
  fread(&dib, sizeof(dib), 1, f);

  if (bmp.bfType != 0x4D42 || dib.biBitCount != 24 || dib.biCompression != 0) {
    fclose(f);
    return false;
  }

  *width = dib.biWidth;
  *height = dib.biHeight;
  int rowSize = (3 * *width + 3) & ~3;

  uint8_t *row = (uint8_t *)malloc(rowSize);
  uint8_t *pixels = (uint8_t *)malloc(3 * (*width) * (*height));
  if (!row || !pixels) {
    free(row);
    fclose(f);
    return false;
  }

  fseek(f, bmp.bfOffBits, SEEK_SET);
  for (int y = 0; y < *height; y++) {
    fread(row, 1, rowSize, f);
    for (int x = 0; x < *width; x++) {
      RGB *px = (RGB *)(row + x * 3);
      int idx = ((*height - 1 - y) * *width + x) * 3;
      pixels[idx + 0] = px->r;
      pixels[idx + 1] = px->g;
      pixels[idx + 2] = px->b;
    }
  }

  free(row);
  fclose(f);
  *out_pixels = pixels;
  return true;
}

bool load24BitBMPFromBuffer(const uint8_t *data, size_t len,
                            uint8_t **out_pixels, int *width, int *height) {
  if (len < sizeof(BMPHeader) + sizeof(DIBHeader))
    return false;

  const BMPHeader *bmp = (const BMPHeader *)data;
  const DIBHeader *dib = (const DIBHeader *)(data + sizeof(BMPHeader));

  if (bmp->bfType != 0x4D42 || dib->biBitCount != 24 || dib->biCompression != 0)
    return false;

  *width = dib->biWidth;
  *height = dib->biHeight;
  int rowSize = (3 * *width + 3) & ~3;

  const uint8_t *pixelData = data + bmp->bfOffBits;
  uint8_t *pixels = (uint8_t *)malloc(3 * (*width) * (*height));
  if (!pixels)
    return false;

  for (int y = 0; y < *height; y++) {
    const uint8_t *row = pixelData + y * rowSize;
    for (int x = 0; x < *width; x++) {
      const RGB *px = (const RGB *)(row + x * 3);
      int idx = ((*height - 1 - y) * *width + x) * 3;
      pixels[idx + 0] = px->r;
      pixels[idx + 1] = px->g;
      pixels[idx + 2] = px->b;
    }
  }

  *out_pixels = pixels;
  return true;
}

int main(int argc, char **argv) {
  uint8_t *pixels = nullptr;
  int width, height;

  if (argc != 2) {
    std::cerr << "Failed to load image from file" << std::endl;
    return -1;
  }

  if (!load24BitBMP(argv[1], &pixels, &width, &height)) {
    std::cerr << "Failed to load image from file\n";
    return -1;
  }

  std::cout << "Loaded image: " << width << " x " << height << "\n";

  size_t size = width * height * 3;

  for (size_t i = 0; i < std::min<size_t>(10, size); ++i) {
    std::cout << "RGB[" << i << "] = (" << (int)pixels[i * 3 + 0] << ", "
              << (int)pixels[i * 3 + 1] << ", " << (int)pixels[i * 3 + 2]
              << ")\n";
  }

  std::cout << std::endl;

  uint8_t *pixels_out = (uint8_t *)malloc(size);
  if (!pixels_out)
    return 4;

  uint8_t *Pin_d, *Pout_d;
  cudaMalloc((void **)&Pin_d, size);
  cudaMalloc((void **)&Pout_d, size);

  cudaMemcpy(Pin_d, pixels, size, cudaMemcpyHostToDevice);

  dim3 dimGrid((width + 15) / 16, (height + 15) / 16);
  dim3 dimBlock(16, 16);

  cudaDeviceSynchronize();
  auto start = std::chrono::high_resolution_clock::now();
  blur_rgb_kernel<<<dimGrid, dimBlock>>>(Pin_d, Pout_d, width, height);
  cudaDeviceSynchronize();

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = end - start;

  std::cout << std::fixed << std::setprecision(3)
            << "gpu naive: " << duration.count() << " ms\n";

  cudaMemcpy(pixels_out, Pout_d, size, cudaMemcpyDeviceToHost);
  saveRGB24BMP("blur_naive.bmp", pixels_out, width, height);

  cudaMemcpy(Pin_d, pixels, size, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  start = std::chrono::high_resolution_clock::now();
  blur_rgb_tiled_kernel<<<dimGrid, dimBlock>>>(Pin_d, Pout_d, width, height);
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();
  duration = end - start;

  std::cout << "gpu shared: " << duration.count() << " ms\n";

  cudaMemcpy(pixels_out, Pout_d, size, cudaMemcpyDeviceToHost);
  saveRGB24BMP("blur_shared.bmp", pixels_out, width, height);

  for (size_t i = 0; i < std::min<size_t>(10, size); ++i) {
    std::cout << "RGB[" << i << "] = (" << (int)pixels_out[i * 3 + 0] << ", "
              << (int)pixels_out[i * 3 + 1] << ", "
              << (int)pixels_out[i * 3 + 2] << ")\n";
  }

  free(pixels);
  free(pixels_out);
  return 0;
}
