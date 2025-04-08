#include <cstdio>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <cmath>

#define KERNEL_SIZE 3
#define KERNEL_RADIUS (KERNEL_SIZE / 2)

__global__ void sobel_kernel(
    unsigned char *out,
    const unsigned char *in,
    size_t width, size_t height)
{
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < KERNEL_RADIUS || col >= width - KERNEL_RADIUS ||
      row < KERNEL_RADIUS || row >= height - KERNEL_RADIUS)
    return;

  float gx = 0.0f;
  float gy = 0.0f;

  for (int dy = -KERNEL_RADIUS; dy <= KERNEL_RADIUS; dy++)
  {
    for (int dx = -KERNEL_RADIUS; dx <= KERNEL_RADIUS; dx++)
    {
      size_t x = col + dx;
      size_t y = row + dy;
      uint8_t pixel = in[y * width + x];

      float weight_x = dx * 1.0f;
      float weight_y = dy * 1.0f;

      gx += pixel * weight_x;
      gy += pixel * weight_y;
    }
  }

  float mag = sqrtf(gx * gx + gy * gy);
  out[row * width + col] = (unsigned char)(fminf(255.0f, mag));
}

#pragma pack(push, 1)
typedef struct
{
  uint16_t bfType;
  uint32_t bfSize;
  uint16_t bfReserved1;
  uint16_t bfReserved2;
  uint32_t bfOffBits;
} BMPHeader;

typedef struct
{
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

typedef struct
{
  uint8_t b, g, r;
} RGB;
#pragma pack(pop)

void save_grayscale_bmp(const char *filename, const uint8_t *pixels, int width, int height)
{
  int rowSize = (width + 3) & ~3;
  int imageSize = rowSize * height;

  BMPHeader bmp = {};
  DIBHeader dib = {};

  bmp.bfType = 0x4D42;                                             // 'BM'
  bmp.bfOffBits = sizeof(BMPHeader) + sizeof(DIBHeader) + 256 * 4; // palette after headers
  bmp.bfSize = bmp.bfOffBits + imageSize;
  bmp.bfReserved1 = 0;
  bmp.bfReserved2 = 0;

  dib.biSize = sizeof(DIBHeader);
  dib.biWidth = width;
  dib.biHeight = height;
  dib.biPlanes = 1;
  dib.biBitCount = 8;    // 8-bit grayscale
  dib.biCompression = 0; // BI_RGB
  dib.biSizeImage = imageSize;
  dib.biXPelsPerMeter = 2835; // ~72 DPI
  dib.biYPelsPerMeter = 2835;
  dib.biClrUsed = 256;
  dib.biClrImportant = 256;

  FILE *f = fopen(filename, "wb");
  if (!f)
    return;

  fwrite(&bmp, sizeof(bmp), 1, f);
  fwrite(&dib, sizeof(dib), 1, f);

  for (int i = 0; i < 256; i++)
  {
    uint8_t entry[4] = {(uint8_t)i, (uint8_t)i, (uint8_t)i, 0};
    fwrite(entry, 4, 1, f);
  }

  uint8_t *row = new uint8_t[rowSize];
  for (int y = height - 1; y >= 0; y--)
  {
    for (int x = 0; x < width; x++)
      row[x] = pixels[y * width + x];
    for (int x = width; x < rowSize; x++)
      row[x] = 0;
    fwrite(row, 1, rowSize, f);
  }

  delete[] row;
  fclose(f);
}

bool load_8bit_grayscale_bmp(const char *filename, uint8_t **out_pixels, int *width, int *height)
{
  FILE *f = fopen(filename, "rb");
  if (!f)
    return false;

  BMPHeader bmp;
  DIBHeader dib;

  fread(&bmp, sizeof(bmp), 1, f);
  fread(&dib, sizeof(dib), 1, f);

  if (bmp.bfType != 0x4D42 || dib.biBitCount != 8 || dib.biCompression != 0)
  {
    fclose(f);
    return false;
  }

  *width = dib.biWidth;
  *height = dib.biHeight;
  int rowSize = (*width + 3) & ~3; // 4-byte aligned rows

  fseek(f, 256 * 4, SEEK_CUR);

  uint8_t *pixels = (uint8_t *)malloc(*width * *height);
  uint8_t *row = (uint8_t *)malloc(rowSize);
  if (!pixels || !row)
  {
    free(pixels);
    free(row);
    fclose(f);
    return false;
  }

  for (int y = 0; y < *height; y++)
  {
    fread(row, 1, rowSize, f);
    memcpy(pixels + (*height - 1 - y) * *width, row, *width);
  }

  free(row);
  fclose(f);
  *out_pixels = pixels;
  return true;
}

int main(int argc, char **argv)
{
  uint8_t *pixels = nullptr;
  int width, height;

  if (argc != 2)
  {
    std::cerr << "Failed to load image from file" << std::endl;
    return -1;
  }

  if (!load_8bit_grayscale_bmp(argv[1], &pixels, &width, &height))
  {
    std::cerr << "Failed to load image from file" << std::endl;
    return -1;
  }

  std::cout << "Loaded image: " << width << " x " << height << "\n";

  size_t size = width * height;

  for (int y = 0; y < std::min<size_t>(3, height); ++y)
  {
    for (int x = 0; x < std::min<size_t>(3, width); ++x)
    {
      int index = y * width + x;
      std::cout << "Pixel[" << y << "][" << x << "] = " << (int)pixels[index] << "\n";
    }
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
  sobel_kernel<<<dimGrid, dimBlock>>>(Pout_d, Pin_d, width, height);

  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << "\n";
  }

  cudaMemcpy(pixels_out, Pout_d, size, cudaMemcpyDeviceToHost);

  save_grayscale_bmp("out.bmp", pixels_out, width, height);

  for (int y = 0; y < std::min<size_t>(3, height); ++y)
  {
    for (int x = 0; x < std::min<size_t>(3, width); ++x)
    {
      int index = y * width + x;
      std::cout << "Pixel[" << y << "][" << x << "] = " << (int)pixels_out[index] << "\n";
    }
  }

  free(pixels);
  free(pixels_out);
  return 0;
}
