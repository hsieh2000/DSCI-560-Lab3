#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>

#define K_SIZE 3
#define NUM_KERNELS 3

__global__ void convolutionKernel(
    const float *image,
    const float *kernel,
    float *output,
    int w,
    int h,
    int K
) {
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row
    int x = blockIdx.x * blockDim.x + threadIdx.x; // col

    if (x >= w || y >= h) return;

    int pad = K / 2;
    float sum = 0.0f;

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            int img_y = y + i - pad;
            int img_x = x + j - pad;

            if (img_x >= 0 && img_x < w &&
                img_y >= 0 && img_y < h) {
                sum += image[img_y * w + img_x] *
                       kernel[i * K + j];
            }
        }
    }

    output[y * w + x] = sum;
}

extern "C" void convolutionGPU(
    float *h_image,
    float *h_kernel,
    float *h_output,
    int w,
    int h,
    int K
) {
    float *d_image, *d_kernel, *d_output;

    size_t img_size = w * h * sizeof(float);
    size_t ker_size = K * K * sizeof(float);

    cudaMalloc(&d_image, img_size);
    cudaMalloc(&d_kernel, ker_size);
    cudaMalloc(&d_output, img_size);

    cudaMemcpy(d_image, h_image, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, ker_size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((w + 15) / 16, (h + 15) / 16);

    convolutionKernel<<<grid, block>>>(d_image, d_kernel, d_output, w, h, K);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, img_size, cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_kernel);
    cudaFree(d_output);
}

char *make_output_path(const char *input_path, const char *kernel_name) {
    const char *filename = strrchr(input_path, '/');
    filename = filename ? filename + 1 : input_path;

    const char *dot = strrchr(filename, '.');
    size_t base_len = dot ? (size_t)(dot - filename) : strlen(filename);

    const char *dir = "png_out/gpu/";
    const char *ext = ".png";

    // dir + base + "_" + kernel_name + ext + \0
    size_t total_len = strlen(dir) + base_len + 1 + strlen(kernel_name) + strlen(ext) + 1;
    char *out = (char *)malloc(total_len);
    if (!out) return NULL;

    sprintf(out, "%s%.*s_%s%s", dir, (int)base_len, filename, kernel_name, ext);
    return out;
}

int main(int argc, char **argv) {

    const char *img_path = (argc > 1) ? argv[1] : "./png/png_1.png";

    int w, h, c;
    unsigned char *img = stbi_load(img_path, &w, &h, &c, 3);
    if (!img) {
        fprintf(stderr, "load failed: %s\n", stbi_failure_reason());
        return 1;
    }

    float *gray = (float *)malloc(w * h * sizeof(float));
    for (int i = 0; i < w * h; i++) {
        int idx = i * 3;
        gray[i] = (0.299f * img[idx] +
                   0.587f * img[idx + 1] +
                   0.114f * img[idx + 2]) / 255.0f;
    }
    stbi_image_free(img);

    float kernels[NUM_KERNELS][K_SIZE * K_SIZE] = {
        { -1,-1,-1, -1, 8,-1, -1,-1,-1 },              // edge
        { 1.0f/9,1.0f/9,1.0f/9, 1.0f/9,1.0f/9,1.0f/9, 1.0f/9,1.0f/9,1.0f/9 }, // blur
        { 0,-1,0, -1,5,-1, 0,-1,0 }                     // sharpen
    };

    const char *kernel_names[NUM_KERNELS] = {
        "edge",
        "blur",
        "sharpen"
    };

    float *output = (float *)malloc(w * h * sizeof(float));
    unsigned char *out_img = (unsigned char *)malloc(w * h);

    for (int k = 0; k < NUM_KERNELS; k++) {
        char *path = make_output_path(img_path, kernel_names[k]);
        if (!path) {
            fprintf(stderr, "malloc failed for path\n");
            continue;
        }

        clock_t start = clock();
        convolutionGPU(gray, kernels[k], output, w, h, K_SIZE);
        clock_t end = clock();

        fprintf(stderr, "GPU [%s] execution time: %f seconds\n",
                kernel_names[k],
                (double)(end - start) / CLOCKS_PER_SEC);

        float min = output[0], max = output[0];
        for (int i = 1; i < w * h; i++) {
            if (output[i] < min) min = output[i];
            if (output[i] > max) max = output[i];
        }

        float range = max - min;
        if (range == 0.0f) range = 1.0f;

        for (int i = 0; i < w * h; i++) {
            out_img[i] = (unsigned char)(((output[i] - min) / range) * 255.0f);
        }

        stbi_write_png(path, w, h, 1, out_img, w);
        printf("Output saved to: %s\n", path);

        free(path);
    }

    free(out_img);
    free(output);
    free(gray);
    return 0;
}
