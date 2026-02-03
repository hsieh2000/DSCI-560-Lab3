#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


#include <stdio.h>
#include <stdlib.h>
#include <time.h> 
#include <string.h>

#define K_SIZE 3
#define NUM_KERNELS 3

const char *kernel_names[NUM_KERNELS] = {
    "edge_detection",
    "blur",
    "sharpen"
};

void convolutionCPU(
		const char *out_path,
    const float *image,
    const float *kernel,
    int w,
    int h,
    int K
) {
		float *output = malloc(w * h * sizeof(float));
		if (!output) {
		    fprintf(stderr, "malloc failed\n");
		    return;
		}
		
    int pad = K / 2;

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
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
    }
    
		float min = output[0], max = output[0];
		for (int i = 1; i < w * h; i++) {
	
		    if (output[i] < min) min = output[i];
		    if (output[i] > max) max = output[i];
		}
		
		unsigned char *out_img = malloc(w * h * sizeof(unsigned char));
		if (!out_img) {
		    free(output);
		    return;
        }
    
		float range = max - min;
		if (range == 0.0f) range = 1.0f;

		for (int i = 0; i < w * h; i++) {
		    float norm = (output[i] - min) / range ;
		    out_img[i] = (unsigned char)(norm * 255.0f);
		}
		
    stbi_write_png(out_path, w, h, 1, out_img, w);
    
		free(output);
		free(out_img);
}

char *make_output_path(const char *input_path, const char *kernel_name) {
    const char *filename = strrchr(input_path, '/');
    filename = filename ? filename + 1 : input_path;

    const char *dot = strrchr(filename, '.');
    size_t base_len = dot ? (size_t)(dot - filename) : strlen(filename);

    const char *dir = "png_out/cpu/";
    const char *ext = ".png";

    // dir + base + "_" + kernel_name + ext + \0
    size_t total_len = strlen(dir) + base_len + 1 + strlen(kernel_name) + strlen(ext) + 1;
    char *out = malloc(total_len);
    if (!out) return NULL;

    sprintf(out, "%s%.*s_%s%s", dir, (int)base_len, filename, kernel_name, ext);
    return out;
}



int main(int argc, char **argv) {		
		const char *img_path = (argc > 1) ? argv[1] : "./png/png_1.png";
		
		int w, h, c;
		// read PNG，forcibly convert RGBA to RGB
		unsigned char *img = stbi_load(img_path, &w, &h, &c, 3);
		printf("channels = %d\n", c);

    if (!img) {
        fprintf(stderr, "load failed: %s\n", stbi_failure_reason());
        return 1;
    }

		float *gray = malloc(w * h * sizeof(float));
		if (!gray) {
		    fprintf(stderr, "malloc failed\n");
		    return 1;
		}
		
    
    // RGB → Grayscale（R=G=B）
    for (int i = 0; i < w * h; i++) {
        int idx = i * 3;

        unsigned char r = img[idx];
        unsigned char g = img[idx + 1];
        unsigned char b = img[idx + 2];
        gray[i] = (0.299*r + 0.587*g + 0.114*b) / 255.0f;
    }
    
		stbi_image_free(img);
		
		float kernels[NUM_KERNELS][K_SIZE * K_SIZE] = {
		    // Edge detection
		    {
				    -1.0f, -1.0f, -1.0f, 
						-1.0f, 8.0f, -1.0f, 
						-1.0f, -1.0f, -1.0f
				},
		    // Blur
		    {
		        1.0f/9, 1.0f/9, 1.0f/9,
		        1.0f/9, 1.0f/9, 1.0f/9,
		        1.0f/9, 1.0f/9, 1.0f/9
		    },
		    // Sharpen
		    {
		         0.0f, -1.0f,  0.0f,
		        -1.0f,  5.0f, -1.0f,
		         0.0f, -1.0f,  0.0f
		    }
		};

		for (int k = 0; k < NUM_KERNELS; k++) {
				char *path = make_output_path(img_path, kernel_names[k]);
				if (!path) {
						fprintf(stderr, "malloc failed for path\n");
						continue;
				}
			
		    clock_t start = clock();
				convolutionCPU(path, gray, kernels[k], w, h, K_SIZE);
		    clock_t end = clock();
		    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
		    fprintf(stderr, "CPU [%s] execution time: %f seconds\n", kernel_names[k], elapsed);
				printf("Output saved to: %s\n", path);

				free(path);
		}

		free(gray);

		return 0;
}
