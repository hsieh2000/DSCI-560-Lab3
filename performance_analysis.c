#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    char cmd[128];
    const char *programs[] = {
        "./matrix_cpu",
        "./matrix_gpu",
        "./matrix_gpu_tiling",
        "./cuBLAS"
    };
    
    const char *n_dims[] = {
        "512",
        "1024",
        "2048"
    };
    
    const char *title[] = {
	"CPU (C)",
	"Naïve CUDA",
	"Optimized CUDA",
	"cuBLAS "
    };

    FILE *csv = fopen("results.csv", "w");
    if (!csv) {
        perror("fopen");
        return 1;
    }

    fprintf(csv, "Implementation,N=512,N=1024,N=2048\n");
		for (int i = 0; i < 4; i ++) {
		
				char row[256];
				strcpy(row, title[i]);  // include <string.h>
				
		    for (int j = 0; j < 3; j++) {
					    char buffer[128];
					    snprintf(cmd, sizeof(cmd), "%s %s", programs[i], n_dims[j]);
			        FILE *pipe = popen(cmd, "r");
			        if (!pipe) {
			            perror("popen");
			            continue;
			        }

			        if (fgets(buffer, sizeof(buffer), pipe)) {
									buffer[strcspn(buffer, "\n")] = '\0';
					        size_t used = strlen(row);
					        size_t remain = sizeof(row) - used - 1;
					
					        snprintf(row + used, remain, ",%s", buffer);			       
			        }
			        
			        pclose(pipe);
			        
		    }
		    
		    fprintf(csv, "%s\n", row);
		}
	
    fclose(csv);
    return 0;
}
