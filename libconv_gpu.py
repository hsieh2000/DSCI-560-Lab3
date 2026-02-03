import ctypes
import time 
import numpy as np

lib = ctypes.CDLL("./libconv_gpu.so")

lib.convolutionGPU.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int
]

w, h = 512, 512
image = np.random.rand(w*h).astype(np.float32)
edge_kernel = np.array([-1,-1,-1,-1,8,-1,-1,-1,-1], dtype=np.float32)
blur_kernel = np.array([1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9], dtype=np.float32)
sharpen_kernel = np.array([0,-1,0, -1,5,-1, 0,-1,0], dtype=np.float32)

kernel_names = ["edge_kernel", "blur_kernel ", "sharpen_kernel "]
output = np.zeros(w*h, dtype=np.float32)

for i, j in enumerate([edge_kernel, blur_kernel, sharpen_kernel]):
		start = time.time()
		lib.convolutionGPU(
		    image.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
		    j.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
		    output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
		    w, h, 3
		)
		
		end = time.time()
		print(f"Python call to CUDA library 'libconv_gpu.so' using {kernel_names[i]} completed in {end - start:.4f} seconds") 
		
		
