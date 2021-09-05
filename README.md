# writing test
## Statement
This code is used to implement the 2D convolution, relu, max-pooling operation in the forward propagation process of the convolutional neural network. The code is written and compiled on Microsoft Windows 10, Visual Studio 2019.

## Goals
1. Implement convolution, relu, pooling function using c/c++.
2. Fuse the Conv + relu + pooling into one function to reduce memory access.
3. Use OpenMP to compute in parallel on the CPU.

## Instructions
1. `Intel_test_wuhao.exe` is a compiled file on the windows platform.
2. `Intel_test_wuhao.cpp` is the source code.
3. `Report_WuHao.docx` contains some summary and screenshots of experimental results.
4. The way to input data can refer to `example.png`

## Notice
For experiments on time complexity, high-dimensional matrices are required as input to reflect the advantages of parallel computing. The current code displays the input and output matrices on the screen. If you put a high-dimensional matrix as input to test the time complexity, countless data will fill the screen. So you can add program annotation to shield some "printf" code.

The code generates a random matrix as the input and convolution kernel is just for the convenience of verifying the correctness.
