#include <iostream>
#include <omp.h>

using namespace std;

float** fused_conv(float** feature, int& row_i, int& col_i, float** filter, int size_k, const unsigned stride, const unsigned pooling_num, const float bias)
{
	double startTime, endTime;
	int row_t = (row_i - size_k) / stride + 1;
	int col_t = (col_i - size_k) / stride + 1;
	int row_tp = row_t / pooling_num + ((row_t % pooling_num) ? 1 : 0);
	int col_tp = col_t / pooling_num + ((col_t % pooling_num) ? 1 : 0);
	float** tensor_c = new float* [row_t];
	float** tensor_p = new float* [row_tp];
	for (int i = 0; i < row_t; ++i)
	{
		tensor_c[i] = new float[col_t];
		for (int j = 0; j < col_t; ++j)
			tensor_c[i][j] = bias;
	}
	for (int i = 0; i < row_tp; ++i)
	{
		tensor_p[i] = new float[col_tp];
		for (int j = 0; j < col_tp; ++j)
			tensor_p[i][j] = -FLT_MAX;
	}
	//convolution-relu-pooling fused function
	startTime = omp_get_wtime();
#pragma omp parallel for collapse(2)
	for (int m = 0; m < row_t; ++m)
		for (int n = 0; n < col_t; ++n)
		{
			//convolution
			for (int a = 0; a < size_k && (m * stride + a) < row_i; ++a)
				for (int b = 0; b < size_k && (n * stride + b) < col_i; ++b)
					tensor_c[m][n] += (filter[a][b] * feature[m * stride + a][n * stride + b]);
			//relu
			tensor_c[m][n] = (tensor_c[m][n] > 0) ? tensor_c[m][n] : 0;
			//max_pooling
			tensor_p[m / pooling_num][n / pooling_num] = (tensor_p[m / pooling_num][n / pooling_num]) < tensor_c[m][n] ? tensor_c[m][n] : (tensor_p[m / pooling_num][n / pooling_num]);
		}
	endTime = omp_get_wtime();

	row_i = row_tp;
	col_i = col_tp;

	cout << "The output of convolution layer :" << endl;
	//	print convolution layer output
	for (unsigned c = 0; c < row_t; ++c)
	{
		for (unsigned b = 0; b < col_t; ++b)
			printf("%5.2f\t", tensor_c[c][b]);
		cout << endl;
	}
	cout << endl;

	delete[]tensor_c;
	//	cout << "Time consumed by fused function without Openmp optimization: " << endTime - startTime << endl;
	cout << "Time consumed by fused function using Openmp optimization: " << endTime - startTime << endl;
	cout << endl;
	return  tensor_p;
}

int main()
{
	// input
	int row, col, ke_s, pooling_size, stride;
	float bias;
	cout << "Using the random 2D matrix and kernel for test." << endl;
	cout << "Please enter two dimension of input matrix:" << endl;
	cin >> row >> col;
	cout << "Please enter the size of convolutional kernel:" << endl;
	cin >> ke_s;
	cout << "Please enter pooling size :" << endl;
	cin >> pooling_size;
	cout << "Please enter convolution bias :" << endl;
	cin >> bias;
	cout << "Please enter convolution stride :" << endl;
	cin >> stride;
	float** img = new float* [row];
	memset(img, NULL, sizeof(float*) * row);
	float** kernel = new float* [ke_s];
	memset(kernel, NULL, sizeof(float*) * ke_s);
	//	cout << "Please enter the input matrix: " << endl;
	for (int i = 0; i < row; ++i)
	{
		img[i] = new float[col];
		for (int j = 0; j < col; ++j)
			//cin >> img[i][j];
			img[i][j] = (rand() * 10 / float(RAND_MAX - 1)) - 5;
	}
	//cout << "Please enter the convolutional kernel matrix: " << endl;

	for (int i = 0; i < ke_s; ++i)
	{
		kernel[i] = new float[ke_s];
		for (int j = 0; j < ke_s; ++j)
			//		cin >> kernel[i][j];
			kernel[i][j] = (rand() * 10 / float(RAND_MAX - 1)) - 5;
	}
	cout << "The input matrix is: " << endl;
	//	print input
	for (auto j = 0; j < row; ++j)
	{
		for (auto k = 0; k < col; ++k)
			printf("%5.2f\t", img[j][k]);
		cout << endl;
	}
	cout << endl;
	cout << "The input Kernel is: " << endl;
	// print kernel
	for (auto j = 0; j < ke_s; ++j)
	{
		for (auto k = 0; k < ke_s; ++k)
			printf("%5.2f\t", kernel[j][k]);
		cout << endl;
	}
	cout << endl;
	//	fused_conv operation.
	float** tensor_out = fused_conv(img, row, col, kernel, ke_s, stride, pooling_size, bias);
	delete[]img;
	delete[]kernel;
	cout << "The result of fused convolution-relu-pooling function is :" << endl;
	//	print output
	for (unsigned c = 0; c < row; ++c)
	{
		for (unsigned b = 0; b < col; ++b)
			printf("%5.2f\t", tensor_out[c][b]);
		cout << endl;
	}
	delete[]tensor_out;
	system("PAUSE");
	return 0;
}
