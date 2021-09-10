#include <stdio.h>
#include <iostream>
using namespace std;

float im2col_get_pixel(float* im, int height, int width, int channels,
    int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width * (row + height * channel)];
}

void im2col(float* data_im,
    int channels, int height, int width,
    int ksize, int stride, int pad, float* data_col)
{
    int c, h, w;
    int height_col = (height + 2 * pad - ksize) / stride + 1; //特征图横排元素个数
    int width_col = (width + 2 * pad - ksize) / stride + 1;   //特征图纵向元素个数
    int channels_col = channels * height_col * width_col;   //卷积核参数数量/ 结果矩阵横排元素个数
    for (c = 0; c < channels_col; ++c) {
        
        int h_offset = c / (channels * width_col);
        int w_offset = (c / channels) % width_col;
        int c_im = c % channels;


        for (h = 0; h < ksize; ++h) {
            for (w = 0; w < ksize; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * ksize + h) * ksize + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                    im_row, im_col, c_im, pad);
            }
        }
    }
}

int main()
{
    cout << "enter img size: 4 number" << endl;
    int width, hight, channel, batchsize;
    cin >> width >> hight >> channel >> batchsize;
    int pixels = width * hight * channel;
    float** img = new float* [batchsize];
    for (int i = 0; i < batchsize; ++i)
    {
        img[i] = new float[pixels];
        for (int j = 0; j < pixels; ++j)
            img[i][j] = double(j);
    }


    cout << "enter kernel size: 2 number" << endl;
    int ksize, koutput;
    cin >> ksize >> koutput;
    int weights = ksize * ksize * channel;
    float** knernel = new float* [koutput];
    for (int i = 0; i < koutput; ++i)
    {
        knernel[i] = new float[weights];
        for (int j = 0; j < weights; ++j)
            knernel[i][j] = 2.0;
    }

    int width_out, hight_out, channel_out, batchsize_out;
    int pad, stride;
    cin >> pad >> stride;
    batchsize_out = batchsize;
    channel_out = koutput;
    width_out = (width + (2 * pad) - ksize) / stride + 1;
    hight_out = (hight + (2 * pad) - ksize) / stride + 1;
    
    int col_width = weights;
    int col_hight = width_out * hight_out;
    int col_sum = col_width * col_hight;

    float** col = new float* [batchsize];
    for (int i = 0; i < batchsize; ++ i)
    {
        col[i] = new float[col_sum];
        for (int j = 0; j < col_sum; ++j)
            col[i][j] = 0.00000000000001;
    }

    im2col(img[0], channel, hight, width, ksize,stride, pad,col[0]);
    delete[]col;
    delete[]img;
    delete[]knernel;

    return 0;
}