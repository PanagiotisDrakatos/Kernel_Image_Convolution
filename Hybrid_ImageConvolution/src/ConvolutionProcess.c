#include "ConvolutionProcess.h"
#include <stdlib.h>
#include <stdio.h>
#include "omp.h"

//call the necessary process base on image
inline void Convolution(uint8_t *src, uint8_t *dst, int row_start, int row_dst, int col_start, int col_dst, int width, float** blur, TypeColor imageType) {
    imageType==GREY_Image ?GreyConvolution(src,dst,row_start,row_dst,col_start,col_dst,width,blur):RGBConvolution(src,dst,row_start,row_dst,col_start,col_dst,width,blur);
}

//output of kernel image with grey color mixed with the gaussian filter
inline void PerformGreyConvolution(uint8_t *src, uint8_t *dst, int rows, int cols, int width, float** blur) {
    float sum = 0;
    //loop arrays and schedule arrays dest and src with parallel open mp
#pragma omp parallel for shared(src, dst) schedule(static) collapse(3)
    for (int i = rows-1, k = 0 ; i <= rows+1 ; i++, k++)
        for (int j = cols-1, l = 0 ; j <= cols+1 ; j++, l++)
            sum += src[width * i + j] * blur[k][l];
    dst[width * rows + cols] = sum;
  //  printf("Value of x is %f, address of x \n", sum);
}

//output of kernel image with rgb color mixed with the gaussian filter
inline void PerformRGBConvolution(uint8_t *src, uint8_t *dst, int row, int col, int width, float** blur) {
    int i, j, k, l;
    float redval = 0, greenval = 0, blueval = 0;
    //loop arrays and schedule arrays dest and src with parallel open mp
#pragma omp parallel for shared(src, dst) schedule(static) collapse(3)
    for (i = row-1, k = 0 ; i <= row+1 ; i++, k++)
        for (j = col-3, l = 0 ; j <= col+3 ; j+=3, l++){
            redval += src[width * i + j]* blur[k][l];
            greenval += src[width * i + j+1] * blur[k][l];
            blueval += src[width * i + j+2] * blur[k][l];
        }
    dst[width * row + col] = redval;
    dst[width * row + col+1] = greenval;
    dst[width * row + col+2] = blueval;
  //    printf("Value of x is %f, address of x \n", redval);
}


//find the sub array which is splitted in blocks based on the available processors gor grey image
inline void GreyConvolution(uint8_t *src, uint8_t *dst, int row_start, int row_dst, int col_start, int col_dst, int width, float** blur){
    for (int i = row_start ; i <= row_dst ; i++)
        for (int j = col_start ; j <= col_dst ; j++)
            PerformGreyConvolution(src, dst, i, j, width+2, blur);
}

//find the sub array which is splitted in blocks based on the available processors gor rgb image
inline void RGBConvolution(uint8_t *src, uint8_t *dst, int row_start, int row_dst, int col_start, int col_dst, int width, float** blur){
    for (int i = row_start ; i <= row_dst ; i++)
        for (int j = col_start ; j <= col_dst ; j++)
            PerformRGBConvolution(src, dst, i, j*3, width*3+6, blur);
}


/* Divide rows and columns in order to make best effort and minimize perimeter of blocks */
inline int Partition(int rows, int cols, int workers) {
    int per, rows_to, cols_to, best = 0;
    int per_min = rows + cols + 1;
    for (rows_to = 1 ; rows_to <= workers ; ++rows_to) {
        if(rows_to!=0) {
            if (workers % rows_to || rows % rows_to) continue;
            cols_to = workers / rows_to;
            if (cols % cols_to) continue;
            per = rows / rows_to + cols / cols_to;
            if (per < per_min) {
                per_min = per;
                best = rows_to;
            }
        }
    }
    return best;
}

