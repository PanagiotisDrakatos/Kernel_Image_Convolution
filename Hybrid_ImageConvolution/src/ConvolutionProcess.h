#include <stdint.h>
#include <stdbool.h>
#include "mpi.h"
#include "MPI_Setup.h"

// define all the necessary methods based on image color
// to calculate all the necessary equations when filter will be enabled
extern inline int Partition(int rows, int cols, int workers);

extern inline void Convolution(uint8_t *, uint8_t *, int, int, int, int, int, float **, TypeColor);

extern inline void PerformGreyConvolution(uint8_t *src, uint8_t *dst, int rows, int cols, int width, float** blur);

extern inline void PerformRGBConvolution(uint8_t *src, uint8_t *dst, int row, int col, int width, float** blur);

extern inline void GreyConvolution(uint8_t *src, uint8_t *dst, int row_start, int row_dst, int col_start, int col_dst, int width, float** blur);

extern inline void RGBConvolution(uint8_t *src, uint8_t *dst, int row_start, int row_dst, int col_start, int col_dst, int width, float** blur);

