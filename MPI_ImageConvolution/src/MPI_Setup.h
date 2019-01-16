#include <stdint.h>
#include <stdbool.h>
#include "mpi.h"


#ifndef TypeColor_tc
typedef enum {
    RGB_Image, GREY_Image
} TypeColor;
#define TypeColor_tc
#endif

#ifndef TypePoint_tc
typedef enum {
    North,
    South,
    West,
    East
} TypePoint;
#define TypePoint_tc
#endif

#ifndef Matrix
typedef struct Kernel_Matrix {
    TypeColor typeColor;
} Kernels_Matrix;
#define Matrix
#endif


#ifndef BORDER
typedef struct Border {
    TypePoint typePoint;
    int value;
} Borders;
#define BORDER
#endif


//instantiate the two dimensional array
static inline void SetupGaussianSmoothing(float **);
//instantiate the source and destination arrays which it will need for the communication(Sending-Receiving)
static inline void Initialization(uint8_t **src, uint8_t **dst, int cols, int rows, TypeColor typeColor);
//Release the memory on exit
static inline void FreeMemory(uint8_t *src, uint8_t *dst);

/* MPI status */
MPI_Status status;
/* MPI data types */
MPI_Datatype grey_col_type;
MPI_Datatype rgb_col_type;
MPI_Datatype grey_row_type;
MPI_Datatype rgb_row_type;
/* MPI requests */
MPI_Request send_north_req;
MPI_Request send_south_req;
MPI_Request send_west_req;
MPI_Request send_east_req;