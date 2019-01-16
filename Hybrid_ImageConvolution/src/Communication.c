#include <stdint.h>
#include <stdbool.h>
#include "mpi.h"
#include "MPI_Setup.h"
#include "ConvolutionProcess.h"
#include <stdio.h>
#include <stdlib.h>

// Get pointer to internal array position */
static inline uint8_t *reference(uint8_t *array, int i, int j, int width) {
    return &array[width * i + j];
}

//Nonblocking Send
void SendRequestProcess(uint8_t *offset, MPI_Datatype datatype, int PointValue, MPI_Request *request) {
    MPI_Isend(offset, 1, datatype, PointValue, 0, MPI_COMM_WORLD, request);
}
//Nonblocking Receive
void ReceiveRequestProcess(uint8_t *offset, MPI_Datatype datatype, int PointValue, MPI_Request *request) {
    MPI_Irecv(offset, 1, datatype, PointValue, 0, MPI_COMM_WORLD, request);
}


// Send Receive requests between processors based on border
void BorderGreySelectionProcess(Borders * LocalPointWest,Borders *LocalPointSouth,Borders *LocalPointEast,Borders *LocalPointNorth,
                            uint8_t *src,uint8_t *dst,int rows,int cols,float ** gaussian_blur,TypeColor typeColor) {
    if (LocalPointNorth->value != -1) {
        SendRequestProcess(reference(src, 1, 1, cols + 2),grey_row_type,LocalPointNorth->value,&send_north_req);
        ReceiveRequestProcess(reference(src, 0, 1, cols+2), grey_row_type, LocalPointNorth->value, &send_north_req);

        MPI_Wait(&send_north_req, &status);
        Convolution(src, dst, 1, 1, 2, cols - 1, cols, gaussian_blur, typeColor);
    }
    if (LocalPointWest->value != -1) {

        SendRequestProcess(reference(src, 1, 1, cols + 2), grey_col_type, LocalPointWest->value, &send_west_req);
        ReceiveRequestProcess(reference(src, 1, 0, cols + 2), grey_col_type, LocalPointWest->value, &send_west_req);

        MPI_Wait(&send_west_req, &status);
        Convolution(src, dst, 2, rows - 1, 1, 1, cols, gaussian_blur, typeColor);
    }
    if (LocalPointSouth->value != -1) {

        SendRequestProcess(reference(src, rows, 1, cols + 2), grey_row_type, LocalPointSouth->value, &send_south_req);
        ReceiveRequestProcess(reference(src, rows + 1, 1, cols + 2), grey_row_type, LocalPointSouth->value, &send_south_req);

        /* Request and compute */
        MPI_Wait(&send_south_req, &status);
        Convolution(src, dst, rows, rows, 2, cols - 1, cols, gaussian_blur, typeColor);
    }
    if (LocalPointEast->value != -1) {

        SendRequestProcess(reference(src, 1, cols, cols + 2), grey_col_type, LocalPointEast->value, &send_east_req);
        ReceiveRequestProcess(reference(src, 1, cols + 1, cols + 2), grey_col_type, LocalPointEast->value, &send_east_req);

        /* Request and compute */
        MPI_Wait(&send_east_req, &status);
        Convolution(src, dst, 2, rows - 1, cols, cols, cols, gaussian_blur, typeColor);
    }
}

void BorderRGBSelectionProcess(Borders * LocalPointWest,Borders *LocalPointSouth,Borders *LocalPointEast,Borders *LocalPointNorth,
                               uint8_t *src,uint8_t *dst,int rows,int cols,float ** gaussian_blur,TypeColor typeColor){

    if (LocalPointNorth->value != -1) {
        SendRequestProcess(reference(src, 1, 3, 3 * cols + 6), rgb_row_type, LocalPointNorth->value, &send_north_req);
        ReceiveRequestProcess(reference(src, 0, 3, 3 * cols + 6), rgb_row_type, LocalPointNorth->value, &send_north_req);

        /* Request and compute */
        MPI_Wait(&send_north_req, &status);
        Convolution(src, dst, 1, 1, 2, cols - 1, cols, gaussian_blur, typeColor);
    }
    if (LocalPointWest->value != -1) {
        SendRequestProcess(reference(src, 1, 3, 3 * cols + 6), rgb_col_type, LocalPointWest->value, &send_west_req);
        ReceiveRequestProcess(reference(src, 1, 0, 3 * cols + 6), rgb_col_type, LocalPointWest->value, &send_west_req);

        /* Request and compute */
        MPI_Wait(&send_west_req, &status);
        Convolution(src, dst, 2, rows - 1, 1, 1, cols, gaussian_blur, typeColor);
    }
    if (LocalPointSouth->value != -1) {
        SendRequestProcess(reference(src, rows, 3, 3 * cols + 6), rgb_row_type, LocalPointSouth->value, &send_south_req);
        ReceiveRequestProcess(reference(src, rows + 1, 3, 3 * cols + 6), rgb_row_type, LocalPointSouth->value, &send_south_req);

        /* Request and compute */
        MPI_Wait(&send_south_req, &status);
        Convolution(src, dst, rows, rows, 2, cols - 1, cols, gaussian_blur, typeColor);
    }
    if (LocalPointEast->value != -1) {
        SendRequestProcess(reference(src, 1, 3 * cols, 3 * cols + 6), rgb_col_type, LocalPointEast->value, &send_east_req);
        ReceiveRequestProcess(reference(src, 1, 3 * cols + 3, 3 * cols + 6), rgb_col_type, LocalPointEast->value, &send_east_req);

        /* Request and compute */
        MPI_Wait(&send_east_req, &status);
        Convolution(src, dst, 2, rows - 1, cols, cols, cols, gaussian_blur, typeColor);
    }


}



