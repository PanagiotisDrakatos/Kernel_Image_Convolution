#include <stdint.h>
#include <stdbool.h>
#include "mpi.h"
#include "MPI_Setup.h"


//find the appropriate border to Send Receive requests between processors.
extern inline void BorderGreySelectionProcess(Borders * LocalPointWest,Borders *LocalPointSouth,Borders *LocalPointEast,Borders *LocalPointNorth,
                                          uint8_t *src,uint8_t *dst,int rows,int cols,float ** gaussian_blur,TypeColor typeColor);

extern inline void BorderRGBSelectionProcess(Borders * LocalPointWest,Borders *LocalPointSouth,Borders *LocalPointEast,Borders *LocalPointNorth,
                                              uint8_t *src,uint8_t *dst,int rows,int cols,float ** gaussian_blur,TypeColor typeColor);

//Nonblocking Send Receive
extern inline void SendRequestProcess(uint8_t* offset, MPI_Datatype datatype, int PointValue, MPI_Request* request);
//Nonblocking Send Receive
extern inline void ReceiveRequestProcess(uint8_t* offset, MPI_Datatype datatype, int PointValue, MPI_Request* request);
