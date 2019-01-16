#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <stdint.h>
#include "mpi.h"

//get the necessary  segmentation reference from image unsinged int
static inline uint8_t *get_imgOffset(uint8_t *array, int i, int j, int width) {
    return &array[width * i + j];
}

/* Parallel read to the grey image file with MPI */
static inline void OverloadReadFileSegmentGrey(uint8_t *source, uint8_t *temporary, char* image, int Start_Row, int Start_Col, int size,
                            int numCols, int width, MPI_Status status) {
    static MPI_File file;

    MPI_File_open(MPI_COMM_WORLD, image, MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
    for (int i = 1; i <= size; i++) {
        MPI_File_seek(file, (Start_Row + i - 1) * width + Start_Col, MPI_SEEK_SET);
        temporary = get_imgOffset(source, i, 1, numCols + 2);
        MPI_File_read(file, temporary, numCols, MPI_BYTE, &status);
    }
    MPI_File_close(&file);
}

/* Parallel read to the rgb image file with MPI */
static inline void OverloadReadFileSegmentRGB(uint8_t *source, uint8_t *temporary, char* image, int Start_Row, int Start_Col, int size,
                           int numCols, int width, MPI_Status status) {
    static MPI_File file;

    MPI_File_open(MPI_COMM_WORLD, image, MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
    for (int i = 1; i <= size; i++) {
        MPI_File_seek(file, 3 * (Start_Row + i - 1) * width + 3 * Start_Col, MPI_SEEK_SET);
        temporary = get_imgOffset(source, i, 3, numCols * 3 + 6);
        MPI_File_read(file, temporary, numCols * 3, MPI_BYTE, &status);
    }
    MPI_File_close(&file);
}

/* Parallel write to the grey image file with MPI techniques*/

static inline void OverloadWriteFileSegmentGrey(uint8_t *source, uint8_t *temporary, char* image, int start_row, int start_col, int size,
                             int numCols, int width) {
    char * outImage=malloc((strlen(image) + 9) * sizeof(char));
    static MPI_File outfile;

    strcpy(outImage, "blur_");
    strcat(outImage, image);

    MPI_File_open(MPI_COMM_WORLD, outImage, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &outfile);
    for (int i = 1; i <= size; i++) {
        MPI_File_seek(outfile, (start_row + i-1) * width + start_col, MPI_SEEK_SET);
        temporary = get_imgOffset(source, i, 1, numCols+2);
        MPI_File_write(outfile, temporary, numCols, MPI_BYTE, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&outfile);
}

/* Parallel write to the rgb image file with MPI techniques*/
static inline void OverloadWriteFileSegmentRGB(uint8_t *source, uint8_t *temporary, char* image, int start_row, int start_col, int size,
                            int numCols, int width) {
    char *outImage = malloc((strlen(image) + 9) * sizeof(char));
    static MPI_File outfile;

    strcpy(outImage, "blur_");
    strcat(outImage, image);

    MPI_File_open(MPI_COMM_WORLD, outImage, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &outfile);
    for (int i = 1; i <= size; i++) {
        MPI_File_seek(outfile, 3*(start_row + i-1) * width + 3*start_col, MPI_SEEK_SET);
        temporary = get_imgOffset(source, i, 3, numCols*3+6);
        MPI_File_write(outfile, temporary, numCols*3, MPI_BYTE, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&outfile);
}