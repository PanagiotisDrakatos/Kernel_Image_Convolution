#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <stdint.h>
#include <unistd.h>
#include <stdio.h>
#include <limits.h>
#include "mpi.h"
#include "src/Parallel_ReadWriteIO.c"
#include "src/ConvolutionProcess.c"
#include "src/MPI_Setup.h"
#include "src/Communication.c"


//  get file path in unix systems
char *getcwd(char *buf, size_t size);


/* Main arguments:
 * 1. executable path/<filename>
 * 2. the image to be processed path/<filename>
 * 3. the width of the image
 * 4. the height of the image
 * 5. number of times to apply the filter on the image
 * 6. is black and white

 EXAMPLE*/
//mpirun -np 1 ./a.out waterfall_1920_2520.raw 1920 2520 1 rgb
/* argv[0] = "aout";
 argv[1] = "waterfall_1920_2520.raw";
 argv[2] = "1920";
 argv[3] = "2520";
 argv[4] = "1";
 argv[5] = "rgb";
 argc = 6;*/

int main(int argc, char **argv) {
    int width, height, loops, row_div, col_div, rows, cols;
    int rank, num_processes;
    char *image, cwd[PATH_MAX];
    double timer, startwtime, endwtime;
    float **gaussian_blur = malloc(3 * sizeof(float *));
    Kernels_Matrix *kernel_matrix = (Kernels_Matrix *) malloc(sizeof(Kernels_Matrix));


    /* Check arguments */
    if (argc < 6 || argc > 7) {
        printf("Wrong number of parameters given.\n");
        printf("Usage: %s <img_filename> <width> <height> <number of repetitions"
               "> <isBW image (0/1)> <withConvergence(0/1)> <number of rounds to "
               "check for convergence [optional]>\n", argv[0]);
        return 1;
    }
    if (argc == 6 && !strcmp(argv[5], "grey")) {
        image = malloc((strlen(argv[1]) + 1) * sizeof(char));
        strcpy(image, argv[1]);
        width = atoi(argv[2]);
        height = atoi(argv[3]);
        loops = atoi(argv[4]);
        kernel_matrix->typeColor = GREY_Image;
    } else if (argc == 6 && !strcmp(argv[5], "rgb")) {
        image = malloc((strlen(argv[1]) + 1) * sizeof(char));
        strcpy(image, argv[1]);
        width = atoi(argv[2]);
        height = atoi(argv[3]);
        loops = atoi(argv[4]);
        kernel_matrix->typeColor = RGB_Image;
    } else {
        fprintf(stderr, "\nError Input!\n%s image_name width height loops [rgb/grey].\n\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        exit(EXIT_FAILURE);
    }

    //mpirun -np 1 ./a.out waterfall_grey_1920_2520.raw 1920 2520 1 rgb

    /* MPI world topology */

    /* Find current task id */

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);




    /* Check arguments */
    if (rank == 0) {
        /* Division of data in each process */
        row_div = Partition(height, width, num_processes);
        if (row_div <= 0 || height % row_div || num_processes % row_div ||
            width % (col_div = num_processes / row_div)) {
            fprintf(stderr, "%s: Cannot divide to processes\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return EXIT_FAILURE;
        }
    }
    if (rank != 0) {
        image = malloc((strlen(argv[1]) + 1) * sizeof(char));
        strcpy(image, argv[1]);
    }

    //Bcast the necessary value to all processors
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&loops, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&kernel_matrix->typeColor, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&row_div, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&col_div, 1, MPI_INT, 0, MPI_COMM_WORLD);
    //BroadcastMessages(&width, &height, &loops, &kernel_matrix->typeColor, &row_div, &col_div);

    /* Compute number of rows per process */
    rows = height / row_div;
    cols = width / col_div;

    /* Create column data type for grey & rgb */
    MPI_Type_vector(rows, 1, cols + 2, MPI_BYTE, &grey_col_type);
    MPI_Type_vector(rows, 3, 3 * cols + 6, MPI_BYTE, &rgb_col_type);

    MPI_Type_commit(&grey_col_type);
    MPI_Type_commit(&rgb_col_type);

    /* Create row data type for grey & rgb*/
    MPI_Type_contiguous(cols, MPI_BYTE, &grey_row_type);
    MPI_Type_contiguous(3 * cols, MPI_BYTE, &rgb_row_type);

    MPI_Type_commit(&grey_row_type);
    MPI_Type_commit(&rgb_row_type);

    /* Compute starting row and column */
    int Local_StartRow = (rank / col_div) * rows;
    int Local_StartCol = (rank % col_div) * cols;
    uint8_t *src = NULL, *dst = NULL, *tmpbuf = NULL, *tmp = NULL;



    //setup the gausian filter for the process
    SetupGaussianSmoothing(gaussian_blur);
    Initialization(&src, &dst, cols, rows, kernel_matrix->typeColor);

    if (src == NULL || dst == NULL) {
        fprintf(stderr, "%s: Not enough memory\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        // return EXIT_FAILURE;
    }


    //read image byte based on color as unsigned_int8
    kernel_matrix->typeColor == GREY_Image ? OverloadReadFileSegmentGrey(src, tmpbuf, image, Local_StartRow, Local_StartCol, rows,
                                                                         cols, width, status)
                                           : OverloadReadFileSegmentRGB(src, tmpbuf, image, Local_StartRow, Local_StartCol, rows,
                                                                        cols, width, status);

    /* Neighbours definition*/
    Borders *LocalPointNorth=(Borders *) malloc(sizeof(Borders));
    Borders *LocalPointSouth=(Borders *) malloc(sizeof(Borders));
    Borders *LocalPointWest=(Borders *) malloc(sizeof(Borders));
    Borders *LocalPointEast=(Borders *) malloc(sizeof(Borders));

    LocalPointNorth->typePoint = North;
    LocalPointSouth->typePoint = South;
    LocalPointWest->typePoint = West;
    LocalPointEast->typePoint = East;

    LocalPointNorth->value = -1;
    LocalPointSouth->value = -1;
    LocalPointWest->value = -1;
    LocalPointEast->value = -1;

    /* now we make a computation for neighbours */
    if (Local_StartRow != 0)
        LocalPointNorth->value = rank - col_div;
    if (Local_StartRow + rows != height)
        LocalPointSouth->value = rank + col_div;
    if (Local_StartCol != 0)
        LocalPointWest->value = rank - 1;
    if (Local_StartCol + cols != width)
        LocalPointEast->value = rank + 1;

    MPI_Barrier(MPI_COMM_WORLD);
    startwtime = MPI_Wtime();
    /* Get time before */
    /* Convolution "loops" times */
    for (int l = 0; l < loops; l++) {

        switch(kernel_matrix->typeColor){
            case GREY_Image:
                BorderGreySelectionProcess(LocalPointWest,
                                           LocalPointSouth,
                                           LocalPointEast,
                                           LocalPointNorth,src,dst,rows,cols,gaussian_blur,kernel_matrix->typeColor);
                break;
            case RGB_Image:
                BorderRGBSelectionProcess(LocalPointWest,
                                          LocalPointSouth,
                                          LocalPointEast,
                                          LocalPointNorth,src,dst,rows,cols,gaussian_blur,kernel_matrix->typeColor);
                break;
            default:
                break;
        }

        /* Inner Data Convolution */
        Convolution(src, dst, 1, rows, 1, cols, cols, gaussian_blur, kernel_matrix->typeColor);


        /* Corner data calculations */
        if (LocalPointNorth->value != -1 && LocalPointWest->value != -1)
            Convolution(src, dst, 1, 1, 1, 1, cols, gaussian_blur, kernel_matrix->typeColor);
        if (LocalPointWest->value != -1 && LocalPointSouth->value != -1)
            Convolution(src, dst, rows, rows, 1, 1, cols, gaussian_blur, kernel_matrix->typeColor);
        if (LocalPointSouth->value != -1 && LocalPointEast->value != -1)
            Convolution(src, dst, rows, rows, cols, cols, cols, gaussian_blur, kernel_matrix->typeColor);
        if (LocalPointEast->value != -1 && LocalPointNorth->value != -1)
            Convolution(src, dst, 1, 1, cols, cols, cols, gaussian_blur, kernel_matrix->typeColor);


        /* swap arrays */
        tmp = src;
        src = dst;
        dst = tmp;
    }
    /* Get time elapsed */
    endwtime = MPI_Wtime() - timer;

    //write file segments in a raw file based on the color type
    kernel_matrix->typeColor == GREY_Image ? OverloadWriteFileSegmentGrey(src, tmpbuf, image, Local_StartRow, Local_StartCol,
                                                                          rows, cols, width)
                                           : OverloadWriteFileSegmentRGB(src, tmpbuf, image, Local_StartRow, Local_StartCol, rows,
                                                                         cols, width);

    /* Get the elapse time of the execution and return the saved image path */
    if (rank == 0) {
        if (getcwd(cwd, sizeof(cwd)) != NULL) {
            printf("Raw file was save with success at current working dir: %s\n", cwd);
        } else {
            perror("getcwd() error");
            return 1;
        }
        printf("Time elapsed: %.4f ms\n", (endwtime - startwtime));
    }


    FreeMemory((uint8_t *) src, (uint8_t *) dst);
    return EXIT_SUCCESS;
}

//instantiate the two dimensional array
void SetupGaussianSmoothing(float **gaussian_blur) {
    static int Blur[3][3] = {{1, 2, 1},
                             {2, 4, 2},
                             {1, 2, 1}};
    for (int i = 0; i < 3; i++)
        gaussian_blur[i] = malloc(3 * sizeof(float));
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            gaussian_blur[i][j] = Blur[i][j] / 16.0;
        }
    }
}

/* Init arrays */
void Initialization(uint8_t* *src, uint8_t* *dst, int cols, int rows, TypeColor typeColor) {
    uint8_t *source ,*destination;

    if (typeColor == GREY_Image) {
        source = calloc((rows + 2) * (cols + 2), sizeof(uint8_t));
        destination = calloc((rows + 2) * (cols + 2), sizeof(uint8_t));

        *src = source;
        *dst = destination;
    } else if (typeColor == RGB_Image) {
        source = calloc((rows + 2) * (cols * 3 + 6), sizeof(uint8_t));
        destination = calloc((rows + 2) * (cols * 3 + 6), sizeof(uint8_t));

        *src = source;
        *dst = destination;
    }


}

//Release the memory on exit
void FreeMemory(uint8_t *src, uint8_t *dst) {
    /* Release the allocated space  and set to null*/

    MPI_Type_free(&rgb_col_type);
    MPI_Type_free(&rgb_row_type);
    MPI_Type_free(&grey_col_type);
    MPI_Type_free(&grey_row_type);

    if (src != NULL) {
        free(src);
    }
    if (dst != NULL) {
        free(dst);
    }

    /* Finalize and exit */
    MPI_Finalize();
}