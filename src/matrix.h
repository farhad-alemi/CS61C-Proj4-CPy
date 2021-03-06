#include <Python.h>

#define DARK_ERROR -3
#define RUNTIME_ERROR -2
#define VALUE_ERROR -1
#define STRIDE 16
#define DIMENSION_THRESHOLD 128
#define MAX_POWER 11

typedef struct matrix {
    int rows;              /* number of rows */
    int cols;              /* number of columns */
    double *data;          /* pointer to rows * columns doubles */
    int *ref_cnt;          /* How many slices/matrices are referring to this matrix's data*/
    struct matrix *parent; /* NULL if matrix is not a slice, else the parent matrix of
                           the slice */
    int _is_special;       /* Is the sliced matrix to be treated differently? */
} matrix;

double rand_double(double low, double high);
void rand_matrix(matrix *result, unsigned int seed, double low, double high);
int allocate_matrix(matrix **mat, int rows, int cols);
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols);
void deallocate_matrix(matrix *mat);
double get(matrix *mat, int row, int col);
void set(matrix *mat, int row, int col, double val);
void fill_matrix(matrix *mat, double val);
int add_matrix(matrix *result, matrix *mat1, matrix *mat2);
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2);
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2);
int pow_matrix(matrix *result, matrix *mat, int pow);
int neg_matrix(matrix *result, matrix *mat);
int abs_matrix(matrix *result, matrix *mat);
