# numc

Here's what I did in project 4:
## matrix.h
typedef struct matrix {
    int rows;              /* number of rows */
    int cols;              /* number of columns */
    double *data;          /* pointer to rows * columns doubles */
    int *ref_cnt;          /* How many slices/matrices are referring to this matrix's data*/
    struct matrix *parent; /* NULL if matrix is not a slice, else the parent matrix of
                           the slice */
    int _is_special;       /* Is the sliced matrix to be treated differently? */
} matrix;


## matrix.c
/* Generates a random double between low and high */
double rand_double(double low, double high)

/* Generates a random matrix */
void rand_matrix(matrix *result, unsigned int seed, double low, double high)

/* A helper method which allocates matrices and data rows if the matrix is
 not a slice. */
int allocator(matrix **mat, int rows, int cols, matrix *from, int row_offset, int col_offset, int isSlice)

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data
 * array and initialize all entries to be zeros. `parent` should be set to NULL
 * to indicate that this matrix is not a slice. You should also set `ref_cnt`
 * to 1. You should return -1 if either `rows` or `cols` or both have invalid
 * values, or if any call to allocate memory in this function fails. If you
 * don't set python error messages here upon failure, then remember to set it in
 * numc.c. Return 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols)

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and
 * `cols` columns. Its data should point to the `offset`th entry of `from`'s
 * data (you do not need to allocate memory) for the data field. `parent` should
 * be set to `from` to indicate this matrix is a slice of `from`. You should
 * return -1 if either `rows` or `cols` or both are non-positive or if any call
 * to allocate memory in this function fails. If you don't set python error
 * messages here upon failure, then remember to set it in numc.c. Return 0 upon
 * success.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols)

/*
 * You need to make sure that you only free `mat->data` if `mat` is not a slice
 * and has no existing slices, or if `mat` is the last existing slice of its
 * parent matrix and its parent matrix has no other references (including
 * itself). You cannot assume that mat is not NULL.
 */
void deallocate_matrix(matrix *mat)

/*
 * Returns the addrress at the specified location.
 */
double *get_addr(matrix *mat, int row, int col)

/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid.
 */
double get(matrix *mat, int row, int col)

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid.
 */
void set(matrix *mat, int row, int col, double val)

/*
 * Sets all entries in mat to val
 */
void fill_matrix(matrix *mat, double val)

/*
 * Validates the matrix and, if appropriate, returns the error. */
int validate(matrix *result, matrix *mat1, matrix *mat2)

/*
 * Performs various matrix operations based on OPERATION.
 * Sacrifices abstraction for the sake of performance!
 */
int mat_operator(matrix *result, matrix *mat1, matrix *mat2, char operation)

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2)

/*
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2)

/*
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int neg_matrix(matrix *result, matrix *mat)

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int abs_matrix(matrix *result, matrix *mat)

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that matrix multiplication is not the same as multiplying individual
 * elements.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2)

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise
 * multiplication.
 */
int pow_matrix(matrix *result, matrix *mat, int pow)