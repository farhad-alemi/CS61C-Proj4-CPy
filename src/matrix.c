/* THIS PROGRAM SACRIFICES MANY ABSTRACTION LEVELS FOR THE SAKE OF HIGHER PERFORMANCE SPEEDS!!! */

#include "matrix.h"

#include <omp.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

/* Include SSE intrinsics */
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
 */

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/* A helper method which allocates matrices and data rows if the matrix is
 not a slice. */
int allocator(matrix **mat, int rows, int cols, matrix *from, int row_offset, int col_offset, int isSlice) {
    if (cols <= 0 || rows <= 0) {
        return VALUE_ERROR;
    } else if (isSlice) {
        if (from == NULL) {
            return RUNTIME_ERROR;
        } else if (row_offset < 0 || col_offset < 0 || row_offset + rows > from->rows || col_offset + cols > from->cols) {
            return VALUE_ERROR;
        }
    }

    *mat = (matrix *)malloc(sizeof(matrix));
    if (*mat == NULL) {
        return RUNTIME_ERROR;
    }

    (*mat)->rows = rows;
    (*mat)->cols = cols;

    if (!isSlice) {
        (*mat)->_is_special = 0;
        (*mat)->data = (double *)calloc(rows * cols, sizeof(double));
        if ((*mat)->data == NULL) {
            return RUNTIME_ERROR;
        }

        (*mat)->ref_cnt = (int *)malloc(sizeof(int));
        if ((*mat)->ref_cnt == NULL) {
            return RUNTIME_ERROR;
        }

        *((*mat)->ref_cnt) = 1;
        (*mat)->parent = NULL;
    } else {
        (*mat)->_is_special = 1;
        (*mat)->data = from->data + (from->cols * row_offset) + col_offset;
        (*mat)->parent = (from->parent == NULL) ? from : from->parent;
        (*mat)->ref_cnt = (*mat)->parent->ref_cnt;
        *((*mat)->ref_cnt) = *((*mat)->ref_cnt) + 1;
    }

    return 0;
}

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
int allocate_matrix(matrix **mat, int rows, int cols) {
    /* YOUR CODE HERE */
    return allocator(mat, rows, cols, NULL, 0, 0, 0);
}

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
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    /* YOUR CODE HERE */
    if (cols <= 0 || rows <= 0 || offset < 0) {
        return VALUE_ERROR;
    } else if (from == NULL) {
        return RUNTIME_ERROR;
    }

    *mat = (matrix *)malloc(sizeof(matrix));
    if (*mat == NULL) {
        return RUNTIME_ERROR;
    }

    (*mat)->rows = rows;
    (*mat)->cols = cols;
    (*mat)->_is_special = 0;

    (*mat)->data = from->data + offset;
    (*mat)->parent = (from->parent == NULL) ? from : from->parent;
    (*mat)->ref_cnt = (*mat)->parent->ref_cnt;
    *((*mat)->ref_cnt) = *((*mat)->ref_cnt) + 1;

    return 0;
}

/*
 * You need to make sure that you only free `mat->data` if `mat` is not a slice
 * and has no existing slices, or if `mat` is the last existing slice of its
 * parent matrix and its parent matrix has no other references (including
 * itself). You cannot assume that mat is not NULL.
 */
void deallocate_matrix(matrix *mat) {
    /* YOUR CODE HERE */
    if (mat == NULL) {
        return;
    } else if (mat->data == NULL || mat->ref_cnt == NULL) {
        exit(RUNTIME_ERROR);
    }

    if (*(mat->ref_cnt) == 1) {
        if (mat->parent == NULL) {
            free(mat->data);
            free(mat->ref_cnt);
        } else {
            free(mat->parent->data);
            free(mat->parent->ref_cnt);
            free(mat->parent);
        }
        free(mat);
    } else {
        *(mat->ref_cnt) = *(mat->ref_cnt) - 1;
        if (mat->parent != NULL) {
            free(mat);
        }
    }
}

/*
 * Returns the addrress at the specified location.
 */
double *get_addr(matrix *mat, int row, int col) {
    int stride;

    if (mat == NULL || mat->data == NULL) {
        return (double *)RUNTIME_ERROR;
    } else if (row < 0 || row >= mat->rows || col < 0 || col >= mat->cols) {
        return (double *)VALUE_ERROR;
    }

    stride = (!mat->_is_special) ? mat->cols : mat->parent->cols;

    return mat->data + (stride * row) + col;
}

/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid.
 */
double get(matrix *mat, int row, int col) {
    /* YOUR CODE HERE */
    double *ret_val = get_addr(mat, row, col);

    if (ret_val == (double *)RUNTIME_ERROR || ret_val == (double *)VALUE_ERROR) {
        exit(DARK_ERROR);
    }

    return *ret_val;
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid.
 */
void set(matrix *mat, int row, int col, double val) {
    /* YOUR CODE HERE */
    double *ret_val = get_addr(mat, row, col);

    if (ret_val == (double *)RUNTIME_ERROR || ret_val == (double *)VALUE_ERROR) {
        exit(DARK_ERROR);
    }

    *ret_val = val;
}

/*
 * Sets all entries in mat to val
 */
void fill_matrix(matrix *mat, double val) {
    /* YOUR CODE HERE */
    for (int r = 0; r < mat->rows; ++r) {
        for (int c = 0; c < mat->cols; ++c) {
            set(mat, r, c, val);
        }
    }
}

/*
 * Validates the matrix and, if appropriate, returns the error. */
int validate(matrix *result, matrix *mat1, matrix *mat2) {
    if (result == NULL || mat1 == NULL || mat2 == NULL || result->data == NULL || mat1->data == NULL || mat2->data == NULL ||
        mat1->rows != result->rows || mat1->cols != result->cols) {
        return RUNTIME_ERROR;
    } else if (mat1->rows != mat2->rows || mat1->cols != mat2->cols) {
        return VALUE_ERROR;
    }

    return 0;
}

/*
 * Performs various matrix operations based on OPERATION.
 * Sacrifices abstraction for the sake of performance!
 */
int mat_operator(matrix *result, matrix *mat1, matrix *mat2, char operation) {
    int dim, threshold, small_stride, err_code;
    double *mat1_data, *mat2_data, *result_data;

    err_code = validate(result, mat1, mat2);
    if (err_code) {
        return err_code;
    }
    mat1_data = mat1->data;
    mat2_data = mat2->data;
    result_data = result->data;
    dim = result->rows * result->cols;
    threshold = dim / STRIDE * STRIDE;

    if (operation == 'I' || dim < DIMENSION_THRESHOLD) {
        small_stride = STRIDE / 2;
#pragma omp parallel
        {
            double *mat1_data_index = NULL, *mat2_data_index = NULL, *result_data_index = NULL;

#pragma omp for
            for (int index = 0; index < threshold; index += small_stride) {
                mat1_data_index = mat1_data + index;
                result_data_index = result_data + index;

                switch (operation) {
                    case '+':
                        mat2_data_index = mat2_data + index;

                        *(result_data_index) = *(mat1_data_index) + *(mat2_data_index);
                        *(result_data_index + 1) = *(mat1_data_index + 1) + *(mat2_data_index + 1);
                        *(result_data_index + 2) = *(mat1_data_index + 2) + *(mat2_data_index + 2);
                        *(result_data_index + 3) = *(mat1_data_index + 3) + *(mat2_data_index + 3);
                        *(result_data_index + 4) = *(mat1_data_index + 4) + *(mat2_data_index + 4);
                        *(result_data_index + 5) = *(mat1_data_index + 5) + *(mat2_data_index + 5);
                        *(result_data_index + 6) = *(mat1_data_index + 6) + *(mat2_data_index + 6);
                        *(result_data_index + 7) = *(mat1_data_index + 7) + *(mat2_data_index + 7);
                        break;
                    case '-':
                        mat2_data_index = mat2_data + index;

                        *(result_data_index) = *(mat1_data_index) - *(mat2_data_index);
                        *(result_data_index + 1) = *(mat1_data_index + 1) - *(mat2_data_index + 1);
                        *(result_data_index + 2) = *(mat1_data_index + 2) - *(mat2_data_index + 2);
                        *(result_data_index + 3) = *(mat1_data_index + 3) - *(mat2_data_index + 3);
                        *(result_data_index + 4) = *(mat1_data_index + 4) - *(mat2_data_index + 4);
                        *(result_data_index + 5) = *(mat1_data_index + 5) - *(mat2_data_index + 5);
                        *(result_data_index + 6) = *(mat1_data_index + 6) - *(mat2_data_index + 6);
                        *(result_data_index + 7) = *(mat1_data_index + 7) - *(mat2_data_index + 7);
                        break;
                    case '~':
                        *(result_data_index) = -*(mat1_data_index);
                        *(result_data_index + 1) = -*(mat1_data_index + 1);
                        *(result_data_index + 2) = -*(mat1_data_index + 2);
                        *(result_data_index + 3) = -*(mat1_data_index + 3);
                        *(result_data_index + 4) = -*(mat1_data_index + 4);
                        *(result_data_index + 5) = -*(mat1_data_index + 5);
                        *(result_data_index + 6) = -*(mat1_data_index + 6);
                        *(result_data_index + 7) = -*(mat1_data_index + 7);
                        break;
                    case '|':
                        *(result_data_index) = fabs(*(mat1_data_index));
                        *(result_data_index + 1) = fabs(*(mat1_data_index + 1));
                        *(result_data_index + 2) = fabs(*(mat1_data_index + 2));
                        *(result_data_index + 3) = fabs(*(mat1_data_index + 3));
                        *(result_data_index + 4) = fabs(*(mat1_data_index + 4));
                        *(result_data_index + 5) = fabs(*(mat1_data_index + 5));
                        *(result_data_index + 6) = fabs(*(mat1_data_index + 6));
                        *(result_data_index + 7) = fabs(*(mat1_data_index + 7));
                        break;
                    case 'I':
                        *(result_data_index) = (((index) / mat1->cols) == ((index) % mat1->cols)) ? 1 : 0;
                        *(result_data_index + 1) = (((index + 1) / mat1->cols) == ((index + 1) % mat1->cols)) ? 1 : 0;
                        *(result_data_index + 2) = (((index + 2) / mat1->cols) == ((index + 2) % mat1->cols)) ? 1 : 0;
                        *(result_data_index + 3) = (((index + 3) / mat1->cols) == ((index + 3) % mat1->cols)) ? 1 : 0;
                        *(result_data_index + 4) = (((index + 4) / mat1->cols) == ((index + 4) % mat1->cols)) ? 1 : 0;
                        *(result_data_index + 5) = (((index + 5) / mat1->cols) == ((index + 5) % mat1->cols)) ? 1 : 0;
                        *(result_data_index + 6) = (((index + 6) / mat1->cols) == ((index + 6) % mat1->cols)) ? 1 : 0;
                        *(result_data_index + 7) = (((index + 7) / mat1->cols) == ((index + 7) % mat1->cols)) ? 1 : 0;
                        break;
                    default:
                        break;
                }
            }
        }
    } else {
#pragma omp parallel
        {
            double *mat1_data_index = NULL, *mat2_data_index = NULL, *result_data_index = NULL;
            __m256d arr[4];
#pragma omp for
            for (int index = 0; index < threshold; index += STRIDE) {
                mat1_data_index = mat1_data + index;
                result_data_index = result_data + index;

                switch (operation) {
                    case '+':
                        mat2_data_index = mat2_data + index;

                        arr[0] = _mm256_add_pd(_mm256_loadu_pd((const double *)(mat1_data_index)),
                                               _mm256_loadu_pd((const double *)(mat2_data_index)));
                        arr[1] = _mm256_add_pd(_mm256_loadu_pd((const double *)(mat1_data_index + 4)),
                                               _mm256_loadu_pd((const double *)(mat2_data_index + 4)));
                        arr[2] = _mm256_add_pd(_mm256_loadu_pd((const double *)(mat1_data_index + 8)),
                                               _mm256_loadu_pd((const double *)(mat2_data_index + 8)));
                        arr[3] = _mm256_add_pd(_mm256_loadu_pd((const double *)(mat1_data_index + 12)),
                                               _mm256_loadu_pd((const double *)(mat2_data_index + 12)));
                        break;
                    case '-':
                        mat2_data_index = mat2_data + index;

                        arr[0] = _mm256_sub_pd(_mm256_loadu_pd((const double *)(mat1_data_index)),
                                               _mm256_loadu_pd((const double *)(mat2_data_index)));
                        arr[1] = _mm256_sub_pd(_mm256_loadu_pd((const double *)(mat1_data_index + 4)),
                                               _mm256_loadu_pd((const double *)(mat2_data_index + 4)));
                        arr[2] = _mm256_sub_pd(_mm256_loadu_pd((const double *)(mat1_data_index + 8)),
                                               _mm256_loadu_pd((const double *)(mat2_data_index + 8)));
                        arr[3] = _mm256_sub_pd(_mm256_loadu_pd((const double *)(mat1_data_index + 12)),
                                               _mm256_loadu_pd((const double *)(mat2_data_index + 12)));
                        break;
                    case '~':
                        arr[0] = _mm256_sub_pd(_mm256_setzero_pd(), _mm256_loadu_pd((const double *)(mat1_data_index)));
                        arr[1] = _mm256_sub_pd(_mm256_setzero_pd(), _mm256_loadu_pd((const double *)(mat1_data_index + 4)));
                        arr[2] = _mm256_sub_pd(_mm256_setzero_pd(), _mm256_loadu_pd((const double *)(mat1_data_index + 8)));
                        arr[3] = _mm256_sub_pd(_mm256_setzero_pd(), _mm256_loadu_pd((const double *)(mat1_data_index + 12)));
                        break;
                    case '|':
                        arr[0] = _mm256_andnot_pd(_mm256_set1_pd(-0.0), _mm256_loadu_pd((const double *)(mat1_data_index)));
                        arr[1] = _mm256_andnot_pd(_mm256_set1_pd(-0.0), _mm256_loadu_pd((const double *)(mat1_data_index + 4)));
                        arr[2] = _mm256_andnot_pd(_mm256_set1_pd(-0.0), _mm256_loadu_pd((const double *)(mat1_data_index + 8)));
                        arr[3] = _mm256_andnot_pd(_mm256_set1_pd(-0.0), _mm256_loadu_pd((const double *)(mat1_data_index + 12)));
                        break;
                    default:
                        break;
                }
                _mm256_storeu_pd(result_data_index, arr[0]);
                _mm256_storeu_pd(result_data_index + 4, arr[1]);
                _mm256_storeu_pd(result_data_index + 8, arr[2]);
                _mm256_storeu_pd(result_data_index + 12, arr[3]);
            }
        }
    }

    /* Tail Case */
    double *mat1_data_index = NULL, *mat2_data_index = NULL, *result_data_index = NULL;

    for (int index = threshold; index < dim; ++index) {
        mat1_data_index = mat1_data + index;
        result_data_index = result_data + index;
        switch (operation) {
            case '+':
                mat2_data_index = mat2_data + index;

                *(result_data_index) = *(mat1_data_index) + *(mat2_data_index);
                break;
            case '-':
                mat2_data_index = mat2_data + index;

                *(result_data_index) = *(mat1_data_index) - *(mat2_data_index);
                break;
            case '~':
                *(result_data_index) = -*(mat1_data_index);
                break;
            case '|':
                *(result_data_index) = fabs(*(mat1_data_index));
                break;
            case 'I':
                *(result_data_index) = ((index / mat1->cols) == (index % mat1->cols)) ? 1 : 0;
                break;
            default:
                return RUNTIME_ERROR;
                break;
        }
    }

    return 0;
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* YOUR CODE HERE */
    double *mat1_data, *mat2_data, *result_data;
    int err_code, dim, threshold;

    err_code = validate(result, mat1, mat2);
    if (err_code) {
        return err_code;
    }

    mat1_data = mat1->data;
    mat2_data = mat2->data;
    result_data = result->data;
    dim = result->rows * result->cols;
    threshold = dim / STRIDE * STRIDE;

#pragma omp parallel
    {
        __m256d arr[4];
        double *mat1_data_index, *mat2_data_index, *result_data_index;
#pragma omp for
        for (int index = 0; index < threshold; index += STRIDE) {
            mat1_data_index = mat1_data + index;
            mat2_data_index = mat2_data + index;
            result_data_index = result_data + index;

            arr[0] = _mm256_add_pd(_mm256_loadu_pd((const double *)(mat1_data_index)),
                                   _mm256_loadu_pd((const double *)(mat2_data_index)));
            arr[1] = _mm256_add_pd(_mm256_loadu_pd((const double *)(mat1_data_index + 4)),
                                   _mm256_loadu_pd((const double *)(mat2_data_index + 4)));
            arr[2] = _mm256_add_pd(_mm256_loadu_pd((const double *)(mat1_data_index + 8)),
                                   _mm256_loadu_pd((const double *)(mat2_data_index + 8)));
            arr[3] = _mm256_add_pd(_mm256_loadu_pd((const double *)(mat1_data_index + 12)),
                                   _mm256_loadu_pd((const double *)(mat2_data_index + 12)));

            _mm256_storeu_pd(result_data_index, arr[0]);
            _mm256_storeu_pd(result_data_index + 4, arr[1]);
            _mm256_storeu_pd(result_data_index + 8, arr[2]);
            _mm256_storeu_pd(result_data_index + 12, arr[3]);
        }
    }

    /* Tail Case. */
    for (int index = threshold; index < dim; ++index) {
        *(result_data + index) = *(mat1_data + index) + *(mat2_data + index);
    }
    return 0;
}

/*
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* YOUR CODE HERE */
    double *mat1_data, *mat2_data, *result_data;
    int err_code, dim, threshold;

    err_code = validate(result, mat1, mat2);
    if (err_code) {
        return err_code;
    }

    mat1_data = mat1->data;
    mat2_data = mat2->data;
    result_data = result->data;
    dim = result->rows * result->cols;
    threshold = dim / STRIDE * STRIDE;

#pragma omp parallel
    {
        __m256d arr[4];
        double *mat1_data_index, *mat2_data_index, *result_data_index;

#pragma omp for
        for (int index = 0; index < threshold; index += STRIDE) {
            mat1_data_index = mat1_data + index;
            mat2_data_index = mat2_data + index;
            result_data_index = result_data + index;

            arr[0] = _mm256_sub_pd(_mm256_loadu_pd((const double *)(mat1_data_index)),
                                   _mm256_loadu_pd((const double *)(mat2_data_index)));
            arr[1] = _mm256_sub_pd(_mm256_loadu_pd((const double *)(mat1_data_index + 4)),
                                   _mm256_loadu_pd((const double *)(mat2_data_index + 4)));
            arr[2] = _mm256_sub_pd(_mm256_loadu_pd((const double *)(mat1_data_index + 8)),
                                   _mm256_loadu_pd((const double *)(mat2_data_index + 8)));
            arr[3] = _mm256_sub_pd(_mm256_loadu_pd((const double *)(mat1_data_index + 12)),
                                   _mm256_loadu_pd((const double *)(mat2_data_index + 12)));

            _mm256_storeu_pd(result_data_index, arr[0]);
            _mm256_storeu_pd(result_data_index + 4, arr[1]);
            _mm256_storeu_pd(result_data_index + 8, arr[2]);
            _mm256_storeu_pd(result_data_index + 12, arr[3]);
        }
    }

    /* Tail Case. */
    for (int index = threshold; index < dim; ++index) {
        *(result_data + index) = *(mat1_data + index) - *(mat2_data + index);
    }
    return 0;
}

/*
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int neg_matrix(matrix *result, matrix *mat) {
    /* YOUR CODE HERE */

    double *mat_data, *result_data;
    int err_code, dim, threshold;

    err_code = validate(result, mat, mat);
    if (err_code) {
        return err_code;
    }

    mat_data = mat->data;
    result_data = result->data;
    dim = result->rows * result->cols;
    threshold = dim / STRIDE * STRIDE;

#pragma omp parallel
    {
        __m256d arr[4];
        double *mat_data_index, *result_data_index;

#pragma omp for
        for (int index = 0; index < threshold; index += STRIDE) {
            mat_data_index = mat_data + index;
            result_data_index = result_data + index;

            arr[0] = _mm256_sub_pd(_mm256_setzero_pd(), _mm256_loadu_pd((const double *)(mat_data_index)));
            arr[1] = _mm256_sub_pd(_mm256_setzero_pd(), _mm256_loadu_pd((const double *)(mat_data_index + 4)));
            arr[2] = _mm256_sub_pd(_mm256_setzero_pd(), _mm256_loadu_pd((const double *)(mat_data_index + 8)));
            arr[3] = _mm256_sub_pd(_mm256_setzero_pd(), _mm256_loadu_pd((const double *)(mat_data_index + 12)));

            _mm256_storeu_pd(result_data_index, arr[0]);
            _mm256_storeu_pd(result_data_index + 4, arr[1]);
            _mm256_storeu_pd(result_data_index + 8, arr[2]);
            _mm256_storeu_pd(result_data_index + 12, arr[3]);
        }
    }

    /* Tail Case. */
    for (int index = threshold; index < dim; ++index) {
        *(result_data + index) = -*(mat_data + index);
    }
    return 0;
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int abs_matrix(matrix *result, matrix *mat) {
    /* YOUR CODE HERE */
    double *mat_data, *result_data;
    int err_code, dim, threshold;

    err_code = validate(result, mat, mat);
    if (err_code) {
        return err_code;
    }

    mat_data = mat->data;
    result_data = result->data;
    dim = result->rows * result->cols;
    threshold = dim / STRIDE * STRIDE;

#pragma omp parallel
    {
        __m256d arr[4];
        double *mat_data_index, *result_data_index;

#pragma omp for
        for (int index = 0; index < threshold; index += STRIDE) {
            mat_data_index = mat_data + index;
            result_data_index = result_data + index;

            arr[0] = _mm256_andnot_pd(_mm256_set1_pd(-0.0), _mm256_loadu_pd((const double *)(mat_data_index)));
            arr[1] = _mm256_andnot_pd(_mm256_set1_pd(-0.0), _mm256_loadu_pd((const double *)(mat_data_index + 4)));
            arr[2] = _mm256_andnot_pd(_mm256_set1_pd(-0.0), _mm256_loadu_pd((const double *)(mat_data_index + 8)));
            arr[3] = _mm256_andnot_pd(_mm256_set1_pd(-0.0), _mm256_loadu_pd((const double *)(mat_data_index + 12)));

            _mm256_storeu_pd(result_data_index, arr[0]);
            _mm256_storeu_pd(result_data_index + 4, arr[1]);
            _mm256_storeu_pd(result_data_index + 8, arr[2]);
            _mm256_storeu_pd(result_data_index + 12, arr[3]);
        }
    }

    /* Tail Case. */
    for (int index = threshold; index < dim; ++index) {
        *(result_data + index) = fabs(*(mat_data + index));
    }
    return 0;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that matrix multiplication is not the same as multiplying individual
 * elements.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* YOUR CODE HERE */
    int thresh_k, thresh_j, mat1_cols, mat2_cols, result_rows, result_cols;
    double *mat1_data, *mat2_data, *result_data;

    if (result == NULL || result->data == NULL || mat1 == NULL || mat1->data == NULL || mat2 == NULL || mat2->data == NULL ||
        result->rows != mat1->rows || result->cols != mat2->cols) {
        return RUNTIME_ERROR;
    } else if (mat1->cols != mat2->rows) {
        return VALUE_ERROR;
    }

    /* Register Blocking - This reduces the number of accesses to matrix fields. */
    mat1_cols = mat1->cols;
    mat1_data = mat1->data;
    mat2_cols = mat2->cols;
    mat2_data = mat2->data;
    result_rows = result->rows;
    result_cols = result->cols;
    result_data = result->data;
    thresh_k = mat1_cols / 8 * 8;
    thresh_j = result_cols / 4 * 4;

#pragma omp parallel for
    for (int i = 0; i < result_rows; ++i) {
        double *result_data_i_result_cols = result_data + (i * result_cols);
        double *mat1_data_i_mat1_cols = mat1_data + (i * mat1_cols);

        for (int k = 0; k < thresh_k; k += 8) {
            double *mat2_data_k_mat2_cols0 = mat2_data + (k * mat2_cols);
            double *mat2_data_k_mat2_cols1 = mat2_data + ((k + 1) * mat2_cols);
            double *mat2_data_k_mat2_cols2 = mat2_data + ((k + 2) * mat2_cols);
            double *mat2_data_k_mat2_cols3 = mat2_data + ((k + 3) * mat2_cols);
            double *mat2_data_k_mat2_cols4 = mat2_data + ((k + 4) * mat2_cols);
            double *mat2_data_k_mat2_cols5 = mat2_data + ((k + 5) * mat2_cols);
            double *mat2_data_k_mat2_cols6 = mat2_data + ((k + 6) * mat2_cols);
            double *mat2_data_k_mat2_cols7 = mat2_data + ((k + 7) * mat2_cols);

            __m256d mat1_data_i_mat1_cols_k0 = _mm256_set1_pd(mat1_data_i_mat1_cols[k]);
            __m256d mat1_data_i_mat1_cols_k1 = _mm256_set1_pd(mat1_data_i_mat1_cols[k + 1]);
            __m256d mat1_data_i_mat1_cols_k2 = _mm256_set1_pd(mat1_data_i_mat1_cols[k + 2]);
            __m256d mat1_data_i_mat1_cols_k3 = _mm256_set1_pd(mat1_data_i_mat1_cols[k + 3]);
            __m256d mat1_data_i_mat1_cols_k4 = _mm256_set1_pd(mat1_data_i_mat1_cols[k + 4]);
            __m256d mat1_data_i_mat1_cols_k5 = _mm256_set1_pd(mat1_data_i_mat1_cols[k + 5]);
            __m256d mat1_data_i_mat1_cols_k6 = _mm256_set1_pd(mat1_data_i_mat1_cols[k + 6]);
            __m256d mat1_data_i_mat1_cols_k7 = _mm256_set1_pd(mat1_data_i_mat1_cols[k + 7]);

            for (int j = 0; j < thresh_j; j += 4) {
                double *mat2_data_k_mat2_cols_j0 = mat2_data_k_mat2_cols0 + j;
                double *mat2_data_k_mat2_cols_j1 = mat2_data_k_mat2_cols1 + j;
                double *mat2_data_k_mat2_cols_j2 = mat2_data_k_mat2_cols2 + j;
                double *mat2_data_k_mat2_cols_j3 = mat2_data_k_mat2_cols3 + j;
                double *mat2_data_k_mat2_cols_j4 = mat2_data_k_mat2_cols4 + j;
                double *mat2_data_k_mat2_cols_j5 = mat2_data_k_mat2_cols5 + j;
                double *mat2_data_k_mat2_cols_j6 = mat2_data_k_mat2_cols6 + j;
                double *mat2_data_k_mat2_cols_j7 = mat2_data_k_mat2_cols7 + j;

                double *result_data_i_result_cols_j = result_data_i_result_cols + j;

                _mm256_storeu_pd(
                    result_data_i_result_cols_j,
                    _mm256_fmadd_pd(
                        mat1_data_i_mat1_cols_k7, _mm256_loadu_pd(mat2_data_k_mat2_cols_j7),
                        _mm256_fmadd_pd(
                            mat1_data_i_mat1_cols_k6, _mm256_loadu_pd(mat2_data_k_mat2_cols_j6),
                            _mm256_fmadd_pd(
                                mat1_data_i_mat1_cols_k5, _mm256_loadu_pd(mat2_data_k_mat2_cols_j5),
                                _mm256_fmadd_pd(
                                    mat1_data_i_mat1_cols_k4, _mm256_loadu_pd(mat2_data_k_mat2_cols_j4),
                                    _mm256_fmadd_pd(
                                        mat1_data_i_mat1_cols_k3, _mm256_loadu_pd(mat2_data_k_mat2_cols_j3),
                                        _mm256_fmadd_pd(
                                            mat1_data_i_mat1_cols_k2, _mm256_loadu_pd(mat2_data_k_mat2_cols_j2),
                                            _mm256_fmadd_pd(mat1_data_i_mat1_cols_k1, _mm256_loadu_pd(mat2_data_k_mat2_cols_j1),
                                                            _mm256_fmadd_pd(mat1_data_i_mat1_cols_k0,
                                                                            _mm256_loadu_pd(mat2_data_k_mat2_cols_j0),
                                                                            _mm256_loadu_pd(result_data_i_result_cols_j))))))))));
            }

            /* Tail Case for j */
            for (int j = thresh_j; j < result_cols; ++j) {
                result_data_i_result_cols[j] += mat1_data_i_mat1_cols[k] * mat2_data_k_mat2_cols0[j] +
                                                mat1_data_i_mat1_cols[k + 1] * mat2_data_k_mat2_cols1[j] +
                                                mat1_data_i_mat1_cols[k + 2] * mat2_data_k_mat2_cols2[j] +
                                                mat1_data_i_mat1_cols[k + 3] * mat2_data_k_mat2_cols3[j] +
                                                mat1_data_i_mat1_cols[k + 4] * mat2_data_k_mat2_cols4[j] +
                                                mat1_data_i_mat1_cols[k + 5] * mat2_data_k_mat2_cols5[j] +
                                                mat1_data_i_mat1_cols[k + 6] * mat2_data_k_mat2_cols6[j] +
                                                mat1_data_i_mat1_cols[k + 7] * mat2_data_k_mat2_cols7[j];
            }
        }

        /* Tail Case for k */
        for (int k = thresh_k; k < mat1_cols; k++) {
            double mat1_data_i_mat1_cols_k = mat1_data_i_mat1_cols[k];
            double *mat2_data_k_mat2_cols = mat2_data + (k * mat2_cols);

            for (int j = 0; j < result_cols; j++) {
                result_data_i_result_cols[j] += mat1_data_i_mat1_cols_k * mat2_data_k_mat2_cols[j];
            }
        }
    }

    return 0;
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise
 * multiplication.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    /* YOUR CODE HERE */
    int err_code, max_pow_needed, mat_rows, mat_cols, result_rows, result_cols;
    double *mat_data, *result_data, *temp_matrix_data;
    matrix **pow_2_matrices, *temp_matrix = NULL;

    if (result == NULL || mat == NULL || result->data == NULL || mat->data == NULL || mat->rows != result->rows ||
        mat->cols != result->cols) {
        return RUNTIME_ERROR;
    } else if (pow < 0 || mat->rows != mat->cols) {
        return VALUE_ERROR;
    }

    /* Register Blocking */
    mat_rows = mat->rows;
    mat_cols = mat->cols;
    mat_data = mat->data;
    result_rows = result->rows;
    result_cols = result->cols;
    result_data = result->data;

    if (pow == 0) {
        memset(result_data, 0, sizeof(double) * result_rows * result_cols);
        for (int rc = 0; rc < result_rows; ++rc) {
            result_data[rc * result_cols + rc] = 1;
        }
        return 0;
    } else if (pow == 1) {
        memcpy(result_data, mat_data, sizeof(double) * result_rows * result_cols);
        return 0;
    } else {
        /* ---Repeated Squaring--- */

        /* Binary Rep */
        int binary_rep[MAX_POWER];
        memset(binary_rep, -1, sizeof(int) * MAX_POWER);
        int temp = pow;
        for (max_pow_needed = 0; temp > 0; ++max_pow_needed, temp /= 2) {
            binary_rep[i] = temp % 2;
        }
        /* Undo Last Increment */
        --max_pow_needed;

        /* Calcuating Powers of 2 Matrices */
        pow_2_matrices = (matrix **)malloc(sizeof(matrix *) * (max_pow_needed + 1));
        if (pow_2_matrices == NULL) {
            return RUNTIME_ERROR;
        }

        /* Allocating Matrices */
        pow_2_matrices[0] = mat;
        for (int i = 1; i <= max_pow_needed; ++i) {
            err_code = allocate_matrix(pow_2_matrices + i, mat_rows, mat_cols);
            if (err_code) {
                return err_code;
            }
        }

        /* Calculating Powers of 2 Matrices */
        for (int i = 1; i <= max_pow_needed; ++i) {
            err_code = mul_matrix(pow_2_matrices[i], pow_2_matrices[i - 1], pow_2_matrices[i - 1]);
            if (err_code) {
                return err_code;
            }
        }

        memcpy(result_data, pow_2_matrices[max_pow_needed]->data, sizeof(double) * result_rows * result_cols);
        if (pow - (1U << (size_t)max_pow_needed) > 0) {
            err_code = allocate_matrix(&temp_matrix, mat_rows, mat_cols);
            if (err_code) {
                return err_code;
            }
            temp_matrix_data = temp_matrix->data;
        }

        for (int i = max_pow_needed - 1; i > 0; --i) {  // todo slight chance for parallelization
            if (binary_rep[i] == 1) {
                err_code = mul_matrix(temp_matrix, result, pow_2_matrices[i]);
                if (err_code) {
                    return err_code;
                }
                memcpy(result_data, temp_matrix_data, sizeof(double) * result_rows * result_cols);
            }
        }

        deallocate_matrix(temp_matrix);

        /* Deallocating Powers of 2 Matrices */
        for (int i = 1; i <= max_pow_needed; ++i) {
            deallocate_matrix(pow_2_matrices[i]);
        }
        free(pow_2_matrices);
    }

    return 0;
}
