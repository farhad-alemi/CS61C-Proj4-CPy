#include <omp.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"

// Include SSE intrinsics
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
        (*mat)->data = (double *)malloc(sizeof(double) * rows * cols);
        if ((*mat)->data == NULL) {
            return RUNTIME_ERROR;
        }

        (*mat)->ref_cnt = (int *)malloc(sizeof(int));
        if ((*mat)->ref_cnt == NULL) {
            return RUNTIME_ERROR;
        }

        *((*mat)->ref_cnt) = 1;
        (*mat)->parent = NULL;
        fill_matrix(*mat, INITIAL_VALUE);
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
 * Performs various matrix operations based on OPERATION.
 */
int mat_operator(matrix *result, matrix *mat1, matrix *mat2, char operation) {
    double first_val;

    if (result == NULL || mat1 == NULL || mat2 == NULL || result->data == NULL || mat1->data == NULL || mat2->data == NULL ||
        mat1->rows != result->rows || mat1->cols != result->cols) {
        return RUNTIME_ERROR;
    } else if (mat1->rows != mat2->rows || mat1->cols != mat2->cols) {
        return VALUE_ERROR;
    }

    for (int r = 0; r < result->rows; ++r) {
        for (int c = 0; c < result->cols; ++c) {
            switch (operation) {
                case '+':
                    set(result, r, c, get(mat1, r, c) + get(mat2, r, c));
                    break;
                case '-':
                    set(result, r, c, get(mat1, r, c) - get(mat2, r, c));
                    break;
                case '~':
                    set(result, r, c, -get(mat1, r, c));
                    break;
                case '|':
                    first_val = get(mat1, r, c);
                    set(result, r, c, (first_val >= 0) ? first_val : -first_val);
                    break;
                case 'I':
                    set(result, r, c, (r == c) ? 1 : 0);
                    break;
                case '=':
                    set(result, r, c, get(mat1, r, c));
                    break;
                default:
                    return RUNTIME_ERROR;
                    break;
            }
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
    return mat_operator(result, mat1, mat2, '+');
}

/*
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* YOUR CODE HERE */
    return mat_operator(result, mat1, mat2, '-');
}

/*
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int neg_matrix(matrix *result, matrix *mat) {
    /* YOUR CODE HERE */
    return mat_operator(result, mat, mat, '~');
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int abs_matrix(matrix *result, matrix *mat) {
    /* YOUR CODE HERE */
    return mat_operator(result, mat, mat, '|');
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that matrix multiplication is not the same as multiplying individual
 * elements.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* YOUR CODE HERE */
    int err_code;
    double temp;

    if (result == NULL || result->data == NULL || mat1 == NULL || mat1->data == NULL || mat2 == NULL || mat2->data == NULL ||
        result->rows != mat1->rows || result->cols != mat2->cols) {
        return RUNTIME_ERROR;
    } else if (mat1->cols != mat2->rows) {
        return VALUE_ERROR;
    }

    for (int i = 0; i < result->rows; ++i) {
        for (int k = 0; k < mat1->cols; ++k) {
            temp = get(mat1, i, k);
            for (int j = 0; j < result->cols; ++j) {
                set(result, i, j, get(result, i, j) + (temp * get(mat2, k, j)));
            }
        }
    }
}

/*
 * Returns the largest power of two that is smaller than POW.
 */
int calculate_largest_pow2(int number) {
    if (number < 1) {
        return DARK_ERROR;
    }

    for (int pow_2 = 1; pow_2 < 8 * sizeof(unsigned int); ++pow_2) {
        if ((1U << (unsigned int)pow_2) > number) {
            return pow_2 - 1;
        }
    }
    return -1;
}

/*
 * Calculates all powers of 2 matrices upto LARGEST_POW.
 */
int calculate_pow2_matrices(matrix *mat, matrix ***pow_2_matrices, int largest_pow) {
    int err_code;

    if (largest_pow < 1) {
        return VALUE_ERROR;
    }

    *pow_2_matrices = (matrix **)malloc(sizeof(matrix **) * (largest_pow + 1));
    if (pow_2_matrices == NULL) {
        return RUNTIME_ERROR;
    }

    (*pow_2_matrices)[0] = mat;

    for (int i = 1; i <= largest_pow; ++i) {
        err_code = allocate_matrix((*pow_2_matrices + i), mat->rows, mat->cols);
        if (err_code) {
            return err_code;
        }
    }

    for (int i = 1; i <= largest_pow; ++i) {
        err_code = mul_matrix(*(*pow_2_matrices + i), *(*pow_2_matrices + i - 1), *(*pow_2_matrices + i - 1));
        if (err_code) {
            return err_code;
        }
    }
    return 0;
}

/*
 * Frees the power of 2 matrices and the array.
 */
int deallocate_pow2_matrices(matrix **matrices, int len) {
    if (matrices == NULL) {
        return DARK_ERROR;
    }

    for (int i = 1; i < len; ++i) {
        deallocate_matrix(matrices[i]);
    }

    free(matrices);

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
    int err_code, largest_pow_2, remaining_power, curr_power;
    matrix **pow_2_matrices, *temp_matrix = NULL;

    if (result == NULL || mat == NULL || result->data == NULL || mat->data == NULL || mat->rows != result->rows ||
        mat->cols != result->cols) {
        return RUNTIME_ERROR;
    } else if (pow < 0 || mat->rows != mat->cols) {
        return VALUE_ERROR;
    }

    if (pow == 0) {
        return mat_operator(result, mat, mat, 'I');
    } else if (pow == 1) {
        return mat_operator(result, mat, mat, '=');
    } else {
        /* Repeated Squaring */
        largest_pow_2 = calculate_largest_pow2(pow);
        if (largest_pow_2 == 0 || largest_pow_2 == -1) {
            return VALUE_ERROR;
        }

        err_code = calculate_pow2_matrices(mat, &pow_2_matrices, largest_pow_2);
        if (err_code) {
            return err_code;
        }

        err_code = mat_operator(result, pow_2_matrices[largest_pow_2], pow_2_matrices[largest_pow_2], '=');
        if (err_code) {
            return err_code;
        }

        remaining_power = pow - (1U << (size_t)largest_pow_2);
        if (remaining_power > 0) {
            allocate_matrix(&temp_matrix, mat->rows, mat->cols);
        }

        while (remaining_power > 0) {
            curr_power = calculate_largest_pow2(remaining_power);
            err_code = mul_matrix(temp_matrix, result, pow_2_matrices[curr_power]);
            if (err_code) {
                return err_code;
            }

            err_code = mat_operator(result, temp_matrix, temp_matrix, '=');
            if (err_code) {
                return err_code;
            }

            remaining_power -= (1U << (size_t)curr_power);
        }

        deallocate_matrix(temp_matrix);
        deallocate_pow2_matrices(pow_2_matrices, largest_pow_2 + 1);
    }

    return 0;
}