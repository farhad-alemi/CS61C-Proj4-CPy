from utils import *
from unittest import TestCase

"""
- For each operation, you should write tests to test on matrices of different sizes.
- Keep in mind that the tests provided in the starter code are NOT comprehensive. That is, we strongly
advise you to modify them and add new tests.
- Hint: use dp_mc_matrix to generate dumbpy and numc matrices with the same data and use
        cmp_dp_nc_matrix to compare the results
"""

class TestAdd(TestCase):
    def test_small_add(self):
        # YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(2, 2, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(2, 2, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "add")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_medium_add(self):
        # YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(2001, 47, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(2001, 47, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "add")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_large_add_2(self):
        # YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(12000, 12000, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(12000, 12000, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "add")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_large_add(self):
        # YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(12000, 12000, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(12000, 12000, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "add")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

class TestSub(TestCase):
    def test_small_sub(self):
        # YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(2, 2, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(2, 2, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "sub")
        self.assertTrue(is_correct)
        try:
            nc.Matrix(3, 3) - nc.Matrix(2, 2)
            self.assertTrue(False)
        except ValueError as e:
            print(e)
            pass
        print_speedup(speed_up)

    def test_medium_sub(self):
        # YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(3029, 1, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(3029, 1, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "sub")
        self.assertTrue(is_correct)
        print_speedup(speed_up)
        

    def test_large_sub(self):
        # YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(15000, 15000, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(15000, 15000, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "sub")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_large_sub_2(self):
        # YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(12000, 12000, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(12000, 12000, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "sub")
        self.assertTrue(is_correct)
        print_speedup(speed_up)
        
class TestAbs(TestCase):
    def test_small_abs(self):
        # YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        is_correct, speed_up = compute([dp_mat], [nc_mat], "abs")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_medium_abs(self):
        # YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(1111, 93, seed=0)
        is_correct, speed_up = compute([dp_mat], [nc_mat], "abs")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_large_abs(self):
        # YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(10000, 10000, seed=0)
        is_correct, speed_up = compute([dp_mat], [nc_mat], "abs")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_large_abs_2(self):
        # YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(12000, 12000, seed=0)
        is_correct, speed_up = compute([dp_mat], [nc_mat], "abs")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

class TestNeg(TestCase):
    def test_small_neg(self):
        # YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        is_correct, speed_up = compute([dp_mat], [nc_mat], "neg")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_medium_neg(self):
        # YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(1, 1999, seed=0)
        is_correct, speed_up = compute([dp_mat], [nc_mat], "neg")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_medium_neg_2(self):
        # YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(1024, 1024, seed=0)
        is_correct, speed_up = compute([dp_mat], [nc_mat], "neg")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_large_neg(self):
        # YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(10000, 10000, seed=0)
        is_correct, speed_up = compute([dp_mat], [nc_mat], "neg")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_large_neg(self):
        # YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(12000, 12000, seed=0)
        is_correct, speed_up = compute([dp_mat], [nc_mat], "neg")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

class TestMul(TestCase):
    def test_small_mul(self):
        # YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(2, 2, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(2, 2, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "mul")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_medium_mul(self):
        # YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(1998, 247, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(247, 1262, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "mul")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_large_mul(self):
        # YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(2903, 303, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(303, 93, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "mul")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_large_mul_op(self):
        # YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(1024, 1024, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(1024, 1024, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "mul")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

class TestPow(TestCase):
    def test_small_pow(self):
        # YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        is_correct, speed_up = compute([dp_mat, 3], [nc_mat, 3], "pow")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_medium_pow(self):
        # YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(47, 47, seed=0)
        is_correct, speed_up = compute([dp_mat, 5], [nc_mat, 5], "pow")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_large_pow(self):
        # YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(999, 999, seed=0)
        is_correct, speed_up = compute([dp_mat, 10], [nc_mat, 10], "pow")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_large_pow_op(self):
        # YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(1024, 1024, seed=0)
        is_correct, speed_up = compute([dp_mat, 10], [nc_mat, 10], "pow")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_large_pow_I(self):
        # YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(1024, 1024, seed=0)
        is_correct, speed_up = compute([dp_mat, 0], [nc_mat, 0], "pow")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_large_pow_1(self):
        # YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(1024, 1024, seed=0)
        is_correct, speed_up = compute([dp_mat, 1], [nc_mat, 1], "pow")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

class TestGet(TestCase):
    def test_get(self):
        # YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        rand_row = np.random.randint(dp_mat.shape[0])
        rand_col = np.random.randint(dp_mat.shape[1])
        self.assertEqual(round(dp_mat.get(rand_row, rand_col), decimal_places),
            round(nc_mat.get(rand_row, rand_col), decimal_places))
    
    def test_large_get(self):
        # YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(5864, 10023, seed=0)
        rand_row = np.random.randint(dp_mat.shape[0])
        rand_col = np.random.randint(dp_mat.shape[1])
        self.assertEqual(round(dp_mat[rand_row][rand_col], decimal_places),
                         round(nc_mat[rand_row][rand_col], decimal_places))

class TestSet(TestCase):
    def test_set(self):
        # YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        rand_row = np.random.randint(dp_mat.shape[0])
        rand_col = np.random.randint(dp_mat.shape[1])
        dp_mat.set(rand_row, rand_col, 2)
        nc_mat.set(rand_row, rand_col, 2)
        self.assertTrue(cmp_dp_nc_matrix(dp_mat, nc_mat))
        self.assertEqual(nc_mat.get(rand_row, rand_col), 2)

    def test_large_set(self):
        # YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(9999, 9989, seed=0)
        rand_row = np.random.randint(dp_mat.shape[0])
        rand_col = np.random.randint(dp_mat.shape[1])
        self.assertEqual(round(dp_mat[rand_row][rand_col], decimal_places),
                          round(nc_mat[rand_row][rand_col], decimal_places))

class TestShape(TestCase):
    def test_shape(self):
        # YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        self.assertTrue(dp_mat.shape == nc_mat.shape)

    def test_large_shape(self):
        dp_mat, nc_mat = rand_dp_nc_matrix(1064, 1, seed=0)
        self.assertTrue(dp_mat.shape == nc_mat.shape)

class TestIndexGet(TestCase):
    def test_index_get(self):
        # YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        rand_row = np.random.randint(dp_mat.shape[0])
        rand_col = np.random.randint(dp_mat.shape[1])
        self.assertEqual(round(dp_mat[rand_row][rand_col], decimal_places),
            round(nc_mat[rand_row][rand_col], decimal_places))

class TestIndexSet(TestCase):
    def test_set(self):
        # YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        rand_row = np.random.randint(dp_mat.shape[0])
        rand_col = np.random.randint(dp_mat.shape[1])
        dp_mat[rand_row][rand_col] = 2
        nc_mat[rand_row][rand_col] = 2
        self.assertTrue(cmp_dp_nc_matrix(dp_mat, nc_mat))
        self.assertEqual(nc_mat[rand_row][rand_col], 2)

class TestSlice(TestCase):
    def test_slice(self):
        # YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        self.assertTrue(cmp_dp_nc_matrix(dp_mat[0], nc_mat[0]))
        self.assertTrue(cmp_dp_nc_matrix(dp_mat[1], nc_mat[1]))
