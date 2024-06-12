import unittest
import countoscope
import numpy as np
import numpy.testing
import time

# Countoscope test suite
# at the moment this doesn't do much except check no exceptions are thrown,
# but I hope it can grow in scope
# I have not included test_data.dat in the repository for the moment, let me (Adam) know if you need it

# run all tests with `python test.py` (you must be in the directory)
# run a single test with `python test.py Tests.test_squares_single_sep` etc.
import matplotlib.pyplot as plt
class Tests(unittest.TestCase):
    def setUp(self):        
        data = np.fromfile('test_data/example_dataset.txt', dtype=float, sep=' ') # load data array as a single big line
        self.data = data.reshape((-1, 3)) # reshape it so that it has x / y / t on each row
        # the test data is on a 100 x 100 window btw
        
    def test_nmsd(self):
        results = countoscope.calculate_nmsd(self.data, box_sizes=(2, 4, 8, 16), sep_sizes=(7, 5, -1, -9), window_size_x=100, window_size_y=100)
        expected_results = np.load('test_data/example_counted_expectation.npz')
        numpy.testing.assert_allclose(results.nmsd, expected_results['N2_mean'])

    def test_squares_single_sep(self):
        countoscope.calculate_nmsd(self.data, box_sizes=(2, 4, 8), sep_sizes=2)
        
    def test_rectangles_single_x(self):
        countoscope.calculate_nmsd(self.data, box_sizes_x=(2, 4, 8), box_sizes_y=2, sep_sizes=2)
        
    def test_rectangles_single_y(self):
        countoscope.calculate_nmsd(self.data, box_sizes_x=2, box_sizes_y=(2, 4, 8), sep_sizes=2)
        
    def test_rectangles_xy(self):
        countoscope.calculate_nmsd(self.data, box_sizes_x=(2, 4, 8), box_sizes_y=(4, 8, 16), sep_sizes=2)

    def test_overlap_equivalence(self):
        # we count with overlap and then multiple times without offset so that some of the boxes line up
        # then check that the lined up box counts are the same

        res_overlapped       = countoscope.calculate_nmsd(self.data, box_sizes=[6], sep_sizes=[-2], offset_xs=[1], offset_ys=[1], return_counts=True)
        res_non_overlapped_1 = countoscope.calculate_nmsd(self.data, box_sizes=[6], sep_sizes=[ 2], offset_xs=[0], offset_ys=[0], return_counts=True)
        res_non_overlapped_2 = countoscope.calculate_nmsd(self.data, box_sizes=[6], sep_sizes=[ 2], offset_xs=[4], offset_ys=[4], return_counts=True)
        
        numpy.testing.assert_allclose(res_overlapped.counts[0, 0::2, 0::2, :], res_non_overlapped_1.counts[0, :, :, :])
        numpy.testing.assert_allclose(res_overlapped.counts[0, 1::2, 1::2, :], res_non_overlapped_2.counts[0, :, :, :])

if __name__ == '__main__':
    unittest.main()