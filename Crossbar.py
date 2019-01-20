import gl
from gl import cb_size
from gl import (w_tot, w_int)
from gl import crossbar_set
from gl import input_set
import numpy as np
class Crossbar():
    def __init__(self, cb_size = cb_size, tot_bit = w_tot, int_bit = w_int):
        self.cb_size = cb_size # CB size
        self.array = np.zeros((cb_size, cb_size)) # create 2D array for crossbar set
        self.tot_bit = tot_bit
        self.int_bit = int_bit
    def update_array(self, new_array): # take 2D decimal array
        if new_array.shape == (cb_size, cb_size):
            self.array = new_array
        else:
            new_row_cnt = new_array.shape[0]
            new_col_cnt = new_array.shape[1]
            self.array[:new_row_cnt, : new_col_cnt] = new_array
    def get_array(self):
        return self.array
    def matmul(self, input_vector):
        return np.matmul(input_vector, self.array)