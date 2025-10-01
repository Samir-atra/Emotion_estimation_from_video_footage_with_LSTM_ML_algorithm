"""
add an index to the dataset to be processed and analyzed later for errors
"""

import numpy as np
from utils.csv_writer import csv_writer
from data.data_cleaning import sets_cleaner
from dotenv import load_dotenv
import os
import sys

load_dotenv()

test_set_hus = sets_cleaner(os.getenv("TEST_DATASET"))

nums = np.array(
    list(range(0, 1611))
)  # create a list of numbers to be appended as indices
nums = np.reshape(nums, (1611, 1))
test_set_hus = np.array(test_set_hus)
test_set_hus = np.delete(
    test_set_hus, 2, 1
)  # delete the third column which contains the name of the split
test_set_hus = np.hstack((test_set_hus, nums))  # add the indices column
fields = ["emotion", "pixels", "Index"]
# csv_writer("test_set_full_index.csv", fields, test_set_hus)  # creat an indexed dataset
