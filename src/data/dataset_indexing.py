"""Module for adding an index to the test dataset for later error analysis.

This script processes the cleaned test dataset, removes the 'Usage' column,
and appends a unique index to each instance. The indexed dataset can then
be used for error analysis and visualization.
"""

import numpy as np
from utils.csv_writer import csv_writer
from data.data_cleaning import sets_cleaner
from dotenv import load_dotenv
import os
import sys

load_dotenv()

# Load the cleaned test set using the sets_cleaner function
test_set_hus = sets_cleaner(os.getenv("TEST_DATASET"))

# Create a list of numbers to be used as indices, reshaped to a column vector
# The number 1611 should ideally be dynamic based on the actual size of test_set_hus
nums = np.array(list(range(0, 1611)))
nums = np.reshape(nums, (1611, 1))

# Convert the test set to a NumPy array for easier manipulation
test_set_hus = np.array(test_set_hus)

# Delete the third column (index 2), which typically contains the 'Usage' split name
test_set_hus = np.delete(test_set_hus, 2, 1)

# Horizontally stack the test set with the new indices column
test_set_hus = np.hstack((test_set_hus, nums))

# Define the fields for the new CSV file, including the added 'Index'
fields = ["emotion", "pixels", "Index"]

# Write the indexed dataset to a new CSV file
# csv_writer("test_set_full_index.csv", fields, test_set_hus)  # creat an indexed dataset
