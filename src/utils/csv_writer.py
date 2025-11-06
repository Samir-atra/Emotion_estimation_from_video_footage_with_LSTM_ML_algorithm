"""Module for writing data to CSV files.

This module provides a utility function to create or append data to a CSV file,
including writing a header row and multiple data rows.
"""

import csv


def csv_writer(filename, fields, data):
    """
    Creates a CSV file (or appends to an existing one) and writes the given contents into it.

    The function opens the specified file in append mode (`'a'`), writes a header row
    using the `fields` list, and then writes all rows from the `data` list.

    Args:
        filename (str): The name of the CSV file to create or append to.
        fields (list): A list of strings representing the header row of the CSV file.
        data (list): A list of lists, where each inner list represents a row of data
                     to be written to the CSV file.

    Returns:
        bool: True if the data was successfully written to the CSV file.
    """
    csvfile = filename # Assign the filename to a local variable
    header_fields = fields # Assign the header fields to a local variable
    
    # Open the CSV file in append mode with UTF-8 encoding
    with open(csvfile, mode="a", encoding="utf-8") as file_obj:
        csvwriter = csv.writer(file_obj)
        csvwriter.writerow(header_fields) # Write the header row to the CSV file
        csvwriter.writerows(data)        # Write all data rows to the CSV file

        return True
