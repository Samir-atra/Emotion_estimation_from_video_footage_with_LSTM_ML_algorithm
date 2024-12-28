"""write data into csv files for a given filename using inputted fields and data"""

import csv


def csv_writer(filename, fields, data):
    """create a csv file and write the given contents into it"""
    csvfile = filename
    fields = fields
    with open(csvfile, mode="a", encoding="utf-8") as first:
        csvwriter = csv.writer(first)
        csvwriter.writerow(fields)
        csvwriter.writerows(data)

        return True
