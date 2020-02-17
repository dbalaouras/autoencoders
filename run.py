#!/usr/bin/env python
import csv
import os
import random
from optparse import OptionParser

__author__ = "Dimi Balaouras"
__copyright__ = "Copyright 2020, uth.gr"
__license__ = "Apache License 2.0, see LICENSE for more details."
__version__ = "0.0.1"
__description__ = ""
__abs_dirpath__ = os.path.dirname(os.path.abspath(__file__))


def load_icd10_file():
    """
    """
    with open(cli_options.icd_file, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile, delimiter=',')
        # extracting each data row one by one
        icd10_codes = [row[1] for row in csvreader]

    return icd10_codes


def create_dataset():
    """
    Create a artificial PHR dataset
    """
    # erase cli_options.features_export_file
    open(cli_options.features_export_file, 'w').close()

    icd10_codes = load_icd10_file()
    age_range = range(1, 100)
    genders = [0, 1]
    random.seed(1)
    for i in range(cli_options.size_of_phr):
        age = random.choice(age_range)
        gender = random.choice(genders)
        idc10_sampled_codes = random.sample(icd10_codes,
                                            random.randint(cli_options.min_cond_num, cli_options.max_cond_num))
        features_list = [0] * len(icd10_codes)
        for idc10_code in idc10_sampled_codes:
            index = icd10_codes.index(idc10_code)
            features_list[index] = 1

        features_list.insert(0, gender)
        features_list.insert(0, age)

        with open(cli_options.features_export_file, 'a') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_NONE)
            wr.writerow(features_list)


def save_dataset_csv(dataset):
    """

    :param dataset:
    :return:
    """


def run():
    """
    Runs the various exercises
    :return: None
    """
    create_dataset()


def parse_options():
    """
    Parses all options passed via the command line.
    """

    # Define version and usage strings
    usage = "python %prog [options]"

    # Initiate the cli options parser
    parser = OptionParser(usage=usage, version=__version__, description=__description__)

    # Define available command line options

    parser.add_option("-d", "--icd_file", action="store", dest="icd_file", default="icd10.csv",
                      help="Path to icd10 data csv file")

    parser.add_option("-f", "--features_export_file", action="store", dest="features_export_file",
                      default="features_export_file.csv",
                      help="Filename of the exported random feature file")

    parser.add_option("-s", "--size_of_phr", action="store", dest="size_of_phr", default=1000,
                      type="int", help="Number of PHRs to generate")

    parser.add_option("--min_cond_num", action="store", dest="min_cond_num", default=3,
                      type="int", help="Minimum number of conditions per EHR")

    parser.add_option("--max_cond_num", action="store", dest="max_cond_num", default=5,
                      type="int", help="Maximum number of conditions per EHR")

    # Parse the options
    (options, args) = parser.parse_args()

    return options, args


if __name__ == '__main__':
    """
    Entry point of this module. This method will be invoked first when running the app.
    """

    # Parse command line options
    (cli_options, cli_args) = parse_options()

    print("\n")

    # Run the app
    run()
