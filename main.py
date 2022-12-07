import csv

import pandas as pd

import dataimport as dataimport
import model
import eda as eda
import constants as constants

""" this step combines all the data => do not delete"""
filename = dataimport.create_dataset_and_return_filename()

""" this steps visualize shape of single df can be set as comment"""
eda.exploratory_data_analysis_simple(constants.HOLIDAY_FILE)
eda.exploratory_data_analysis_simple(constants.SCHOOL_HOLIDAY_FILE)
eda.exploratory_data_analysis_simple(constants.SALES_FILENAME)
eda.exploratory_data_analysis_simple(constants.WEATHER_FILE)

df = eda.exploratory_data_analysis(filename)

""" if you dont want to visualize eda => use this to get the df
dataset = open(filename)
csv_reader = csv.reader(dataset)
header = []
header = next(csv_reader)
df = pd.read_csv(filename)
"""

model.create_model(df)
