import csv
import helperfunctions as helperfunctions
import pandas as pd
import glob
import constants as constant
from decimal import Decimal


class SunDay:
    def __init__(self, date):
        self.date = date
        self.sunhours = 0

    def set_sunhours(self, sunhours):
        self.sunhours = sunhours


def read_sun_hours_and_return_dic_of_sunhours():
    clima_old_file = open("data/weather-data/old_klima.csv")
    clima_new_file = open("data/weather-data/new_clima.csv")

    files = {
        "data/weather-data/old_klima.csv": clima_old_file,
        "data/weather-data/new_clima.csv": clima_new_file
    }

    sun_days = {}
    for filename, file in files.items():
        csv_reader = csv.reader(file, delimiter=";")
        header = []
        header = next(csv_reader)
        test_data = pd.read_csv(filename, delimiter=";")

        for row in csv_reader:
            date = helperfunctions.convert_sun_date_to_weather_date(row[constant.SUN_DATE_INDEX])
            sun_day = SunDay(date)
            sun = row[constant.SUN_INDEX] if row[constant.SUN_INDEX] != "" else "0"
            # sales = row[constant.SALES_INDEX]
            sun_day.set_sunhours(float(sun.replace(".", "")) / 1000)
            sun_days[date] = sun_day

    clima_new_file.close()
    clima_old_file.close()

    return sun_days

