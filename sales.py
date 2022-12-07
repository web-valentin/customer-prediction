import csv
import helperfunctions as helperfunctions
import pandas as pd
import glob
import constants as constant
from decimal import Decimal


class SalesDay:
    def __init__(self, date):
        self.date = date
        self.customer_count = 0
        self.sales = 0

    def set_customer_count(self, customer_count):
        self.customer_count = customer_count

    def set_sales(self, sales):
        self.sales = sales


def read_sales_file_and_return_dic_of_sales_days(filename):
    extension = "csv"
    all_filenames = [i for i in glob.glob("./data/sales-data/combined/sales*.{}".format(extension))]
    combined_csv = pd.concat([pd.read_csv(f, delimiter=";") for f in all_filenames])
    combined_csv.to_csv(filename, index=False, encoding="UTF8")

    sales_file = open(filename)
    sales_csv_reader = csv.reader(sales_file)

    header = []
    header = next(sales_csv_reader)

    sales_days = {}
    for row in sales_csv_reader:
        date = helperfunctions.convert_sales_date_to_weather_date(row[constant.SALES_DATE_INDEX])
        sales_day = SalesDay(date)
        sales = row[constant.SALES_INDEX] if row[constant.SALES_INDEX] != "" else "0"
        # sales = row[constant.SALES_INDEX]
        sales_day.set_sales(float(sales.replace(",", ".")))
        # sales_day.set_sales(sales)
        customers = row[constant.CUSTOMER_INDEX] if row[constant.CUSTOMER_INDEX] != "" else 0.0
        # customers = row[constant.CUSTOMER_INDEX]
        sales_day.set_customer_count(customers)
        sales_days[date] = sales_day

    sales_file.close()

    return sales_days

