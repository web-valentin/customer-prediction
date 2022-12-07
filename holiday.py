import json
import csv


class HolidayDay:
    def __init__(self, date):
        self.date = date


def read_holiday_files_and_return_dic_of_holidays_days(files):
    holiday_days = {}
    for file in files:
        holiday_file = open(file)
        data = json.load(holiday_file)

        for element in data:
            single_holiday = data[element]
            date = single_holiday["datum"]
            holiday_days[date] = HolidayDay(date)

    fieldnames = ["date", "holiday"]
    holiday_data = []
    for day in holiday_days:
        current_data = {
            "date": day,
            "holiday": True,
        }
        holiday_data.append(current_data)

    with open("./data/holiday-data/holiday-combined.csv", "w", encoding="UTF8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(holiday_data)

    return holiday_days
