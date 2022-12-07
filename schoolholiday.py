import json
import csv


class SchoolHolidayDay:

    def __init__(self, date):
        self.date = date


def read_school_holiday_files_and_return_dic_of_school_holiday_days(files):
    school_holiday_days = {}
    for file in files:
        school_holiday_file = open(file)
        data = json.load(school_holiday_file)

        for element in data:
            start = element["start"]
            start = start[0:10]
            start_year = int(start[0:4])
            start_month = int(start[5:7])
            start_day = int(start[8:10])

            end = element["end"]
            end = end[0:10]
            end_year = int(end[0:4])
            end_month = int(end[5:7])
            end_day = int(end[8:10])

            while start_day <= end_day:
                date = convert_to_date_string(start_year, start_month, start_day)
                school_holiday_days[date] = SchoolHolidayDay(date)
                start_day += 1

            if start_month < end_month:
                while start_month < end_month:
                    last_day = 32
                    while start_day <= last_day:
                        date = convert_to_date_string(start_year, start_month, start_day)
                        school_holiday_days[date] = SchoolHolidayDay(date)
                        if start_day == 31:
                            start_day = 0
                            start_month += 1
                            if start_month == end_month:
                                last_day = end_day
                        start_day += 1

            if start_year < end_year:
                while start_year < end_year:
                    last_month = 13
                    while start_month < last_month:
                        last_day = 32
                        while start_day <= last_day:
                            date = convert_to_date_string(start_year, start_month, start_day)
                            school_holiday_days[date] = SchoolHolidayDay(date)
                            if start_day == 31:
                                start_day = 0
                                if start_month == 12:
                                    start_month = 0
                                start_month += 1
                                start_year += 1

                                if start_month == end_month:
                                    last_day = end_day
                                    last_month = end_month
                            start_day += 1

    fieldnames = ["date", "school_holiday"]
    holiday_data = []
    for day in school_holiday_days:
        current_data = {
            "date": day,
            "school_holiday": True,
        }
        holiday_data.append(current_data)

    with open("./data/schoolholiday-data/schoolholiday-combined.csv", "w", encoding="UTF8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(holiday_data)
    return school_holiday_days


def convert_to_date_string(year, month, day):
    day_string = "0" + str(day) if day < 10 else str(day)
    month_string = "0" + str(month) if month < 10 else str(month)
    year_string = str(year)

    return year_string + "-" + month_string + "-" + day_string
