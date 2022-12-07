import csv
import pandas as pd

import sun
import weather as weather
import sales as sale
import holiday as holiday
import schoolholiday as school_holiday
import helperfunctions as helperfunctions
import constants as constant


def create_dataset_and_return_filename():
    sales_days = sale.read_sales_file_and_return_dic_of_sales_days(constant.SALES_FILENAME)
    holiday_days = holiday.read_holiday_files_and_return_dic_of_holidays_days(constant.HOLIDAY_FILES)
    school_holiday_days = school_holiday.read_school_holiday_files_and_return_dic_of_school_holiday_days(
        constant.SCHOOL_HOLIDAY_FILES)
    sun_days = sun.read_sun_hours_and_return_dic_of_sunhours()

    weather_file = open(constant.WEATHER_FILE)
    weather_csv_reader = csv.reader(weather_file)

    header = []
    header = next(weather_csv_reader)
    weather_data = pd.read_csv("./data/weather-data/weather.csv")

    fieldnames = constant.FIELDNAMES

    rows = []
    day = weather.WeatherDay()
    for row in weather_csv_reader:
        date_and_time = row[1]
        date = date_and_time[0:10]
        time = date_and_time[11:19]

        day_as_number = int(helperfunctions.get_day_from_date(date))
        month_as_number = int(helperfunctions.get_month_from_date(date))
        year = helperfunctions.get_year_from_date(date)
        year_as_number = int(year)

        if year in constant.SKIP_YEARS:
            continue

        if time == constant.START_OF_THE_DAY:
            day.date = date

        if time in constant.MORNING_MIDDAY_AFTERNOON:
            daytime = constant.MORNING_MIDDAY_AFTERNOON[time]
            day.temperature[daytime] = float(row[constant.TEMPERATURE_INDEX])
            day.weather_main[daytime] = day.get_weather_as_number(row[constant.MAIN_WEATHER_INDEX], daytime)
            day.clouds_all[daytime] = float(row[constant.CLOUDS_ALL_INDEX])
            day.wind_speed[daytime] = float(row[constant.WIND_SPEED_INDEX])
            day.humidity[daytime] = float(row[constant.HUMIDITY_INDEX])
            day.rain[daytime] = float(row[constant.RAIN_INDEX]) if row[constant.RAIN_INDEX] != "" else 0.0
            day.snow[daytime] = float(row[constant.SNOW_INDEX]) if row[constant.SNOW_INDEX] != "" else 0.0

        day.temperature["total"] += float(row[constant.TEMPERATURE_INDEX])
        day.clouds_all["total"] += float(row[constant.CLOUDS_ALL_INDEX])
        day.wind_speed["total"] += float(row[constant.WIND_SPEED_INDEX])
        day.humidity["total"] += float(row[constant.HUMIDITY_INDEX])
        day.rain["total"] += float(row[constant.RAIN_INDEX]) if row[constant.RAIN_INDEX] != "" else 0.0
        day.snow["total"] += float(row[constant.SNOW_INDEX]) if row[constant.SNOW_INDEX] != "" else 0.0

        if time == constant.END_OF_THE_DAY:
            day.temperature["average"] = round(day.temperature["total"] / constant.MEASUREMENTS_PER_DAY, 2)
            day.clouds_all["average"] = round(day.clouds_all["total"] / constant.MEASUREMENTS_PER_DAY, 2)
            day.wind_speed["average"] = round(day.wind_speed["total"] / constant.MEASUREMENTS_PER_DAY, 2)
            day.humidity["average"] = round(day.humidity["total"] / constant.MEASUREMENTS_PER_DAY, 2)
            day.rain["average"] = round(day.rain["total"] / constant.MEASUREMENTS_PER_DAY, 2)
            day.snow["average"] = round(day.snow["total"] / constant.MEASUREMENTS_PER_DAY, 2)

            weekday = helperfunctions.get_weekday_from_date(day.date)

            data = {
                "date": day.date,
                "year": year_as_number,
                "month": month_as_number,
                "day": day_as_number,
                "weekday": weekday,
                "holiday": '0',
                "school_holiday": '0',
                "customers_total": "",
                "sales": "",
                "temp_average": day.temperature["average"],
                "clouds_all_average": day.clouds_all["average"],
                "wind_speed_average": day.wind_speed["average"],
                "humidity_average": day.humidity["average"],
                "rain_average": day.rain["average"],
                "rain_total": round(day.rain["total"], 2),
                "snow_average": day.snow["average"],
                "snow_total": round(day.snow["total"], 2)
            }

            if day.date in school_holiday_days:
                data["school_holiday"] = "1"

            if day.date in holiday_days:
                data["holiday"] = "1"

            if day.date in sales_days:
                data["sales"] = sales_days[day.date].sales
                data["customers_total"] = sales_days[day.date].customer_count

            if day.date in sun_days:
                data["sun_hours"] = sun_days[day.date].sunhours

            for key in constant.MORNING_MIDDAY_AFTERNOON:
                value = constant.MORNING_MIDDAY_AFTERNOON[key]
                data["temp_" + value] = day.temperature[value]
                data["weather_main_" + value] = day.weather_main[value]
                data["clouds_all_" + value] = day.clouds_all[value]
                data["wind_speed_" + value] = day.wind_speed[value]
                data["humidity_" + value] = day.humidity[value]
                data["rain_" + value] = day.rain[value]
                data["snow_" + value] = day.snow[value]


            if day.date not in sales_days:
                day.reset_variables()
                continue

            rows.append(data)
            day.reset_variables()

    weather_file.close()

    with open("./data/dataset/data-combined.csv", "w", encoding="UTF8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return "./data/dataset/data-combined.csv"
