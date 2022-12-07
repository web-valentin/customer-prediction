import datetime as dt


def convert_sales_date_to_weather_date(sales_date):
    year = sales_date[6:10]
    month = sales_date[3:5]
    day = sales_date[0:2]
    date = year + "-" + month + "-" + day
    return date


def get_year_from_date(date):
    return date[0:4]


def get_month_from_date(date):
    return date[5:7]


def get_day_from_date(date):
    return date[8:10]


def get_weekday_from_date(date):
    only_year = get_year_from_date(date)
    only_month = get_month_from_date(date)
    only_day = get_day_from_date(date)

    weekday = dt.datetime(int(only_year), int(only_month), int(only_day)).weekday()
    return weekday


def convert_sun_date_to_weather_date(sun_date):
    year = sun_date[0:4]
    month = sun_date[4:6]
    day = sun_date[6:8]
    date = year + "-" + month + "-" + day
    return date

