import numpy as np

MORNING_MIDDAY_AFTERNOON = {
    "06:00:00": "morning",
    "10:00:00": "midday",
    "14:00:00": "afternoon"
}

SKIP_YEARS = [
    "2014",
    "2015",
    "2016",
    # "2017",
    # "2018",
    # "2019",
    # "2020",
    # "2021",
    "2022"
]

SKIP_CORONA_PERIODS = 0

START_OF_THE_DAY = "00:00:00"
END_OF_THE_DAY = "23:00:00"
MEASUREMENTS_PER_DAY = 24

FIELDNAMES = ["date",
              "year",
              "month",
              "day",
              "weekday",
              "holiday",
              "school_holiday",
              "customers_total",
              "sales",
              "temp_average",
              "clouds_all_average",
              "wind_speed_average",
              "humidity_average",
              "rain_average",
              "rain_total",
              "snow_average",
              "snow_total",
              "temp_morning",
              "weather_main_morning",
              "clouds_all_morning",
              "wind_speed_morning",
              "humidity_morning",
              "rain_morning",
              "snow_morning",
              "temp_midday",
              "weather_main_midday",
              "clouds_all_midday",
              "wind_speed_midday",
              "humidity_midday",
              "rain_midday",
              "snow_midday",
              "temp_afternoon",
              "weather_main_afternoon",
              "clouds_all_afternoon",
              "wind_speed_afternoon",
              "humidity_afternoon",
              "rain_afternoon",
              "snow_afternoon",
              "sun_hours",
              ]

SALES_FILENAME = "./data/sales-data/combined/combined-sales-data.csv"

HOLIDAY_FILES = [
    "./data/holiday-data/2017.json",
    "./data/holiday-data/2018.json",
    "./data/holiday-data/2019.json",
    "./data/holiday-data/2020.json",
    "./data/holiday-data/2021.json",
    "./data/holiday-data/2022.json",
]

HOLIDAY_FILE = "./data/holiday-data/holiday-combined.csv"
SCHOOL_HOLIDAY_FILE = "./data/schoolholiday-data/schoolholiday-combined.csv"

SCHOOL_HOLIDAY_FILES = [
    "./data/schoolholiday-data/school2017.json",
    "./data/schoolholiday-data/school2018.json",
    "./data/schoolholiday-data/school2019.json",
    "./data/schoolholiday-data/school2020.json",
    "./data/schoolholiday-data/school2021.json",
    "./data/schoolholiday-data/school2022.json",
]

WEATHER_FILE = "./data/weather-data/weather.csv"

#weather
TEMPERATURE_INDEX = 6
MAIN_WEATHER_INDEX = 25
CLOUDS_ALL_INDEX = 23
WIND_SPEED_INDEX = 16
HUMIDITY_INDEX = 15
RAIN_INDEX = 19
SNOW_INDEX = 21

PREDICT = "customers_total"

MAIN_WEATHER = {
            "Clear": 1,
            "Clouds": 2,
            "Haze": 3,
            "Drizzle": 4,
            "Rain": 5,
            "Snow": 6,
            "Mist": 7,
            "Fog": 8,
            "Thunderstorm": 9,
        }

MAIN_WEATHER_UNENCODED = {
            1: "Clear",
            2: "Clouds",
            3: "Haze",
            4: "Drizzle",
            5: "Rain",
            6: "Snow",
            7: "Mist",
            8: "Fog",
            9: "Thunderstorm",
        }

#sales
SALES_DATE_INDEX = 0
SALES_INDEX = 7
CUSTOMER_INDEX = 5

SUN_DATE_INDEX = 1
SUN_INDEX = 8

WEEKDAYS = {
    "0": "MONDAY",
    "1": "TUESDAY",
    "2": "WEDNESDAY",
    "3": "THURSDAY",
    "4": "FRIDAY",
    "5": "SATURDAY",
    "6": "SUNDAY"
}

NUMBER_OF_DAYS_PER_MONTH = {
    "1": 31,
    "2": 28,
    "3": 31,
    "4": 30,
    "5": 31,
    "6": 30,
    "7": 31,
    "8": 31,
    "9": 30,
    "10": 31,
    "11": 30,
    "12": 31
}

CATEGORICAL_COLUMNS = [
        "weather_main_morning",
        "weather_main_midday",
        "weather_main_afternoon",
        "holiday",
        "school_holiday"
    ]

CATEGORIES = [
    ["Clear", "Clouds", "Haze", "Drizzle", "Rain", "Snow", "Mist", "Fog", "Thunderstorm"],
    ["Clear", "Clouds", "Haze", "Drizzle", "Rain", "Snow", "Mist", "Fog", "Thunderstorm"],
    ["Clear", "Clouds", "Haze", "Drizzle", "Rain", "Snow", "Mist", "Fog", "Thunderstorm"],
    ["False", "True"],
    ["False", "True"],
]

CATEGORY_VALUES = [
    ["Clear", "Clouds", "Haze", "Drizzle", "Rain", "Snow", "Mist", "Fog", "Thunderstorm"],
    ["Clear", "Clouds", "Haze", "Drizzle", "Rain", "Snow", "Mist", "Fog", "Thunderstorm"],
    ["Clear", "Clouds", "Haze", "Drizzle", "Rain", "Snow", "Mist", "Fog", "Thunderstorm"],
    ["False", "True"],
    ["False", "True"],
]

RANDOM_STATE = np.random.RandomState(0)

PREVIOUS_YEAR = {
    "2016": "2015",
    "2017": "2016",
    "2018": "2017",
    "2019": "2018",
    "2020": "2019",
    "2021": "2020",
    "2022": "2021",
}

