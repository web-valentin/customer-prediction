import numpy as np
import plotly.graph_objects as go
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import FunctionTransformer
import constants


def remove_sundays(df):
    df = df.drop(df[df.weekday == 6].index)
    return df


def remove_outliers(df):
    df = remove_sundays(df)
    print(df[(df.customers_total == 0.0)].holiday)
    return df


def remove_days_without_sales_but_customers(df):
    df = df.drop(df[(df.sales == 0) & (df.customers_total > 0)].index)
    return df


def perform_sin_cos_encoding(df):
    """visualization"""
    # plt.show()
    # days = df["day"];
    # days.plot(y='A', use_index=True)
    # plt.ylabel("Days in month")
    # plt.xlabel("Days in year")
    # plt.show()

    """ this part is to add the transformed data to the df """
    df["weekday_sin"] = np.sin((df.weekday * 2. * np.pi) / 7)
    df["weekday_cos"] = np.cos((df.weekday * 2. * np.pi) / 7)

    df["month_sin"] = np.sin((df.month * 2. * np.pi) / 12)
    df["month_cos"] = np.cos((df.month * 2. * np.pi) / 12)

    """ this visualizes the difference """
    # df["day_sin_test"] = np.sin((df.day * 2. * np.pi) / 31)
    # df["day_cos_test"] = np.cos((df.day * 2. * np.pi) / 31)

    """ months do not have the same length => adjust accordingly """
    day_sin = []
    day_cos = []
    for index, row in df.iterrows():
        days_per_month = constants.NUMBER_OF_DAYS_PER_MONTH[str(row.month)]
        day_sin.append(np.sin((row.day * 2. * np.pi) / days_per_month))
        day_cos.append(np.cos((row.day * 2. * np.pi) / days_per_month))

    df["day_sin"] = day_sin
    df["day_cos"] = day_cos

    """ visualization """
    # Set the figure size in inches
    plt.figure(figsize=(10, 6))

    # plt.scatter(df["day_sin"], df["day_cos"])
    # plt.title("Plot for sin and cos transformation of feature day")
    # plt.xlabel("cos_x")
    # plt.ylabel("sin_x")
    # plt.show()

    df_vis = df.copy()

    # df_vis["x_norm"] = 2 * math.pi * df_vis["day"] / (df_vis["day"].max())
    # df_vis["cos_x"] = np.cos(df_vis["x_norm"])
    # df_vis["sin_x"] = np.sin(df_vis["x_norm"])

    # plt.title("Plot for sin and cos of feature day")
    # plt.plot(df_vis.x_norm, df_vis.sin_x, df_vis.x_norm, df_vis.cos_x)
    # plt.legend(['sin(x)', 'cos(x)'])

    # plt.show()

    """ this visualizes the difference => same for months with 31 days but diffrent for february """
    # test = df[["day", "day_sin_test", "day_cos_test", "day_sin", "day_cos"]].copy()
    # print(test.head(50))

    # plt.show()

    return df


def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))


def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))


def remove_small_categories(df):
    # print(df["weather_main_morning"].value_counts())
    # print(df["weather_main_midday"].value_counts())
    # print(df["weather_main_afternoon"].value_counts())

    # replace thunderstorm with rain
    df["weather_main_morning"].replace(to_replace=9, value=5, inplace=True)
    df["weather_main_midday"].replace(to_replace=9, value=5, inplace=True)
    df["weather_main_afternoon"].replace(to_replace=9, value=5, inplace=True)

    # replace haze with clear
    df["weather_main_morning"].replace(to_replace=3, value=1, inplace=True)
    df["weather_main_midday"].replace(to_replace=3, value=1, inplace=True)
    df["weather_main_afternoon"].replace(to_replace=3, value=1, inplace=True)

    # replace drizzle with rain
    df["weather_main_morning"].replace(to_replace=4, value=5, inplace=True)
    df["weather_main_midday"].replace(to_replace=4, value=5, inplace=True)
    df["weather_main_afternoon"].replace(to_replace=4, value=5, inplace=True)

    # replace fog with mist
    df["weather_main_morning"].replace(to_replace=8, value=7, inplace=True)
    df["weather_main_midday"].replace(to_replace=8, value=7, inplace=True)
    df["weather_main_afternoon"].replace(to_replace=8, value=7, inplace=True)
    return df


def transform_weather_back(df):
    for x in range(1, 10):
        df["weather_main_morning"].replace(to_replace=x, value=constants.MAIN_WEATHER_UNENCODED[x], inplace=True)
        df["weather_main_midday"].replace(to_replace=x, value=constants.MAIN_WEATHER_UNENCODED[x], inplace=True)
        df["weather_main_afternoon"].replace(to_replace=x, value=constants.MAIN_WEATHER_UNENCODED[x], inplace=True)

    return df


def transform_holiday_back(df):
    df["holiday"].replace(to_replace=1, value="True", inplace=True)
    df["holiday"].replace(to_replace=0, value="False", inplace=True)

    df["school_holiday"].replace(to_replace=1, value="True", inplace=True)
    df["school_holiday"].replace(to_replace=0, value="False", inplace=True)

    return df
