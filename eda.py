import csv
import pandas as pd

import constants
import preprocessing as preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import OneHotEncoder


def exploratory_data_analysis(filename):
    dataset = open(filename)
    csv_reader = csv.reader(dataset)

    header = []
    header = next(csv_reader)
    df = pd.read_csv(filename)

    """ 
    this adds extra features customer_count last year etc 
    unfortunately it did not improve the model
    """
    """
    lag_dates = []
    lag_customer_counts = []
    lag_average_customer_count_per_weekday_months = []
    lag_average_customer_count_per_months = []

    for key, row in df.iterrows():
        date = row["date"]
        year = row["year"]
        month = row["month"]
        weekday = row["weekday"]
        previous_year = constants.PREVIOUS_YEAR[str(year)]
        lag_date = previous_year + "-" + date[5:10]
        previous_data = df[(df.date == lag_date)]
        previous_months = df[(df.month == month)]
        previous_months_weekdays = df[(df.month == month) & (df.weekday == weekday)]
        total_customers = 0.
        weekday_total_customers = 0.
        counter = 0
        weekday_counter = 0
        if previous_data.empty:
            for m_key, m_row in previous_months.iterrows():
                counter += 1
                total_customers += m_row["customers_total"]
            month_average = total_customers / counter

            for m_key, m_row in previous_months_weekdays.iterrows():
                weekday_counter += 1
                weekday_total_customers += m_row["customers_total"]
            lag_customer_count = weekday_total_customers / weekday_counter
            weekday_month_average = weekday_total_customers / weekday_counter
        else:
            lag_customer_count = previous_data.iloc[0].customers_total
            for m_key, m_row in previous_months.iterrows():
                counter += 1
                total_customers += m_row["customers_total"]
            month_average = total_customers / counter

            for m_key, m_row in previous_months_weekdays.iterrows():
                weekday_counter += 1
                weekday_total_customers += m_row["customers_total"]
            weekday_month_average = weekday_total_customers / weekday_counter

        lag_dates.append(lag_date)
        lag_customer_counts.append(lag_customer_count)
        lag_average_customer_count_per_months.append(month_average)
        lag_average_customer_count_per_weekday_months.append(weekday_month_average)

    df['lag_date'] = lag_dates
    df['lag_customer_counts'] = lag_customer_counts
    df['lag_customer_average_per_month'] = lag_average_customer_count_per_months
    df['lag_customer_average_per_weekday_month'] = lag_average_customer_count_per_weekday_months

    df = df.drop(df[df.year == 2016].index)

    df.to_csv("data/test.csv", index=False, encoding="UTF8")
    """

    """ this is the numbers of rows and columns eg 1762 rows and 38 columns"""
    print(df.shape)

    """ this shows basic info about the df as we can see we already eliminated all Nan values"""
    df.info()

    """ df.head"""
    print(df.head(10))

    """ check for missing values"""
    print(df.isnull().sum() / df.shape[0] * 100)

    """ statistic overview """
    print(df.describe())
    print(df.sales.describe())
    print(df.describe().transpose())

    """ histogram to see the distribution """
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.displot(df, x="customers_total", kde="True")
    sns.displot(df, x="customers_total", hue="year", kde="True")

    # you can clearly see the ammount of null values
    plt.figure(figsize=(9, 8))
    sns.distplot(df['customers_total'], color='g', bins=100, hist_kws={'alpha': 0.4})

    """ show histogram for all features """
    # df_num = df.select_dtypes(include=['float64', 'int64'])
    # df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
    plt.show()

    """ check which features are highly correlated with customers_total """
    # removing sundays increases correlation between weekday and customers_total
    # df = preprocessing.remove_sundays(df)

    df_corr = df.corr()['customers_total']
    print("All correlations:")
    print(df_corr.sort_values(ascending=False))

    features_with_correlation = df_corr[abs(df_corr) > 0.3].sort_values(ascending=False)
    print("There is {} strongly correlated values with customers_total:\n{}".format(len(features_with_correlation),
                                                                             features_with_correlation))

    """ plot
    we can see a seasonal trend when watching the months
    we can see a upwards trend when watching the weekdays
    we seem to have a holiday where there were customers
    we can see that there are less customers during school_holidays
    sales: clear linear_relation but some outliers
    sales: outliers days where customers_total > 0 but sales == 0 => remove rows 
    rain and wind show: more wind / rain tends to have a negative correlation with customer_count
    weather main needs to be updated
    feiertage removen?
    "Rain": 5, "Snow": 6, "Mist": 7, "Fog": 8, shows negative correlation
    """
    df = df.drop(df[(df.sales == 0) & (df.holiday == 1)].index)

    for i in range(0, len(df.columns), 3):
       sns.pairplot(data=df, x_vars=df.columns[i:i + 3], y_vars=['customers_total'])

    plt.show()

    """ remove outliers in sales """
    # print(df[(df.sales == 0) & (df.customers_total > 0)].date)
    """ the removing part will be done in model.py"""
    # df = preprocessing.remove_days_without_sales_but_customers(df)

    """ check for sales to find outliers """
    # for i in range(0, len(df.columns), 5):
    #    sns.pairplot(data=df, x_vars=df.columns[i:i + 5], y_vars=['sales'])

    # plt.show()

    """ correlation matrix
    we can see obvious correlation in weather features => temp_morning and temp_total etc
    reduce features?
    """
    # corr_matrix = df.corr()
    # plt.figure(figsize=(12, 10))

    # sns.heatmap(corr_matrix[(corr_matrix >= 0.3) | (corr_matrix <= -0.3)],
    #             cmap="coolwarm", vmax=1.0, vmin=-1.0, linewidths=0.1,
    #            annot=True, annot_kws={"size": 8}, square=True)

    # sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    # plt.show()

    """ boxplot waether_main """
    # plt.figure(figsize=(12, 6))
    # ax = sns.boxplot(x='weather_main_morning', y='customers_total', data=df)
    # plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")
    # plt.xticks(rotation=45)
    # plt.show()

    """ boxplot """
    # sns.boxplot(x=df["customers_total"])F
    # plt.show()


    # dataset.close()

    return df


def exploratory_data_analysis_simple(filename):
    dataset = open(filename)
    csv_reader = csv.reader(dataset)

    header = []
    header = next(csv_reader)
    df = pd.read_csv(filename)

    """ this is the numbers of rows and columns eg 1762 rows and 38 columns"""
    print(df.shape)

    """ this shows basic info about the df as we can see we already eliminated all Nan values"""
    df.info()

    """ df.head"""
    print(df.head(10))
    print(df.index)
