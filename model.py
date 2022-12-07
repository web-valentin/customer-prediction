import csv

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.kernel_approximation import Nystroem
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline, FeatureUnion

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer, SplineTransformer, \
    PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, RANSACRegressor, TheilSenRegressor, HuberRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, AdaBoostRegressor, \
    GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import KFold, cross_val_score, train_test_split, RepeatedKFold, cross_validate, \
    TimeSeriesSplit, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit, GroupKFold, GroupShuffleSplit
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import joblib
from sklearn.metrics import mean_absolute_percentage_error

import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import LSTM

import dill
import pickle

from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
import math

import constants
import preprocessing


def create_model(df):
    df_model = df.copy()

    """ data preprocessing removing sundays does not increase the score """
    # df_model = preprocessing.remove_sundays(df_model)

    """ to perform crossvalidation replace small categories, perdorm sin_cos, and remove data with errors"""
    df_model = preprocessing.remove_small_categories(df_model)
    df_model = preprocessing.perform_sin_cos_encoding(df_model)
    df_model = preprocessing.remove_days_without_sales_but_customers(df_model)

    """ get the best performing model 
        this function loops through multiple piplines,
        estimators, cv methods, and scalers
    """
    compare_multiple(df_model)

    """ tune the best performing model
        performs random and grid search
        takes a lot of time
     """
    tune_tree_models(df_model)

    create_final_model(df_model)


def create_final_model(df_model):
    df = df_model.copy()
    df = preprocessing.transform_weather_back(df)
    df = preprocessing.transform_holiday_back(df)

    mae_total = 0.
    counter = 0
    declared_variation = 0.
    total_variation = 0.
    y_mean = 0.
    rmse_total = 0.

    x_real = []
    y_real = []

    x_predictions = []
    y_predictions = []

    for i in range(150, 0, -1):
        place = i - 1
        df_new = df.tail(i)
        df_test = df_new.copy()
        df_test.drop(df_test.tail(place).index, inplace=True)
        y_new = df_test[constants.PREDICT]
        X_new = df_test.drop(["date", "sales", constants.PREDICT], axis=1)

        df_model = df.copy()
        df_model.drop(df.tail(i).index, inplace=True)

        y = df_model[constants.PREDICT]
        X = df_model.drop(["date", "sales", constants.PREDICT], axis=1)

        y_mean = y.mean()

        estimator = XGBRegressor(
            tree_method="exact",
            subsample=0.8,
            n_estimators=500,
            max_depth=90,
            learning_rate=0.04,
            colsample_bytree=0.89,
            colsample_bylevel=0.5
        )
        """ to check results without hyperparamter optimization just use this estimator """
        # XGBRegressor()

        scaler = StandardScaler()

        pipeline, X, y = get_spline_pipeline(estimator, X, y, scaler)
        pipeline.fit(X, y)
        y_predict = pipeline.predict(X_new)
        mae_total += mean_absolute_error(y_new, y_predict)
        x_real.append(counter)
        y_real.append(y_new.iloc[0])

        x_predictions.append(counter)
        y_predictions.append(y_predict)

        current = (y_predict - y_mean) ** 2
        declared_variation += current

        current_2 = (y_new.iloc[0] - y_mean) ** 2
        total_variation += current_2

        rmse = (y_predict - y_new.iloc[0]) ** 2
        rmse_total += rmse

        print(str(counter))
        counter += 1

    # r2 = declared_variation / total_variation

    z = 0.
    c = 0
    n = 0.
    for y_exp in y_real:
        z += (y_predictions[c] - y_exp)**2
        n += (y_exp - y_mean)**2
        c += 1

    r2 = 1 - (z / n)

    mae_result = mae_total / counter

    fieldnames = [
        "algorithm",
        "mae",
        "rmse",
        "r2"
    ]

    rows = [
        {
            "algorithm": "XGBoostingRegressor()",
            "mae": mae_result,
            "rmse": math.sqrt(rmse_total / counter),
            "r2": r2
        }
    ]

    with open("./data/final/final-modell-results.csv", "w", encoding="UTF8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    plt.plot(x_real, y_real)
    plt.plot(x_predictions, y_predictions)
    plt.legend(["Expected Results", "Predicted Results"])
    plt.show()


def tune_tree_models(df_model):
    df = df_model.copy()
    df = preprocessing.transform_weather_back(df)
    df = preprocessing.transform_holiday_back(df)

    y = df[constants.PREDICT]
    X = df.drop(["date", "sales", constants.PREDICT], axis=1)

    # ShuffleSplit(n_splits=number_of_splits, random_state=rng),
    # RepeatedKFold(n_splits=number_of_splits, n_repeats=3, random_state=rng)
    # TimeSeriesSplit(n_splits=number_of_splits),
    cv = cross_validation(ShuffleSplit(n_splits=3, random_state=constants.RANDOM_STATE), X, y)

    fieldnames = [
        "algorithm",
        "pipeline",
        "best_params",
    ]

    params = [
        {
            'xgbregressor__tree_method': ["hist", "exact", "approx"],
            'xgbregressor__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
            'xgbregressor__n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
            'xgbregressor__learning_rate': [0.02, 0.05, 0.07, 0.1, 0.2],
            'xgbregressor__subsample': np.arange(0.5, 1.0, 0.1),
            'xgbregressor__colsample_bytree': np.arange(0.4, 1.0, 0.1),
            'xgbregressor__colsample_bylevel': np.arange(0.4, 1.0, 0.1),
        }

    ]

    algorithms = {
        "0": XGBRegressor(),
    }

    """ perform random search first to get an overview of what good params could be """
    rows = []
    for key, algorithm in algorithms.items():
        spline_pipeline, X, y = get_spline_pipeline(algorithm, X, y)
        polynomial_pipeline, X, y = get_polynomial_pipeline(algorithm, X, y)
        ordinal_pipeline, X, y = get_ordinal_encoding_pipeline(algorithm, X, y)

        pipelines = [
            spline_pipeline,
            polynomial_pipeline,
            ordinal_pipeline,
        ]

        for pipeline in pipelines:
            param_grid = params[int(key)]

            random_search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_grid,
                n_iter=100,
                cv=cv,
                random_state=constants.RANDOM_STATE,
                n_jobs=-1
            )
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            random_search.fit(X_train, y_train)
            print(random_search.best_params_)
            data = {
                "algorithm": str(algorithm),
                "pipeline": str(pipeline),
                "best_params": random_search.best_params_,
            }
            rows.append(data)

    with open("./data/tuning/random-search-combined.csv", "w", encoding="UTF8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    """ use the best results for gridsearch """
    params = [
        {
            'xgbregressor__tree_method': ["exact"],
            'xgbregressor__max_depth': [90, 100, 110],
            'xgbregressor__n_estimators': [500, 600, 700],
            'xgbregressor__learning_rate': [0.04, 0.05],
            'xgbregressor__subsample': [0.8],
            'xgbregressor__colsample_bytree': [0.89],
            'xgbregressor__colsample_bylevel': [0.4, 0.5]
        },
    ]

    algorithms = {
        "0": XGBRegressor(),
    }

    rows = []
    for key, algorithm in algorithms.items():
        spline_pipeline, X, y = get_spline_pipeline(algorithm, X, y)
        # polynomial_pipeline, X, y = get_polynomial_pipeline(algorithm, X, y)
        # ordinal_pipeline, X, y = get_ordinal_encoding_pipeline(algorithm, X, y)

        pipelines = [
            spline_pipeline,
            # polynomial_pipeline,
            # ordinal_pipeline,
        ]
        counter = 0
        for pipeline in pipelines:
            param_grid = params[counter]
            counter += 1

            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=cv,
                n_jobs=-1
            )
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            grid_search.fit(X_train, y_train)
            print(grid_search.best_params_)

            data = {
                "algorithm": str(algorithm),
                "pipeline": str(pipeline),
                "best_params": grid_search.best_params_,
            }
            rows.append(data)

    with open("./data/tuning/grid-search-combined.csv", "w", encoding="UTF8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compare_multiple(df_model):
    """
    transform weather and holiday back to original string values
    perform one hot encoding instead of label encoding
    """
    df = df_model.copy()
    df = preprocessing.transform_weather_back(df)
    df = preprocessing.transform_holiday_back(df)

    y = df[constants.PREDICT]
    X = df.drop(["date", "sales", constants.PREDICT], axis=1)

    algorithms = [
        LinearRegression(),
        RandomForestRegressor(),
        GradientBoostingRegressor(),
        XGBRegressor(),
    ]

    fieldnames = [
        "algorithm",
        "feature_engineering",
        "cv_method",
        "scaler",
        "mean_absolute_error",
        "root_mean_squared_error",
        "mean_absolute_percentage_error",
        "r2"
    ]

    rows = []
    scalers = [StandardScaler()]

    for algo in algorithms:
        for scaler in scalers:
            cur_score, cur_method, mae, rmse, std, pipeline, cv, mape = pipeline_oridinal_encoding(algo, X, y, scaler)
            data = {
                "algorithm": str(algo),
                "feature_engineering": "oridinal",
                "cv_method": cv,
                "scaler": str(scaler),
                "mean_absolute_error": mae,
                "root_mean_squared_error": rmse,
                "mean_absolute_percentage_error": mape,
                "r2": cur_score
            }
            rows.append(data)

            cur_score, cur_method, mae, rmse, std, pipeline, cv, mape = pipeline_one_hot_encoding(algo, X, y, scaler)
            data = {
                "algorithm": str(algo),
                "feature_engineering": "one-hot-encoding",
                "cv_method": cv,
                "scaler": str(scaler),
                "mean_absolute_error": mae,
                "root_mean_squared_error": rmse,
                "mean_absolute_percentage_error": mape,
                "r2": cur_score
            }
            rows.append(data)

            cur_score, cur_method, mae, rmse, std, pipeline, cv, mape = pipeline_one_hot_encoding_time_data(algo, X, y,
                                                                                                            scaler)
            data = {
                "algorithm": str(algo),
                "feature_engineering": "one-hot-encoding timedata",
                "cv_method": cv,
                "scaler": str(scaler),
                "mean_absolute_error": mae,
                "root_mean_squared_error": rmse,
                "mean_absolute_percentage_error": mape,
                "r2": cur_score
            }
            rows.append(data)

            cur_score, cur_method, mae, rmse, std, pipeline, cv, mape = pipeline_spline_encoding_time_data(algo, X, y,
                                                                                                           scaler)
            data = {
                "algorithm": str(algo),
                "feature_engineering": "spline encoding",
                "cv_method": cv,
                "scaler": str(scaler),
                "mean_absolute_error": mae,
                "root_mean_squared_error": rmse,
                "mean_absolute_percentage_error": mape,
                "r2": cur_score
            }
            rows.append(data)

            cur_score, cur_method, mae, rmse, std, pipeline, cv, mape = pipeline_polynomial_encoding(algo, X, y, scaler)
            data = {
                "algorithm": str(algo),
                "feature_engineering": "polynomial encoding",
                "cv_method": cv,
                "scaler": str(scaler),
                "mean_absolute_error": mae,
                "root_mean_squared_error": rmse,
                "mean_absolute_percentage_error": mape,
                "r2": cur_score
            }

            rows.append(data)

    with open("./data/pipeline/algo-combined.csv", "w", encoding="UTF8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(model, X, y):
    global best_pipeline, best_cv
    rng = constants.RANDOM_STATE
    number_of_splits = 5
    cv_methods = [
        KFold(n_splits=number_of_splits),
        ShuffleSplit(n_splits=number_of_splits, random_state=rng),
        TimeSeriesSplit(n_splits=number_of_splits),
        RepeatedKFold(n_splits=number_of_splits, n_repeats=3, random_state=rng)
    ]

    best_score = 0.0
    best_method = ""
    best_mae = 0.
    best_rmse = 0.
    best_std = 0.
    best_r2 = 0.
    best_mape = 0.
    for cv_method in cv_methods:
        cv = cross_validation(cv_method, X, y)
        cv_results = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=[
                "neg_mean_absolute_error",
                "neg_root_mean_squared_error",
                "r2",
                "neg_mean_absolute_percentage_error"
            ],
        )

        mae = -cv_results["test_neg_mean_absolute_error"]
        rmse = -cv_results["test_neg_root_mean_squared_error"]
        r2 = cv_results["test_r2"]
        mape = -cv_results["test_neg_mean_absolute_percentage_error"]

        if r2.mean() > best_score:
            best_score = r2.mean()
            best_method = str(cv_method)[0:20]
            best_mae = mae.mean()
            best_rmse = rmse.mean()
            best_mape = mape.mean()
            best_pipeline = model
            best_cv = cv

    best_pipeline.fit(X, y)

    return best_score, best_method, best_mae, best_rmse, best_std, best_pipeline, best_cv, best_mape


def cross_validation(method, X, y):
    ts_cv = method
    ts_cv.split(X, y)
    return ts_cv


def pipeline_oridinal_encoding(model, X, y, scaler):
    ordinal_pipeline, X, y = get_one_hot_encoding_pipeline(model, X, y, scaler)
    return evaluate(ordinal_pipeline, X, y)


def get_ordinal_encoding_pipeline(model, X, y, scaler=StandardScaler()):
    categories = constants.CATEGORY_VALUES
    categorical_columns = constants.CATEGORICAL_COLUMNS
    ordinal_encoder = OrdinalEncoder(categories=categories)
    pipeline = make_pipeline(
        ColumnTransformer(
            transformers=[
                ("categorical", ordinal_encoder, categorical_columns),
            ],
            remainder=scaler,
        ),
        model,
    )
    return pipeline, X, y


def get_one_hot_encoding_pipeline(model, X, y, scaler=StandardScaler()):
    categorical_columns = constants.CATEGORICAL_COLUMNS
    one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    one_hot_encoding_pipeline = make_pipeline(
        ColumnTransformer(
            transformers=[
                ("categorical", one_hot_encoder, categorical_columns),
            ],
            remainder=scaler,
        ),
        model,
    )

    """ this does visualization for bac """
    # df_copy = X[categorical_columns].copy()
    # one_hot_encoder.fit(df_copy)
    # print(one_hot_encoder.transform(df_copy))
    # print(one_hot_encoder.get_feature_names_out(categorical_columns))

    return one_hot_encoding_pipeline, X, y


def pipeline_one_hot_encoding(model, X, y, scaler=StandardScaler()):
    one_hot_encoding_pipeline, X, y = get_one_hot_encoding_pipeline(model, X, y, scaler)
    return evaluate(one_hot_encoding_pipeline, X, y)


def pipeline_one_hot_encoding_time_data(model, X, y, scaler):
    categorical_columns = constants.CATEGORICAL_COLUMNS
    one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    one_hot_pipeline = make_pipeline(
        ColumnTransformer(
            transformers=[
                ("categorical", one_hot_encoder, categorical_columns),
                ("one_hot_time", one_hot_encoder, ["day", "weekday", "month"]),
            ],
            remainder=scaler,
        ),
        model,
    )
    return evaluate(one_hot_pipeline, X, y)


def get_spline_pipeline(model, X, y, scaler=StandardScaler()):
    categorical_columns = constants.CATEGORICAL_COLUMNS
    one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    cyclic_spline_transformer = ColumnTransformer(
        transformers=[
            ("categorical", one_hot_encoder, categorical_columns),
            ("day", spline_transformer(31, n_splines=15), ["day"]),
            ("month", spline_transformer(12, n_splines=6), ["month"]),
            ("weekday", spline_transformer(7, n_splines=3), ["weekday"]),
        ],
        remainder=scaler
    )
    spline_pipeline = make_pipeline(
        cyclic_spline_transformer,
        model,
    )
    return spline_pipeline, X, y


def pipeline_spline_encoding_time_data(model, X, y, scaler):
    pipeline, X, y = get_spline_pipeline(model, X, y, scaler)
    return evaluate(pipeline, X, y)


def pipeline_polynomial_encoding(model, X, y, scaler=StandardScaler()):
    pipeline, X, y = get_polynomial_pipeline(model, X, y, scaler)
    return evaluate(pipeline, X, y)


def get_polynomial_pipeline(model, X, y, scaler=StandardScaler()):
    categorical_columns = constants.CATEGORICAL_COLUMNS
    one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    day_month_holiday_polynomial = make_pipeline(
        ColumnTransformer(
            [
                ("day", spline_transformer(31, n_splines=10), ["day"]),
                ("month", spline_transformer(12, n_splines=4), ["month"]),
                ("holiday", lambda_transformer(), ["holiday"]),
            ]
        ),
        PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
    )

    spline_transformer_periodic = ColumnTransformer(
        transformers=[
            ("categorical", one_hot_encoder, categorical_columns),
            ("day", spline_transformer(31, n_splines=15), ["day"]),
            ("month", spline_transformer(12, n_splines=6), ["month"]),
            ("weekday", spline_transformer(7, n_splines=3), ["weekday"]),
        ],
        remainder=scaler,
    )

    polynomial_pipeline = make_pipeline(
        FeatureUnion(
            [
                ("spline", spline_transformer_periodic),
                ("polynomial", day_month_holiday_polynomial),
            ]
        ),
        model,
    )
    return polynomial_pipeline, X, y


def lambda_transformer():
    return FunctionTransformer(lambda x: x == "True")


def spline_transformer(period, n_splines):
    n_knots = n_splines + 1
    return SplineTransformer(
        degree=2,
        n_knots=n_knots,
        knots=np.linspace(0, period, n_knots).reshape(n_knots, 1),
        extrapolation="periodic",
        include_bias=True,
    )
