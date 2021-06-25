import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
from pandas.api.types import CategoricalDtype


def prepare_data(train, test, stores, features):
    """
    This function takes 4 dataframes and prepare the data 
    that can be feed into scikit-learn models.
    Return : X_train, y_train, X_test, full_pipe
    """

    # Merge the stores data with train and test
    train = pd.merge(train, stores, how="left", on="Store")
    test = pd.merge(test, stores, how="left", on="Store")

    # Merge the features data with train and test
    train = pd.merge(train, features, how="left", on=["Store", "Date"])
    test = pd.merge(test, features, how="left", on=["Store", "Date"])

    train.drop(["IsHoliday_y"], axis=1, inplace=True)
    test.drop(["IsHoliday_y"], axis=1, inplace=True)

    # rename column
    train.rename(columns={"IsHoliday_x": "IsHoliday"}, inplace=True)
    test.rename(columns={"IsHoliday_x": "IsHoliday"}, inplace=True)

    ## Datetime features
    train["Year"] = train["Date"].dt.year
    train["Month"] = train["Date"].dt.month
    train["Day"] = train["Date"].dt.day
    train["WeekOfYear"] = train["Date"].dt.weekofyear
    train["DayOfWeek"] = train["Date"].dt.dayofweek
    train["Weekend"] = (train["Date"].dt.weekday >= 5).astype(int)

    test["Year"] = test["Date"].dt.year
    test["Month"] = test["Date"].dt.month
    test["Day"] = test["Date"].dt.day
    test["WeekOfYear"] = test["Date"].dt.weekofyear
    test["DayOfWeek"] = test["Date"].dt.dayofweek
    test["Weekend"] = (test["Date"].dt.weekday >= 5).astype(int)

    # convert boolean column to categorical column
    train["IsHoliday"] = train["IsHoliday"].map({True: "Yes", False: "No"})
    test["IsHoliday"] = test["IsHoliday"].map({True: "Yes", False: "No"})
    train["IsHoliday"] = train["IsHoliday"].astype("category")
    test["IsHoliday"] = test["IsHoliday"].astype("category")

    # ordered the categorical store type col
    from pandas.api.types import CategoricalDtype

    cat_type = CategoricalDtype(categories=["C", "B", "A"], ordered=True)
    train["Type"] = train["Type"].astype(cat_type)
    test["Type"] = test["Type"].astype(cat_type)

    # convert to categorical columns
    train["Store"] = train["Store"].astype("category")
    train["Dept"] = train["Dept"].astype("category")
    train["Year"] = train["Year"].astype("category")
    train["Month"] = train["Month"].astype("category")
    train["DayOfWeek"] = train["DayOfWeek"].astype("category")
    train["Weekend"] = train["Weekend"].astype("category")

    # convert to categorical columns
    test["Store"] = test["Store"].astype("category")
    test["Dept"] = test["Dept"].astype("category")
    test["Year"] = test["Year"].astype("category")
    test["Month"] = test["Month"].astype("category")
    test["DayOfWeek"] = test["DayOfWeek"].astype("category")
    test["Weekend"] = test["Weekend"].astype("category")
    

    # features and labels of train and test set
    # labels of test are not provided as we need to predict them

    X_train = train.drop(["Weekly_Sales"], axis=1).copy()
    y_train = train["Weekly_Sales"].copy()

    X_test = test.copy()

    # drop and save the date column in a variable
    train_date = X_train.pop("Date")
    test_date = X_test.pop("Date")

    #### Data preparation pipeline

    # select numerical and categorical columns
    num_cols = X_train.select_dtypes(exclude=["object", "category"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    # numerical date preprocessing pipeline
    num_pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

    # categorical data preprocessing pipeline
    cat_pipe = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="NA"),
        OneHotEncoder(handle_unknown="ignore", sparse=False),
    )

    # full pipeline
    full_pipe = ColumnTransformer(
        [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)]
    )

    full_pipe

    return X_train, y_train, X_test, full_pipe
