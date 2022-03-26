# Imported all necessary libraries here.
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import pylab
import scipy.stats as stats
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
sns.set(style="darkgrid")
# Display all the columns of the dataframe

pd.pandas.set_option('display.max_columns', None)


# Drop specific columns name from the dataframe


def drop_column_from_df(dataframe_name, list_column_name):
    return dataframe_name.drop(list_column_name, axis=1)

# First of let combine day,month, year column and convert that into datetime format.
# This function is call by reference.


def combine_day_month_year(df_name, list_of_day_month_year_columns):
    for column in list_of_day_month_year_columns:
        df_name[column] = df_name[column].astype(str)
        df_name[column] = df_name[column].apply(
            lambda value: value.split('.')[0])
    df_name['Date'] = df_name[list_of_day_month_year_columns].apply(
        lambda x: '-'.join(x.values.astype(str)), axis="columns")
    df_name['Date'] = pd.to_datetime(df_name['Date'])

# Draw line graph


def line_graph(df_name, x_axis_column, y_axis_column, x_axis_title="", y_axis_title=""):
    g = sns.relplot(
        data=df_name,
        x=x_axis_column, y=y_axis_column,
        kind="line",
        height=5, aspect=2
    )
    if(x_axis_title == "" or y_axis_title == ""):
        g = (g.set_axis_labels(x_axis_column, y_axis_column))
    else:
        g = (g.set_axis_labels(x_axis_title, y_axis_title))

# Using Pearson Correlation


def correlation_matrix(df_name):
    plt.figure(figsize=(12, 10))
    cor = df_name.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
    plt.show()

# Function to print all rows which have null values column wise.


def missing_rows(dataframe_name, column_name):
    return dataframe_name[dataframe_name[column_name].isnull()]

# Seperate out (Day,Month,Year) column from single column.


def make_new_columns_day_month_year(dataframe_name, date_column_name):
    dataframe_name["Day"] = dataframe_name[date_column_name].dt.day
    dataframe_name["Month"] = dataframe_name[date_column_name].dt.month
    dataframe_name["Year"] = dataframe_name[date_column_name].dt.year

# First of all we are going to fill missing values in Day column.


def fill_Day_column_missing_values(dataframe_name, row):
    if(np.isnan(row["Month"])):
        # Here axis=1, and inplace=True is important bcz only we can only change value of object,
        # its not possible to mutate whole obbject.
        dataframe_name.drop(row.name, inplace=True)
    else:
        if((dataframe_name.loc[row.name-1, "Day"] - 1) != 0):
            calculated_day = (dataframe_name.loc[row.name-1, "Day"] - 1)
        else:
            if(row["Month"] == 4 or row["Month"] == 6 or row["Month"] == 0 or row["Month"] == 11):
                calculated_day = (
                    dataframe_name.loc[row.name-1, "Day"] - 1) + 30
            elif(row["Month"] == 2):
                calculated_day = (
                    dataframe_name.loc[row.name-1, "Day"] - 1) + 28
            else:
                calculated_day = (
                    dataframe_name.loc[row.name-1, "Day"] - 1) + 31

        row["Day"] = calculated_day
    return row


def fill_missing_values_for_day(dataframe_name):
    dataframe_name = dataframe_name.apply(
        lambda row: fill_Day_column_missing_values(
            dataframe_name, row) if np.isnan(row["Day"]) else row,
        axis=1
    )

# filter on rows with specific values for day column.


def day_filter_with_specific_value(dataframe_name, column_name, value):
    return dataframe_name[dataframe_name[column_name] == value]

# filter on rows with specific values for month column.


def month_filter_with_specific_value(dataframe_name, column_name, value):
    return dataframe_name[dataframe_name[column_name] == value]

# filter on rows with specific values for year column.


def year_filter_with_specific_value(dataframe_name, column_name, value):
    return dataframe_name[dataframe_name[column_name] == value]

# Q-Q plot


def qq_plot(dataframe_name, column_name):
    stats.probplot(dataframe_name['Price'], dist="norm", plot=pylab)
    pylab.show()

# Box plot to detect out lier.


def outlier_box_plot(df_name, column_name):
    plt.figure(figsize=(4, 8))
    sns.boxplot(y=df_name[column_name])

# Outlier Detection using IQR


def outlier_detection_iqr(df, column):
    global lower, upper
    q25, q75 = np.quantile(df[column], 0.25), np.quantile(df[column], 0.75)
    # calculate the IQR
    iqr = q75 - q25
    # calculate the outlier cutoff
    cut_off = iqr * 1.5
    # calculate the lower and upper bound value
    lower, upper = q25 - cut_off, q75 + cut_off
    print('The IQR is', iqr)
    print('The lower bound value is', lower)
    print('The upper bound value is', upper)
    # Calculate the number of records below and above lower and above bound value respectively
    df1 = df[df[column] > upper]
    df2 = df[df[column] < lower]
    return print('Total number of outliers are', df1.shape[0] + df2.shape[0])

# Outlier Distribution plot


def outlier_distribution_graph(df_name, col_name):
    plt.figure(figsize=(10, 6))
    sns.distplot(df_name[col_name], kde=False)
    plt.axvspan(
        xmin=lower, xmax=df_name[col_name].min(), alpha=0.2, color='red')
    plt.axvspan(
        xmin=upper, xmax=df_name[col_name].max(), alpha=0.2, color='red')


# Data Frame without outliers

def remove_rows_having_outlier(df_name, column_name, upper_limit, lower_limit):
    return df_name[(df_name[column_name] < upper_limit) & (df_name[column_name] > lower_limit)]

# Apply Log function on data


def log_on_column(dataframe_name, column_name):
    dataframe_name[column_name] = dataframe_name[column_name].apply(
        lambda value: np.log(value)if value > 0 else value)

# Draw distibution plot


def distribution_plot_of_column(df_name, column_name, title=""):
    sns.distplot(df_name[column_name], color='black').set_title(title)


# standard deviation method to remove outlier from the dataset.
def outlier_detection_std(df, column):
    global lower, upper
    # calculate the mean and standard deviation of the data frame
    data_mean, data_std = df[column].mean(), df[column].std()
    # calculate the cutoff value
    cut_off = data_std * 3
    # calculate the lower and upper bound value
    lower, upper = data_mean - cut_off, data_mean + cut_off
    print('The lower bound value is', lower)
    print('The upper bound value is', upper)
    # Calculate the number of records below and above lower and above bound value respectively
    df1 = df[df[column] > upper]
    df2 = df[df[column] < lower]
    return print('Total number of outliers are', df1.shape[0] + df2.shape[0])

# make csv file (pass file_name as string with .csv)


def make_csv_file(df_name, file_name):
    df_name.to_csv(file_name, index=False)


# Mean Squared Error, Root Mean Squared Error, Mean Absolute Error:
#     Mean Squared Error, or MSE for short, is a popular error metric for regression problems.

#     It is also an important loss function for algorithms fit or optimized using the least squares framing of a regression problem. Here “least squares” refers to minimizing the mean squared error between predictions and expected values.

#     The MSE is calculated as the mean or average of the squared differences between predicted and expected target values in a dataset.

#     MSE = 1 / N * sum for i to N (y_i – yhat_i)^2

#     Where y_i is the i’th expected value in the dataset and yhat_i is the i’th predicted value. The difference between these two values is squared, which has the effect of removing the sign, resulting in a positive error value.

# Return mean_squared_error, root_mean_squared_error, and mean_absolute_error.
def calculate_mse_rmse_mae(y_actual, y_predicted):
    mse_errors = mean_squared_error(y_actual, y_predicted)
    rmse_errors = mean_squared_error(y_actual, y_predicted, squared=False)
    mae_errors = mean_absolute_error(y_actual, y_predicted)
    return mse_errors, rmse_errors, mae_errors
