import scipy
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
#import seaborn as sns


def sliding_window_pd(df, ws=500, overlap=250, w_type="hann", w_center=True, print_stats=False):
    """Applies the sliding window algorithm to the DataFrame rows.

    Args:
        df: The DataFrame with all the values that will be inserted to the sliding window algorithm.
        ws: The window size in number of samples.
        overlap: The hop length in number of samples.
        w_type: The windowing function.
        w_center: If False, set the window labels as the right edge of the window index. If True, set the window
                labels as the center of the window index.
        print_stats: Print statistical inferences from the process (Default: False).

    Returns:
        A list of DataFrames each one corresponding to a produced window.
    """
    counter = 0
    windows_list = list()
    # min_periods: Minimum number of observations in window required to have a value;
    # For a window that is specified by an integer, min_periods will default to the size of the window.
    for window in df.rolling(window=ws, step=overlap, min_periods=ws, win_type=w_type, center=w_center):
        if window[window.columns[0]].count() >= ws:
            if print_stats:
                print("Print Window:", counter)
                print("Number of samples:", window[window.columns[0]].count())
            windows_list.append(window)
        counter += 1
    if print_stats:
        print("List number of window instances:", len(windows_list))

    return windows_list


def apply_filter(arr, order=5, wn=0.1, filter_type="lowpass"):
    """Apply filter to the multi-axis signal.

    Args:
        arr: The initial NumPy signal array values.
        order: The order of the filter.
        wn: The critical frequency or frequencies.
        filter_type: The type of filter. {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}

    Returns:
        NumPy Array with the filtered signal.
    """
    fbd_filter = scipy.signal.butter(N=order, Wn=wn, btype=filter_type, output="sos")
    filtered_signal = scipy.signal.sosfiltfilt(sos=fbd_filter, x=arr, padlen=0)

    return filtered_signal


def filter_instances(instances_list, order, wn, filter_type):
    """Apply filter to a list of windows (each window is a DataFrame).

    Args:
        instances_list: List of DataFrames.
        order: The order of the filter.
        wn: The critical frequency or frequencies.
        filter_type: The type of filter. {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}

    Returns:

    """
    filtered_instances_list = list()
    for item in instances_list:
        filtered_instance = item.apply(apply_filter, args=(order, wn, filter_type))
        filtered_instances_list.append(filtered_instance)
    print("Number of filtered instances in the list:", len(filtered_instances_list))

    return filtered_instances_list


def flatten_instances_df(instances_list):
    """Flatten each instance and create a DataFrame with the whole flattened instances.

    Args:
        instances_list: The list of DataFrames to be flattened

    Returns:
        A DataFrame that includes the whole flattened DataFrames
    """
    flattened_instances_list = list()
    for item in instances_list:
        instance = item.to_numpy().flatten()
        flattened_instances_list.append(instance)
    df = pd.DataFrame(flattened_instances_list)

    return df


def df_rebase(df1, df2, order_list, rename_dict):
    """Concats the gyro and accelerometer data.
    Also changes the order and name of DataFrame columns to the project's needs for readability.

    Args:
        df1: The pandas DataFrame.
        df2: The pandas DataFrame.
        order_list: List object that contains the proper order of the default sensor column names.
        rename_dict: Dictionary object that contains the renaming list based on the project needs.

    Returns:
        A DataFrame with the new columns order and names.
    """

    # make sure that the two files have the same number of rows
    if len(df1) != len(df2):
        if len(df1) > len(df2):
            df1 = df1[:len(df2)]
        else:
            df2 = df2[:len(df1)]

    df = pd.concat([df1, df2], axis=1)  # concat the two DataFrames
    
    df = df[order_list]  # keep and re-order only the necessary columns of the initial DataFrame
    df = df.rename(columns= dict(zip(order_list, rename_dict)))  # rename the columns

    return df


def rename_df_column_values(np_array, y, columns_names=("acc_x", "acc_y", "acc_z")):
    """Creates a DataFrame with a "y" label column and replaces the values of the y with the index
    of the unique values of y.

    Args:
        np_array: 2D NumPy array.
        y: List with the y labels
        columns_names: List with the DF columns names.

    Returns:
        DataFrame with the multi-axes values and the target labels column.
    """
    arr_y = np.array(y)  # list to numpy array
    unique_values_list = np.unique(arr_y)  # unique list of values

    df = pd.DataFrame(np_array, columns=columns_names)
    df["y"] = y

    # replace the row item value in the y column of the df, with its index in the unique list
    for idx, x in enumerate(unique_values_list):
        df["y"] = np.where(df["y"] == x, idx, df["y"])

    return df

def impute_nan(X):
    """ Impute NaN values using KNNImputer with custom weights.
    Args:
        X: 3D NumPy array with NaN values (numpy.ndarray).
    Returns:
        3D NumPy array with imputed values (numpy.ndarray).
    """

    def custom_weights(distances):
        # Calculate distance mod 6 
        mod_distances = distances % (shapes[2])

        # Assign weight 1 / (distance ^ 2) to neighbors with distance mod 6 equal to 0, weight 0 to others
        weights = np.where(mod_distances == 0, 1 / (distances**2), 0)

        return weights
    
    shapes = X.shape
    data = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2]))
    imputer = KNNImputer(n_neighbors=5, weights=custom_weights)
    imputed_data = imputer.fit_transform(data)
    imputed_data = np.reshape(imputed_data, shapes)

    return imputed_data

