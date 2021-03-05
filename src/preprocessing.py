import pandas as pd
import os
dir = os.getcwd()


def get_time_stamp(quantile=0.95):
    """
    This function reads the csv and filter on the days where the VIX has a value greater than
    the 95% quantile to form our event set.
    :param quantile: quantile threshold - default 95%
    :return: time_stamp of events
    """
    df = pd.read_csv(dir + "/../data/VIX.csv").iloc[::-1].reset_index(drop=True)
    df['price_date'] = pd.to_datetime(df['price_date'])
    upper_bound = df.quantile(quantile)['last_price']
    df_filter = df.loc[(df['last_price'] >= upper_bound)]
    time_stamp = list(df_filter.index)
    max_T = df.shape[0]
    return time_stamp, max_T