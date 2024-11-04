import datetime
import numpy as np

__all__ = ["MET_ref_time", \
           "epoch_Unix_time", \
           "MJD_ref_time", \
           "convert_datetime_to_MJD", \
           "convert_MET_to_dates", \
           "convert_epoch_to_dates", \
           "convert_decimal_year_to_dates", \
           "convert_fractional_year_to_datetime", \
           "convert_MJD_to_dates"]

MET_ref_time    = datetime.datetime(2001,  1, 1, 0, 0, 0)
epoch_Unix_time = datetime.datetime(1970,  1, 1, 0, 0, 0)
MJD_ref_time    = datetime.datetime(1858, 11, 17, 0, 0, 0)

def convert_datetime_to_MJD (dt):
    try:
        julian_date = dt.timestamp() / (24 * 3600) + 2440587.5
        mjd = julian_date - 2400000.5
    except:
        mjd = [convert_datetime_to_MJD(dt_) for dt_ in dt]
    return np.array(mjd)

def convert_MET_to_dates (MET_s):
    try:
        dates = MET_ref_time + datetime.timedelta(seconds = MET_s)
    except:
        dates = np.array([convert_MET_to_dates(MET) for MET in MET_s] )
    return dates

def convert_epoch_to_dates (epochs):
    try:
        dates = datetime.datetime.utcfromtimestamp(epochs).replace(tzinfo = datetime.timezone.utc)
    except:
        dates = np.array([convert_epoch_to_dates(epoch) for epoch in epochs] )
    return dates

def convert_decimal_year_to_dates(years):
    try:
        integer_year = int(years)
        decimal_days = (years - integer_year) * 365
        dates = datetime.datetime(integer_year, 1, 1) + datetime.timedelta (days = decimal_days)
    except:
        dates = np.array([convert_decimal_year_to_dates(year) for year in years])
    return dates

def convert_fractional_year_to_datetime (year_fraction):
    year = int(year_fraction)
    fraction = year_fraction - year
    is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    days_in_year = 366 if is_leap else 365
    total_seconds = fraction * days_in_year * 24 * 3600
    start_of_year = datetime.datetime(year, 1, 1)
    dt_object = start_of_year + datetime.timedelta(seconds=total_seconds)
    return dt_object

def convert_MJD_to_dates (mjd):
    try:
        dates = MJD_ref_time + datetime.timedelta(days = mjd)
    except:
        dates = np.array([convert_MJD_to_dates(mjd_) for mjd_ in mjd])
    return dates
