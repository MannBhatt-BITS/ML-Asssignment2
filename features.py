"""Shared, deterministic feature extraction for CrimeCast."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

NUMERIC = ['Latitude','Longitude','Victim_Age','Area_ID','Reporting_District_no','Part 1-2','Premise_Code','Weapon_Used_Code','occ_month','occ_weekday','occ_hour','report_delay_days','modus_count']
CATEGORICAL = ['Victim_Sex','Victim_Descent','modus_first']

class CrimeFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        d = X.copy()
        occurred = pd.to_datetime(d['Date_Occurred'], errors='coerce')
        reported = pd.to_datetime(d['Date_Reported'], errors='coerce')
        d['occ_month'] = occurred.dt.month
        d['occ_weekday'] = occurred.dt.dayofweek
        t = pd.to_numeric(d['Time_Occurred'], errors='coerce')
        d['occ_hour'] = np.where((t >= 0) & (t <= 2359) & ((t % 100) < 60), np.floor(t/100), np.nan)
        d['report_delay_days'] = (reported - occurred).dt.total_seconds()/86400
        d.loc[~d['report_delay_days'].between(0, 3650), 'report_delay_days'] = np.nan
        codes = d['Modus_Operandi'].fillna('').astype(str).str.strip().str.split()
        d['modus_count'] = codes.str.len()
        d['modus_first'] = codes.str[0].fillna('Unknown')
        for c in NUMERIC:
            d[c] = pd.to_numeric(d[c], errors='coerce')
        for c in CATEGORICAL:
            d[c] = d[c].fillna('Unknown').astype(str)
        return d[NUMERIC+CATEGORICAL]
