from sklearn.base import BaseEstimator, TransformerMixin
from TaxiFareModel.utils import haversine_vectorized, extract_time_features


class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    """Extract the day of week (dow), the hour, the month and the year from a
    time column."""
    def __init__(self, time_column, time_zone_name='America/New_York'):
        self.time_column = time_column
        self.time_zone_name = time_zone_name
        
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Returns a copy of the DataFrame X with only four columns: 'dow', 'hour', 'month', 'year'"""
        X_ = X.copy()
        return extract_time_features(X_)


class DistanceTransformer(BaseEstimator, TransformerMixin):
    """Compute the haversine distance between two GPS points."""

    def __init__(self, 
                 start_lat="pickup_latitude",
                 start_lon="pickup_longitude", 
                 end_lat="dropoff_latitude", 
                 end_lon="dropoff_longitude"):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        """Returns a copy of the DataFrame X with only one column: 'distance'"""
        X_ = X.copy()
        X_['distance'] = haversine_vectorized(X_, self.start_lat, self.start_lon, self.end_lat, self.end_lon)
        return X_[['distance']]
