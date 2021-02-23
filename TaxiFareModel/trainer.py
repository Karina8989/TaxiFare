from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data
from sklearn.model_selection import train_test_split
import pandas as pd

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""

        pipe_distance = make_pipeline(DistanceTransformer(), StandardScaler())
        pipe_time = make_pipeline(TimeFeaturesEncoder(time_column='pickup_datetime'), OneHotEncoder())
        
        dist_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
        time_cols = ['pickup_datetime']
        
        preproc = ColumnTransformer([('time', pipe_time, time_cols),
                                    ('distance', pipe_distance, dist_cols)])
                                    
        pipe_cols = Pipeline(steps=[('preproc', preproc),
                                ('regressor', LinearRegression())])
    
    
        return pipe_cols

    def run(self, X_train, y_train):
        """set and train the pipeline"""

        self.pipeline = self.set_pipeline()
        self.pipeline.fit(X_train, y_train)
        return self.pipeline


    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        return compute_rmse(y_pred, y_test)


if __name__ == "__main__":
    # get data
    df = get_data(nrows=10_000)
    # clean data
    df_clean = clean_data(df, test=False)
    # set X and y
    X = df_clean.drop(columns='fare_amount')
    y = df_clean['fare_amount']
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)
    # train
    trainer = Trainer(X, y)
    trainer.run(X_train, y_train)
    # evaluate
    print(trainer.evaluate(X_test, y_test))
