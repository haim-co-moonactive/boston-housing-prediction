from joblib import load
from pandas import read_csv, DataFrame
from numpy import ndarray

# Load the pipeline from a file
model = load('model.joblib')
# Read the data to predict into a pandas data frame
predict_data: DataFrame = read_csv("predict_data_Boston_dataset.csv")
# Predict
predictions: ndarray = model.predict(predict_data)
# The outcome is in the form of an ndarray. Each incoming instance have a correspondent result
print(predictions)
