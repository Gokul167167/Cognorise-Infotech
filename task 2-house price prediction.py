# importing dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

# loading our dataset from csv file into a data framework using pandas
housedat=pd.read_csv('data.csv')

# Exploratory Data Analysis of our dataset
print(housedat.head())
print(housedat.describe())
print(housedat.columns)
print(housedat.isnull().sum())
print(housedat.dtypes)

# Splitting the dataset into dependent and independent variables
x=housedat[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
       'floors', 'waterfront', 'view', 'condition']]
y=housedat['price']
print(x)
print(y)

# Splitting x and y into four arrays as training and testing data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# Building a linear regression model using training data
model=LinearRegression()
model.fit(x_train,y_train)

# Making predictions on the trained model using testing data
predict_y=model.predict(x_test)

# Using mean squared error and r-squared to evaluate the working of model
mse=mean_squared_error(y_test,predict_y)
r2=r2_score(y_test,predict_y)
print("Mean squared error:",mse)
print("R-squared:",r2)

# Visual represention of actual price vs predicted price using scatter plot
plt.scatter(y_test,predict_y)
plt.xlabel(y_test)
plt.ylabel(predict_y)
plt.title("Actual Price Vs Predicted Price")
plt.show()

# Visual representation of actual price vs residual using scatter plot
residual=y_test-predict_y
plt.scatter(y_test,residual)
plt.xlabel(y_test)
plt.ylabel(residual)
plt.axhline(y="0",color="green",linestyle="--")
plt.title("Residual Plot")
plt.show()

# Predicting the price of a house from a given data with the help of our model 
new_data=[[2.0,2.0,1350,2560,1.0,0,0,3]]
price=model.predict(new_data)
print("The price of the house is",price[0])


