#This program trains a simple linear regression model to predict house sale prices in washington's king county.

import sklearn
import pandas as pd

KC_CSV_PATH = 'C:/Users/Iagê/Desktop/MLPy/Modulo1/1_RegressãoLinear/kc_house_data.csv'

#creates dataset based on the csv file, using the library pandas
dataset = pd.read_csv(KC_CSV_PATH)  #https://www.kaggle.com/datasets/harlfoxem/housesalesprediction

#fixes the dataset, eliminating useless fatures for the analysis
dataset.drop('id', axis=1, inplace=True)    #axis=1 means its a column;
dataset.drop('date', axis=1, inplace=True)  #inplace=True means the change will be saved in the original dataset
dataset.drop('lat', axis=1, inplace=True)
dataset.drop('long', axis=1, inplace=True)
dataset.drop('zipcode', axis=1, inplace=True)

#defines the feature variables (x) and the target variable (y)
y = dataset['price']                    #y target variable is the price
x = dataset.drop(['price'], axis=1)     #x feature variables are the all the variables except the price

#splits data between train and test, using the sklearn library; test set percentage is 30%
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.3)

#trains linear regression model using the training set
reg = sklearn.linear_model.LinearRegression()
reg.fit(x_train, y_train)

#calculates the coefficient of determination R² by comparing the trained model against the test set
result = reg.score(x_test, y_test)  
print(result)

