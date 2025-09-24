'''This program trains and compares four different regression models on a dataset for predicting graduate admissions of indian students. It states the best model for predicting the student's
'chance of admit' based on their coefficient of determination.'''

import pandas as pd
import sklearn
ADMISSION_CSV_PATH = "C:/Users/Iagê/Desktop/MLPy/Modulo1/3_MoreRegressionModels/Admission_Predict.csv" #https://www.kaggle.com/datasets/mohansacharya/graduate-admissions/data

def extract_data(CSV_PATH):
    dataframe = pd.read_csv(CSV_PATH)
    dataframe = dataframe.drop(columns=['Serial No.',]) #drops serial number column useless to the analysis

    features = dataframe.drop(columns=['Chance of Admit ']) #there is a space char at the end of feature in the csv file
    target = dataframe['Chance of Admit ']

    return features, target, dataframe

def analysis(x, y, dataframe):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.3)  #splits data between train and test

    linReg = sklearn.linear_model.LinearRegression() #trains the models 
    linReg.fit(x_train, y_train)
    
    ridgeReg = sklearn.linear_model.Ridge()
    ridgeReg.fit(x_train, y_train)

    lassoReg = sklearn.linear_model.Lasso()
    lassoReg.fit(x_train, y_train)

    elNetReg = sklearn.linear_model.ElasticNet()
    elNetReg.fit(x_train, y_train)

    linRegSc = linReg.score(x_test, y_test)     #scores R2 coefficient using test data
    ridgeRegSc = ridgeReg.score(x_test, y_test)
    lassoRegSc = lassoReg.score(x_test, y_test)
    elNetRegSc = elNetReg.score(x_test, y_test)

    print(f"R2 coefficient scores:\nLinear Regression: {linRegSc}, Ridge Regression: {ridgeRegSc}, Lasso Regression: {lassoRegSc}, Elastic Net: {elNetRegSc}")
    regScores = {linRegSc:"Linear Regression",
                ridgeRegSc:"Ridge Regression",
                lassoRegSc:"Lasso Regression",
                elNetRegSc:"Elastic Net"
    }
    bestModel = max(regScores.keys()) #gets the maximum key in the dictionary
    print("The best model for predicting the target value in this dataset is:", regScores[bestModel])

x, y, dataframe = extract_data(ADMISSION_CSV_PATH)
analysis(x, y, dataframe)

