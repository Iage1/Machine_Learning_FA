'''This program displays the percentage of missing data for each variable in a dataset relating to energy benchmarkings of buildings Seattle, 
and replaces missing values in ENERGYSTARScore with its median. It also generates and displays a heatmap of the correlation between numerical variables.'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

SEA_CSV_PATH = 'C:/Users/Iagê/Desktop/MLPy/Modulo1/2_PreProcessamento/2015-building-energy-benchmarking.csv'


def missing_data_treatment():
    dataset = pd.read_csv(SEA_CSV_PATH) #https://www.kaggle.com/city-of-seattle/sea-building-energy-benchmarking

    missing = dataset.isnull().sum()        #dataframe containing sum of missing data for each feature
    percentage = missing/len(dataset)*100   #percentage of missing data for each feature (missing/total)
    print(percentage)

    #replaces missing values in ENERGYSTARScore with its median
    dataset['ENERGYSTARScore'] = dataset['ENERGYSTARScore'].fillna(dataset['ENERGYSTARScore'].median())

    return dataset 

def correlation_heatmap():
    dataset = missing_data_treatment()  

    num_variables = ['YearBuilt', 'NumberofFloors', 'PropertyGFATotal', 'PropertyGFAParking', 'PropertyGFABuilding(s)', 'ENERGYSTARScore', 'Electricity(kWh)', 'NaturalGas(therms)', 'GHGEmissions(MetricTonsCO2e)', ]
    dataset = dataset[num_variables]

    plt.figure(figsize=(5,5))
    sns.heatmap(dataset.corr(method='pearson'))
    plt.show()

correlation_heatmap()

