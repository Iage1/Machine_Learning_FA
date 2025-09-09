import sklearn
import pandas as pd
import numpy as np

KC_CSV_PATH = 'C:/Users/Iagê/Desktop/MLPy/Modulo1/1.RegressãoLinear/kc_house_data.csv'

#cria o dataset com base no arquivo csv, utilizando a biblioteca pandas
dataset = pd.read_csv(KC_CSV_PATH)

#arruma o dataset, excluindo colunas inúteis para a análise
dataset.drop('id', axis=1, inplace=True)    #axis=1 especifica que é uma coluna;
dataset.drop('date', axis=1, inplace=True)  #inplace=True especifica que é pra salvar alterações no dataset original;
dataset.drop('lat', axis=1, inplace=True)
dataset.drop('long', axis=1, inplace=True)
dataset.drop('zipcode', axis=1, inplace=True)

#define e atribui os valores do dataset a variavel alvo e variaveis preditoras
y = dataset['price']
x = dataset.drop(['price'], axis=1)     #atribui a variavel x (features) como sendo o dataset menos a coluna price (target)

#sepera os dados de treino e teste, com uma porcentagem de 30% para teste
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.3)

#cria o modelo de regressão linear, e executa o algoritmo com os dados de treino
reg = sklearn.linear_model.LinearRegression()
reg.fit(x_train, y_train)

#calcula o coeficiente de determinação R2, comparando a regressão obtida com os dados de treino(reg) com os dados de teste
result = reg.score(x_test, y_test)  
print(result)

