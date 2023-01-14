import numpy as np                              
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터 

dataset = load_diabetes()
x = dataset.data
y = dataset.target                           
x_train, x_test, y_train, y_test = train_test_split(X, y,
        train_size=0.7, random_state=123, shuffle=True )

print(x.shape)
