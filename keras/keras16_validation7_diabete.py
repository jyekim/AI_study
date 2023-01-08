#실습 R2 0.62이상

from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
import numpy as np         
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터

dataset = load_diabetes()
x = dataset.data
y = dataset.target
x_train, x_test, y_train, y_test= train_test_split(x, y,
    train_size=0.7, random_state=66
) 

# print(x)
# print(x.shape) #(442, 10)
# print(y)
# print(y.shape) #(442,)
# print ('x_train : ',x_train) 
# print ('x_test : ', x_test)
# print ('y_train : ', y_train) 
# print ('y_test : ', y_test) 



#2.모델구성
model = Sequential()
model.add(Dense(19, input_dim=10, activation = 'relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='linear'))

#3.컴파일 훈련
model.compile(loss='mae', optimizer='adam') 
model.fit(x_train, y_train, epochs=600, batch_size=15, validation_split=0.2)



#4평가 예측 
loss = model.evaluate(x_test , y_test)
print('loss : ', loss)
y_predict = model.predict(x_test)
# print("===================")
# print(y_test)
# print(y_predict)
# print("====================")
#modelpredict에 최적의 가중치가 생성되어 있다/ x_test예측값이 생성되어서 y_predict에 넣어놓겠다 


def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict)) # RMSE ytest와 ypredict값을 받아서 
print("RMSE : ", RMSE(y_test, y_predict))  

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)


"""
결과 
#2.모델구성
model = Sequential()
model.add(Dense(19, input_dim=10, activation = 'relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(62, activation='relu'))
model.add(Dense(58, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='linear'))

#3.컴파일 훈련
model.compile(loss='mae', optimizer='adam') 
model.fit(x_train, y_train, epochs=300, batch_size=10, validation_split=0.2)
R2 :  0.3159580698321376



#2.모델구성
model = Sequential()
model.add(Dense(19, input_dim=10, activation = 'relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='linear'))

#3.컴파일 훈련
model.compile(loss='mae', optimizer='adam') 
model.fit(x_train, y_train, epochs=100, batch_size=5, validation_split=0.2)
R2 :  0.5046417113950639



"""