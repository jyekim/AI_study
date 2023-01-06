# 1. train 0.7이상
# 2.R2 :0.8  이상/ RMSE 사용 

# import sklearn as sk
# print(sk.__version__)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from sklearn.datasets import load_boston
import numpy as np         
from sklearn.model_selection import train_test_split 


#1. 데이터 보스턴 집값에 대한 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test= train_test_split(x, y,
    train_size=0.8, shuffle=True, random_state=123
) 
print ('x_train : ',x_train) 
print ('x_test : ', x_test)
print ('y_train : ', y_train) 
print ('y_test : ', y_test) 

# print(x)
# print(x.shape) #(506, 13)
# print(y)
# print(y.shape) #(506,)


print(dataset.feature_names)
#이런 컬럼이 13개가 있다['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']

print(dataset.DESCR)

#2. 모델구성 
model = Sequential()
model.add(Dense(5,input_dim=13))
model.add(Dense(10))
model.add(Dense(40))
model.add(Dense(27))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(28))
model.add(Dense(31))
model.add(Dense(25))
model.add(Dense(1))

#3.컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics= ['mae']) 

model.fit(x_train, y_train, epochs=5500, batch_size=20) 

#평가 예측 
loss = model.evaluate(x_test , y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
print("===================")
print(y_test)
print(y_predict)
print("====================")
#modelpredict에 최적의 가중치가 생성되어 있다/ x_test예측값이 생성되어서 y_predict에 넣어놓겠다 

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict)) # RMSE ytest와 ypredict값을 받아서 
print("RMSE : ", RMSE(y_test, y_predict))  

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)
#r2는높으면 성능이 좋다라는 것을 알 수 있다. 

"""
    결과 
model.add(Dense(10,input_dim=13))
model.add(Dense(10))
model.add(Dense(1035))
model.add(Dense(7))
model.add(Dense(30))
model.add(Dense(320))
model.add(Dense(140))
model.add(Dense(1))
 epochs=250, batch_size=1) 
RMSE :  5.293825328795981
R2 :  0.6532816000979739
"""