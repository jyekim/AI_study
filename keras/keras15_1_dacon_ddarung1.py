import numpy as np                
import pandas as pd             
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


#1. 데이터
path = './_data/ddarung/'   # , 은 현재 폴더
train_csv = pd.read_csv (path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'submission.csv', index_col=0)


print(train_csv) 
print(train_csv.shape) #.(1459,10)  

print(train_csv.columns) 
    #index (['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
    #    'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
    #    'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
    #   dtype='object') 

print(train_csv.info()) #non- null값은 결측치이다.  
print(test_csv.info())
print(train_csv.describe())

x = train_csv.drop(['count'], axis=1) 
print(x)   #[1459 rows x 9coloums]
y = train_csv['count']
print(y)
print(y.shape)  #(1459,)

x_train,x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7, shuffle=True, random_state=111
)
print(x_train.shape, x_test.shape) #(1021, 9) (438, 9)
print(y_train.shape, y_test.shape) #(1021,) (438,)



#2. 모델구성 
model = Sequential()
model.add(Dense(69,input_dim=9))
model.add(Dense(50))
model.add(Dense(78))
model.add(Dense(80))
model.add(Dense(85))
model.add(Dense(33))
model.add(Dense(21))
model.add(Dense(45))
model.add(Dense(1))

#3.컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics= ['mae']) 

model.fit(x_train, y_train, epochs=10, batch_size=32) 

#평가 예측 
loss = model.evaluate(x_test , y_test) 
print('loss : ', loss)

y_predict = model.predict(x_test) 
print(y_predict)


# 결측치 나쁜거
# print("===================")
# print(y_test)
# print(y_predict)
# print("====================")
#modelpredict에 최적의 가중치가 생성되어 있다/ x_test예측값이 생성되어서 y_predict에 넣어놓겠다 



#RMSE
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse) 

#제출할거 
y_submit =model.predict(test_csv)

# R2
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)