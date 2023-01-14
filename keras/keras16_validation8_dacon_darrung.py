import numpy as np
import pandas as pd
import time as t
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


#1. 데이터
# path = './_data/ddarung/'
path = '../_data/ddarung/'         #이거는 keras폴더로만 사용해서 gpu돌리게 될때 경로 재설정
# path = 'c:/study/_data/ddarung/'   #이거는 cpu gpu 둘 다 쓸때 경로 설정 없이 가능함
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# train_csv = pd.read_csv('./_data/ddarung/train.csv', index_col=0)    # 원래 해야하는거, index_col=0 == 0번째는 데이터 아니다.
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'submission.csv', index_col=0)

print(train_csv)    #(1459, 10) , count는 y값이므로 제외해야한다. input_dim=9
print(submission.shape)
print(train_csv.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')
print(train_csv.info())     #Non-Null Count 결측치(1459- 1457 =2), (1459-1457 = 2), (1459-1450=9) ...
                            # 결측치가 있는 데이터는 삭제해버린다.
print(test_csv.info())
print(train_csv.describe()) #std = 표준편차, 50% = 중간값

###### 결측치 처리  1. 제거#####
print(train_csv.isnull().sum())         # null값 모두 더하기
train_csv = train_csv.dropna()          # 결측치 제거
print(train_csv.isnull().sum())         # null값 모두 더하기
print(train_csv.shape)                  # (1328, 10)

x = train_csv.drop(['count'], axis=1)   # axis=축
print(x)    #   [1459 rows x 9 columns]
y = train_csv['count']
print(y)
print(y.shape)  # (1459, )

x_train, x_test, y_train, y_test = train_test_split(x, y,
                        train_size=0.7, shuffle=True, random_state=1234)
print(x_train.shape, x_test.shape)  #   (929, 9) (399, 9)
print(y_train.shape, y_test.shape)  #   (929,) (399,)


#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=9, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(9, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(10, activation='linear' ))
model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련
#loss = mae or mse optimizer= 'adam', matrix[mae or mse]
model.compile(loss='mse', optimizer='adam',
                metrics=['mae'])
model.fit(x_train, y_train, epochs=600, batch_size=5)

#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
print(y_predict)

# 결측치 처리 x

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)  # RMSE :  83.02001881026747


# 제출
y_submit = model.predict(test_csv)   #예측한 카운트가 y_submit 
# print(y_submit)
#print(y_submit.shape) #(715, 1) 

#.to_csv()를 사용해서
#submission.0105.csv를 완성하시오 

# print(submission)
submission['count'] = y_submit
# print(submission)
 
submission.to_csv(path + 'submission_01050251.csv')


"""
결과


 """