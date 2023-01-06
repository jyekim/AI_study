import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
path = './_data/bike/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)    # train_csv = pd.read_csv('./_data/ddarung/train.csv', index_col=0)==원래 해야하는거// index_col=0 == 0번째는 데이터 아니다.
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'samplesubmission.csv', index_col=0)

print(train_csv)    #, count는 y값이므로 제외해야한다. 
print(train_csv.columns)
# Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#        'humidity', 'windspeed', 'casual', 'registered', 'count'],
#       dtype='object')

print(train_csv.info())     #Non-Null Count 결측치
                            # 결측치가 있는 데이터는 삭제해버린다.
print(test_csv.info())
print(train_csv.describe()) #std = 표준편차, 50% = 중간값

###### 결측치 처리  1. 제거#####
print(train_csv.isnull().sum())         # null값 모두 더하기
train_csv = train_csv.dropna()          # 결측치 제거
print(train_csv.isnull().sum())         # null값 모두 더하기
print(train_csv.shape)                 
print(submission.shape)                  #//평가 데이터에도 결측치가 존재한다(삭제로는 해결 x)



x = train_csv.drop(['casual','registered','count'], axis=1)   # axis=축 #테스트랑 트레인 맞춰줘야하니깐 드롭할거
print(x)    
y = train_csv['count']
print(y)
print(y.shape)  # (10886, )

x_train, x_test, y_train, y_test = train_test_split(x, y,
                        train_size=0.7, shuffle=True, random_state=1234)
print(x_train.shape, x_test.shape)  #   (7620, 9) (3266, 9)
print(y_train.shape, y_test.shape)  #   (7620,) (3266,)

#2. 모델구성
model = Sequential()
model.add(Dense(9,input_dim=8, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(16, activation='sigmoid'))
model.add(Dense(23, activation= 'relu'))
model.add(Dense(40, activation= 'relu'))
model.add(Dense(23, activation= 'relu'))
model.add(Dense(113, activation= 'sigmoid'))
model.add(Dense(43, activation= 'relu'))
model.add(Dense(60, activation= 'relu'))
model.add(Dense(1, activation='linear'))         #제일 마지막값을 sigmoid로 두면 안됨 0~1로 한정되기 때문 , relu도 마지막 안됨

#3. 컴파일, 훈련
#loss = mae or mse optimizer= 'adam', matrix[mae or mse]
import time
model.compile(loss='mse', optimizer='adam',
                metrics=['mae'])
start = time.time()
model.fit(x_train, y_train, epochs=600, batch_size=50)
end = time.time()
print("걸린시간 : ", end - start)
# cpu 걸린시간 :  51.54655790328979560
# gpu 걸린시간 :  22.688528060913086

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
# print(y_predict)

# 결측치 처리 x

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)  # RMSE :  83.02001881026747
                        # RMSE :  53.88971756086701
                        
# submission.to_csv(path +"submission_0105.csv", mode='w')

# 제출
y_submit = model.predict(test_csv)
# print(y_submit)
# print(y_submit.shape)   #(715, 1)

# .to_csv()를 사용
#submission_0106.csv
#print(submission)
submission['count'] = y_submit  # y_submit 저장
print(submission)
submission.to_csv(path +"submission_0106.csv")

"""
결과
model.add(Dense(10,input_dim=8, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(11, activation='sigmoid'))
model.add(Dense(12, activation= 'relu'))
model.add(Dense(10, activation= 'relu'))
model.add(Dense(1, activation='linear'))      
model.fit(x_train, y_train, epochs=250, batch_size=32)
RMSE :  148.2598461464871





 """