import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler



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



# scaler = MinMaxScaler()   #
# scaler.fit(x_train)
# x_train = scaler.fit_transform(x_train)   #minmaxscaler  
# x_test = scaler.transform(x_test)
# test_csv = scaler.transform(test_csv)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)     
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#2. 모델구성
# model = Sequential()
# model.add(Dense(9,input_dim=8, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(23, activation= 'relu'))
# model.add(Dense(40, activation= 'relu'))
# model.add(Dense(113, activation= 'relu'))
# model.add(Dense(43, activation= 'relu'))
# model.add(Dense(1, activation= 'linear'))         #제일 마지막값을 sigmoid로 두면 안됨 0~1로 한정되기 때문 , relu도 마지막 안됨


# #2. 모델구성(함수형)
input1 = Input(shape=(8,))       #인풋레이어는 
dense1 = Dense(50, activation= 'relu')(input1)
dense2 = Dense(40, activation= 'sigmoid')(dense1)
dense3 = Dense(30, activation= 'relu')(dense2)
dense4 = Dense(20, activation= 'linear')(dense3)
output1 = Dense(1, activation= 'linear')(dense4)
model = Model(inputs=input1, outputs=output1)
model.summary()


#3. 컴파일, 훈련
#loss = mae or mse optimizer= 'adam', matrix[mae or mse]
import time
model.compile(loss='mae', optimizer='adam',)
from tensorflow.keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor='val_loss', mode='min',
                              patience=5, restore_best_weights=True, verbose=1) 
start = time.time()
hist = model.fit(x_train, y_train, epochs=3000, batch_size=32, validation_split=(0.2), callbacks=[earlystopping])
end = time.time()
print("걸린시간 : ", end - start)


#4. 평가, 예측
#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
#verbose 1  걸린시간
print("=================================")
print(hist)  
print("==================================")
print(hist.history)      # dictionary 키 value 형태로 되어 있다. list형태 2개이상 /반환값 안에는  loss와 valloss의 dictionary히스토리에 제공된 변수가 있다는 뜻 
print("==================================")
print(hist.history['loss'])      
print("==================================")
print(hist.history['val_loss'])     

# import matplotlib.pyplot as plt

# plt.figure(figsize=(9,6))     #.리스트 형태로 순서대로 되어 있는 것은  x를 명시 안해도 상관없다. 즉, y 만 넣어주면 됨
# plt.plot(hist.history['loss'], c= 'red', marker='.', label='loss')
# plt.plot(hist.history['val_loss'], c= 'blue', marker='.', label='val_loss')
# plt.grid() #격자
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.title('kaggle_bike loss')
# plt.legend()   # 라벨이 명시됨
# #plt.legend(loc='upper left')   
# plt.show()


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
submission.to_csv(path +"submission_0111806.csv")

"""
결과
loss :  105.96356201171875
RMSE :  153.67648720538205


스케일러 후 
MinMaxScaler()   = loss :  105.15605926513672
                   RMSE :  151.8773600528591
                   
StandardScaler()   =loss :  102.59281158447266
                    RMSE :  151.88993582764044



 """