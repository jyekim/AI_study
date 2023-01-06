import numpy as np                  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense          


#1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))
# 실습 슬라이싱으로 자르기 

x_train = x[:11]   #[ 1  2  3  4  5  6  7  8  9 10 11] #시작이 0 부터임/ 끝에서 시작하는 경우 오른쪽시작에서 -로 빼주면 됨 
x_test = x[10:13]  #[11 12 13]    #전체데이터 셋 내에서 어느정도 잘라준다 예를 들면 꼭 뒤에서 세개 빼거나 앞에서 세개빼는게 아니라 중간숫자를 빼는거
y_train = y[:11]
y_test = y[10:13]
x_validaion = x[13:16]  #[14 15 16]
y_validaion = x[13:16]

print (x_train) #잘 모를때는 출력을 해보기 
print (x_test)
print (y_train) 
print (y_test)
print (x_validaion)
print (y_validaion)
"""
# x_train = np.array(range(1,11))
# y_train = np.array(range(1,11))
# x_test = np.array([11,12,13])
# y_test = np.array([11,12,13])
# x_val = np.array([14,15,16])      #validation data 는 val이라고 줄여서 쓴다
# y_val = np.array([14,15,16]) 
 
 
 #2. 모델

model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3, activation='relu'))
model.add(Dense(1))


#컴파일
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_data=(x_validation, y_validation))


#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([17])
print("17의 예측값 : ", result)
"""