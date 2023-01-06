import numpy as np                  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from sklearn.model_selection import train_test_split         


#1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))
# 실습 슬라이싱으로 자르기 
# train_test_split 로 나누기 10:3:3 으로 나눠라 
# x_train = x[:11]   #[ 1  2  3  4  5  6  7  8  9 10 11] #시작이 0 부터임/ 끝에서 시작하는 경우 오른쪽시작에서 -로 빼주면 됨 
# x_test = x[10:13]  #[11 12 13]    #전체데이터 셋 내에서 어느정도 잘라준다 예를 들면 꼭 뒤에서 세개 빼거나 앞에서 세개빼는게 아니라 중간숫자를 빼는거
# y_train = y[:11]
# y_test = y[10:13]
# x_validaion = x[13:16]  #[14 15 16]
# y_validaion = x[13:16]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=1234)
# x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, shuffle=False, test_size=0.5, random_state=123)

print (x_train.shape, x_test.shape) #잘 모를때는 출력을 해보기 
print (y_train.shape, y_test.shape) 
# print (x_val)
# print (y_val)




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
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_split=0.25)


#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([17])
print("17의 예측값 : ", result)
