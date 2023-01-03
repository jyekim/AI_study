import numpy as np                      
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split 

#1. 데이터 
x = np.array([range(10),range(21,31),range(201,211)])  # range(10) 0부터 10-1까지임  (9,30,210)
# print(range(10))     #ctrl+/ 누르면 그 줄이 주석으로 됨. 나중에 주석 풀려면 동일 단축키 누르면됨
print(x.shape)  # (3,10)
y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]])
print(y.shape) #(2,10)

x = x.T  
print(x.shape) 
y = y.T  
print(y.shape)

#실습 train test split 이용해서 7:3으로 잘려서 모델구현 /소스 완성 
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7, 
    #test_size=0.3,
    shuffle=True, #shuffle 랜덤하게 섞겠다는 뜻 
    random_state=123 #동일한 데이터를 쓰려면 
    )

# print('x_train : ',x_train)
# print('x_test : ',x_test)
# print('y_train : ',y_train)
# print('y_test : ',y_test)

print(x_train)
print(x_test)
print(y_train)
print(y_test)


# #2.모델구현
# model = Sequential()
# model.add(Dense(2,input_dim=3))  
# model.add(Dense(50))
# model.add(Dense(83))
# model.add(Dense(10))
# model.add(Dense(71))
# model.add(Dense(7))
# model.add(Dense(22))
# model.add(Dense(2))

# #3.컴파일
# model.compile(loss='mae', optimizer='adam')
# model.fit(x, y, epochs=170, batch_size=2)

# #4.평가
# loss = model.evaluate(x, y)
# print('loss:',loss)
# result = model.predict([[9, 30, 210]])
# print('[9, 30, 210]의 예측값 : ', result)


""" 결과

epochs=180,
batch_size=2
loss: 0.17091241478919983
[9, 30, 210]의 예측값 :  [[10.07843    1.6593785]]



model.add(Dense(2,input_dim=3))  
model.add(Dense(50))
model.add(Dense(83))
model.add(Dense(10))
model.add(Dense(71))
model.add(Dense(7))
model.add(Dense(22))
model.add(Dense(2))
epochs=170
batch_size=2
loss: 0.15813851356506348
[9, 30, 210]의 예측값 :  [[10.052927  1.555505]]


"""