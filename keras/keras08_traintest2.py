import numpy as np  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. 데이터  
x = np.array([1,2,3,4,5,6,7,8,9,10]) #(10, )
y = np.array(range(10))              #(10, )
 
# 실습: 넘파이 리스트 슬라이싱 7대3으로 자르기  train test 전체 범위가 겹치는 과적합 문제가 생김 
x_train = x[:-3]    #시작이 0 부터임/ 끝에서 시작하는 경우 오른쪽시작에서 -로 빼주면 됨 
x_test = x[-3:]     #전체데이터 셋 내에서 어느정도 잘라준다 예를 들면 꼭 뒤에서 세개 빼거나 앞에서 세개빼는게 아니라 중간숫자를 빼는거
y_train = y[:7]
y_test = y[7:]
print (x_train) #잘 모를때는 출력을 해보기 
print (x_test)
print (y_train) 
print (y_test)


'''
2. 모델구성
model = Sequential()
model.add(Dense(1,input_dim=1))
model.add(Dense(30))
model.add(Dense(220))
model.add(Dense(34))
model.add(Dense(1))

          


#3.컴파일 훈련
model.compile(loss= 'mae', optimizer='adam')
model.fit(x_train, y_train, epochs=10,batch_size=1)

#4.평가 
loss = model.evaluate(x_test ,y_test)
print('loss : ', loss)
result = model.predict([11])
print('[11]의 결과: ', result)
'''

