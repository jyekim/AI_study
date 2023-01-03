import numpy as np  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터  
x = np.array([1,2,3,4,5,6,7,8,9,10]) #(10, )
y = np.array(range(10))              #(10, )
 
# 실습: 넘파이 리스트 슬라이싱 7대3으로 자르기  train test 전체 범위가 겹치는 과적합 문제가 생김 
# x_train = x[:-3]    #시작이 0 부터임/ 끝에서 시작하는 경우 오른쪽시작에서 -로 빼주면 됨 
# x_test = x[-3:]     #전체데이터 셋 내에서 어느정도 잘라준다 예를 들면 꼭 뒤에서 세개 빼거나 앞에서 세개빼는게 아니라 중간숫자를 빼는거
# y_train = y[:7]
# y_test = y[7:]

#train test 섞어서 7:3 만들기  힌트:사이킷런
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    test_size=0.3,
    shuffle=True, #shuffle 랜덤하게 섞겠다는 뜻 
    random_state=123 #다음번 동일한 데이터를 쓰기 위해서는 같은 난수를 써야한다
)

# x_train, x_test, y_train, y_test = train_test_split(x, y) 
# train_test_split(x,y, test_size = 0.3, train_size = 0.7) 
# train_test_split(x,y, shuffle=True)  

print ('x_train : ',x_train) 
print ('x_test : ', x_test)
print ('y_train : ', y_train) 
print ('y_test : ', y_test)



#2. 모델구성
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
 

