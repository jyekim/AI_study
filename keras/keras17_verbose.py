from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1.  데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)   #. (506, 13) (506,)

x_train , x_test, y_train , y_test = train_test_split(
    x, y, shuffle=True, random_state=333, test_size=0.2) #. 506 *0.8 


#2. 모델구성 
model = Sequential()
#model.add(Dense(5, input_dim=13))    #.  input_dim  은 행과열로 있을때만 가능
model.add(Dense(5, input_shape=(13,)))    #.   4차원 같은 경우에는 input_shape로 써야함    위에 주석과 동일함
model.add(Dense(4))   
model.add(Dense(3))    
model.add(Dense(2))    
model.add(Dense(1))  


#.컴파일 ,훈련 
import time
model.compile(loss='mse', optimizer='adam')
start = time.time()
model.fit(x_train, y_train, epochs=50, batch_size=1, 
          validation_split=0.2, verbose=3)   #. verbose는 함수 수행시 발생하는 상세한 정보들을 표준 출력으로 자세히 내보낼것인가 나타냄, 딜레이 시간이 있다. 그걸줄이기 위해 사용하는것
                                            #. 0은 출력하지 않고, 1은 자세히, 2는 함축적인 정보만 출력하는 형태 3은 에포만 나옴
end = time.time()                                                                                     
#3. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

print("걸린시간 : ", end-start)
#verbose 1  걸린시간
