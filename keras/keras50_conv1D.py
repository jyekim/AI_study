#카피는 49-2
#return_sequence 엮기 
#시계열 데이터를 할때 이 데이터를 어떻게 자를 것인지 
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, GRU
from tensorflow.keras.layers import Conv1D

#1. 데이터 
a = np.array(range(1, 101))

x_predict = np.array(range(96, 106))  #나올 수 있는 예상 y =100, 107

timesteps = 5  #x는 4개 y는 1개 

def split_x(dataset, timesteps):
    aaa =[]                           #빅리스트 만든것 
    for i in range(len(dataset) - timesteps + 1):   #5-3+1 만큼 반복 
        subset = dataset[i : ( i + timesteps)]          # a[0:(0+3)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a,timesteps)
print(bbb)
print(bbb.shape)  #(96, 5)  

x = bbb[:,:-1]
y = bbb[:,-1]




x=x.reshape(96,4,1)
print (x, y)
print(x.shape, y.shape)  # (96, 4) (96,)


x_predict=split_x(x_predict,4)
print(x_predict)   #(7,4)

x=x.reshape(96,4,1)
# x_predict=x_predict.reshape(7,4,1)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(
    x, y, shuffle=True, random_state=1234) 

print(x_train.shape, y_train.shape)   #(72, 4, 1) (72,)
print(x_test.shape, y_test.shape)    #(24, 4, 1) (24,)
print(x_predict.shape)

x_train = x_train.reshape (72, 4, 1)
x_test = x_test.reshape (24, 4, 1)
x_predict= x_predict.reshape(7,4,1)



#2. 모델 구성
model = Sequential()
# model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=(4,1))) #양방향을 쓰겠다는 뜻임 그래서 
# model.add(LSTM(64))
model.add(Conv1D(100, 2, input_shape=(4,1)))
model.add(Dense(10, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(1))

model.summary()


#3. 컴파일 훈련 
from tensorflow.keras.callbacks import EarlyStopping    
es = EarlyStopping(monitor='val_loss', mode='min', 
                              patience=10, restore_best_weights=True, verbose=1)

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1,callbacks=[es])

#4. 평가예측

# loss = model.evaluate(x_test, y_test)
# print(loss) 
y_pred =x_predict.reshape(7, 4, 1)    
result =model.predict(y_pred)
print('[]의 결과 : ', result)

"""

결과 
bidirection 쓸 때 :
[]의 결과 :  [[100.57158]
 [101.23785]
 [101.88086]
 [102.50067]
 [103.09738]
 [103.67123]
 [104.22251]]

LSTM 쓸 때 :
[]의 결과 :  [[ 99.98472 ]
 [100.985374]
 [101.98609 ]
 [102.98692 ]
 [103.98783 ]
 [104.98881 ]
 [105.9899  ]]

"""