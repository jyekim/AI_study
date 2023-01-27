# 47-2 카피햇음 시계열 데이터를 할때 이 데이터를 어떻게 자를 것인지 
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv2D, Flatten

a = np.array(range(1, 101))
timesteps = 5
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
print(x, y)

ccc = split_x(a,timesteps-1)
print(ccc.shape)



# x=x.reshape(96,4,1)
# print (x, y)
# print(x.shape, y.shape)  # (96, 4) (96,)


x_predict=split_x(x_predict,4)
print(x_predict)   #(7,4)

x=x.reshape(96,4,1)
# x_predict=x_predict.reshape(7,4,1)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(
    x, y, shuffle=True, random_state=1234) 


# cnn일때 4차원으로 바꿔줘야하기 때문에  
x_train = x_train.reshape (72, 2, 2, 1)                          # LSTM (72, 2, 2)일 때 나온 3차원을ㅇ 4차원으로 변경 
x_test = x_test.reshape (24, 2, 2, 1)
x_predict= x_predict.reshape(7, 2, 2)

print(x_train.shape, y_train.shape)   #(72, 2, 2) (72,)
print(x_test.shape, y_test.shape)    #(24, 2, 2) (24,)
print(x_predict.shape)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(2,2,1), activation='relu')) 
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
model.summary()

#3. 컴파일 훈련 


model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping    
es = EarlyStopping(monitor='val_loss', mode='min', 
                              patience=10, restore_best_weights=True, verbose=1)    
                                            
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, callbacks=[es])

#4. 평가예측

loss = model.evaluate(x_test, y_test)
print(loss) 

result = model.predict(x_predict)
print('[]의 결과 : ', result)
