import numpy as np              
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터   와인데이터 
datasets = load_wine()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)     #(178, 13) (178,)
print(y)
print(np.unique(y))         #[0 1 2]     #라벨의 유니크한 값을 찾는거임
print(np.unique(y, return_counts=True))        #(array([0, 1, 2]), array([59, 71, 48], dtype=int64))

from tensorflow.keras.utils import to_categorical 
y = to_categorical(y) 
print(y.shape)  #(178, 3) 


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, test_size=0.2, stratify=y)


scaler = MinMaxScaler()   #
scaler.fit(x_train)
x_train = scaler.fit_transform(x_train)   #minmaxscaler  
x_test = scaler.transform(x_test)


# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)     
# x_test = scaler.transform(x_test)

#2.모델구성 
model= Sequential()
model.add(Dense(100, activation='relu', input_shape=(13, )))
model.add(Dense(50, activation='relu'))
model.add(Dense(82, activation='relu'))
model.add(Dense(511, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(75, activation='linear'))
model.add(Dense(9, activation='linear'))
model.add(Dense(20, activation='linear'))
model.add(Dense(3, activation='softmax')) 

#3.컴파일, 훈련

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', mode='min',
                              patience=30, restore_best_weights=True,
                              verbose=1) 
hist = model.fit(x_train, y_train, epochs=500, batch_size=10,
          validation_split=0.2, callbacks=[earlyStopping],
          verbose=1) 


#4.평가 예측 
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ,loss')  
print('accuracy : ', accuracy)                              

# print(y_test[:5])                                                       
# y_predict = model.predict(x_test[:5]) 
# print(y_predict)                                          

from sklearn.metrics import accuracy_score
import numpy as np                                                            

y_predict = model.predict(x_test)     
y_predict =np.argmax(y_predict, axis=1)   # 가장 큰 값을 찾아내는 것 # 가장 큰 자릿값 뽑아냄   / axis=1 (가로축(행)), axis=0 (세로축(열))
print('y_pred(예측값) :', y_predict)
y_test =np.argmax(y_test, axis=1)     
print('y_test(원래값) : ', y_test)  
print(y_predict)


acc = accuracy_score(y_test, y_predict)     # 소수점 들어가는 실수 형태로 구성// error 발생
print(acc)


"""결과
데이터가 적으니 스케일러 안하는게 더 나음
    accuracy :  0.9166666865348816
    
    스케일러 
 MinMaxScaler :   0.3888888955116272
 StandardScaler :   accuracy :  0.3888888955116272
    
    
"""