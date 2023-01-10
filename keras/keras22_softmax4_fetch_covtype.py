import numpy as np                     
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1.  데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape)   #(581012, 54) (581012,)
print(np.unique(y, return_counts=True))      #(array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510], dtype=int64)

# from tensorflow.keras.utils import to_categorical

# # # Perform one-hot-encoding on the target variable 'y'
# y = to_categorical(y.reshape(-1,1))
# # # Print the new shape of the target variable 'y'
# print(y.shape)

#====쉐이프를 맞추는 작업====== 
import pandas as pd
from sklearn.preprocessing import OneHotEncoder   # Initialize the OneHotEncoder  #첫 번째 방법 
ohe = OneHotEncoder()

# Fit the OneHotEncoder on the target variable 'y'
y = ohe.fit_transform(y.reshape(-1,1)).toarray()  
# Print the new shape of the target variable 'y'
print(y.shape)    #(581012, 7)     


# import pandas as pd                        # 두 번째 방법 
# y = pd.get_dummies(y,drop_first=False)
# y = np.array(y)
# print(y.shape)  # (581012, 7)


# import pandas as pd                       # 세 번째 방법
# y = pd.get_dummies(y)
# y_np = y.to_numpy()



#====쉐이프를 맞추는 작업======= 
# #print(type(y))


#import pandas as pd
#y = pd.get_dummies(y)
# print(type(y))
#힌트 .values .numpy()

print(y)
print(y.shape)  #(581012, 7)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=222, test_size=0.2, stratify=y)

#2. 모델구성 
model = Sequential()
model.add(Dense(5, activation= 'relu', input_shape=(54, )))
model.add(Dense(1000, activation='relu'))
model.add(Dense(430, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(73, activation='relu'))
model.add(Dense(92, activation='relu'))
model.add(Dense(7, activation= 'softmax'))

#3.컴파일, 훈련 
from tensorflow.keras.callbacks import EarlyStopping

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])          
earlystopping = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)
import time
start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=64, 
          validation_split=0.2, callbacks=[earlystopping], verbose=1)
end = time.time()
print('걸린시간 : ', end- start)

#4. 평가 예측 
loss, accuracy = model.evaluate(x_test, y_test)   
print('loss : ', loss)  
print('accuracy : ', accuracy)                              

# print(y_test[:5])                                                       
# y_predict = model.predict(x_test[:5]) 
# print(y_predict)                                          


from sklearn.metrics import accuracy_score
import numpy as np                                                            

y_predict = model.predict(x_test)     
y_predict =np.argmax(y_predict, axis=1)   
print('y_pred(예측값) :', y_predict)
y_test =np.argmax(y_test, axis=1)     # 가장 큰 값을 찾아내는 것 
print('y_test(원래값) : ', y_test)  
print(y_predict)


acc = accuracy_score(y_test, y_predict),
print(acc)


"""
OneHotEncoder (첫번째 방법 썼을때) 
Epoch 00051: early stopping
걸린시간 :  155.87018036842346
3632/3632 [==============================] - 2s 573us/step - loss: 0.5573 - accuracy: 0.7550
loss :  0.5573306679725647
accuracy :  0.7550407648086548
y_pred(예측값) : [6 2 2 ... 1 0 2]
y_test(원래값) :  [6 2 2 ... 1 0 5]
[6 2 2 ... 1 0 2]
(0.7550407476571173,)


OneHotEncoder (첫번째 방법 썼을때) 
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=222, test_size=0.2, stratify=y)
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])          
earlystopping = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)

model.fit(x_train, y_train, epochs=100, batch_size=64, 
          validation_split=0.2, callbacks=[earlystopping], verbose=1)
Epoch 00049: early stopping
걸린시간 :  999.355678319931
3632/3632 [==============================] - 5s 1ms/step - loss: 0.4960 - accuracy: 0.7824
loss :  0.49599209427833557
accuracy :  0.7823894619941711


"""

