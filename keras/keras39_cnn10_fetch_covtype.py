#에러 
import numpy as np                     
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten
from sklearn.preprocessing import MinMaxScaler, StandardScaler

path = './_save/'


#1.  데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape)   #(581012, 54) (581012,)
# print(np.unique(y, return_counts=True))      #(array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510], dtype=int64)

# from tensorflow.keras.utils import to_categorical

# # # Perform one-hot-encoding on the target variable 'y'
# y = to_categorical(y.reshape(-1,1))
# # # Print the new shape of the target variable 'y'
# print(y.shape)

#====쉐이프를 맞추는 작업====== 

#====================1.keras to_categorical 첫번째 방법======================================
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y.reshape(-1,1))
#  # Print the new shape of the target variable 'y'
# print(y.shape)
# print(type(y))
# print(y[:25])
# print(np.unique(y[:,0], return_counts=True))     #  모든 행의 0번째라는 뜻 
# print(np.unique(y[:,1], return_counts=True))     #  모든 행의 0번째라는 뜻 
# print("==========================================")
# y = np.delete(y, 0, axis=1)
# print(y.shape)
# print(y[:10])
# print(np.unique(y[:,0], return_counts=True))     #  모든 행의 0번째라는 뜻 

#====================================================================================


#==========================2.pandas get dummies   두 번째 방법 =======================================
# import pandas as pd                #연산은 잘 되지만  마지막 numpy자료형이 pandas를 못 받아들인다.                
# y = pd.get_dummies(y)
# print(y[:10])
# print(type(y))
# # y =y.values        
# y = y.to_numpy()                  # to_numpy랑  values를 둘 중 골라쓰면 됨  넘파이로 바꿔줄때 사용하는 것 
# print(type(y))                    #<class 'numpy.ndarray'>
# print(y.shape)                    #(581012, 7)



#==========================3.  sklearn OneHotEncoder  세번째 방법=======================
print(y.shape)     #(581012,)  1차원이기때문에 2차원으로 바꿔줘야함 
y = y.reshape(581012,1)
print(y.shape)     #(581012, 1)
from sklearn.preprocessing import OneHotEncoder   
ohe = OneHotEncoder()
# ohe.fit(y)
# y = ohe.transform(y)
y = ohe.fit_transform(y)   #위에 주석처리된 두개를 합친것 
y = y.toarray()             #<class 'numpy.ndarray'>  fittransform , toarray  로 바꿔준다
print(type(y))    
# print(y[:15])
# print(type(y))
# print(y.shape)  #(581012, 7)



#===========================내가 찾은 방법==============================================
# import pandas as pd
# from sklearn.preprocessing import OneHotEncoder   
# ohe = OneHotEncoder()

# # Fit the OneHotEncoder on the target variable 'y'
# y = ohe.fit_transform(y.reshape(-1,1)).toarray()  
# # Print the new shape of the target variable 'y'
# print(y.shape)    #(581012, 7)     


# import pandas as pd                        
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

# print(y)
# print(y.shape)  #(581012, 7)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=222, test_size=0.2, stratify=y)


scaler = MinMaxScaler()   
scaler.fit(x_train)
x_train = scaler.fit_transform(x_train)   #minmaxscaler  
x_test = scaler.transform(x_test)

print(x_train.shape)   #(464809, 54) 
print(x_test.shape)    #(116203, 54)

x_train = x_train.reshape(464809, 54, 1, 1)
x_test = x_test.reshape(116203, 54, 1, 1)
print(x_train.shape, x_test.shape)


# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)     
# x_test = scaler.transform(x_test)

# print(x)
# print(type(x))  




#2. 모델구성 
model = Sequential()
model.add(Conv2D(50, (2,1), input_shape=(54, 1, 1), activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(430, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(250, activation='relu'))
model.add(Dense(73, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(92, activation='relu'))
model.add(Dense(7, activation= 'softmax'))


#2. 모델구성(함수형)
# input1 = Input(shape=(54,))       #인풋레이어는 
# dense1 = Dense(50, activation= 'relu')(input1)
# drop1 = Dropout(0.5)(dense1)
# dense2 = Dense(40, activation= 'sigmoid')(drop1)
# drop2= Dropout(0.3)(dense2) 
# dense3 = Dense(30, activation= 'relu')(drop2)
# drop3 = Dropout(0.2)(dense3)
# dense4 = Dense(20, activation= 'linear')(drop3)
# output1 = Dense(7, activation= 'softmax')(dense4)
# model = Model(inputs=input1, outputs=output1)
# model.summary()



#3.컴파일, 훈련 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])          
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)

import datetime
date = datetime.datetime.now()
print(date)
print(type(date))
date = date.strftime("%m%d_%H%M")
print(date)
print(type(date))

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbpse=1, save_best_only=True,
                      filepath= filepath +'k31_10_' + date + '_'+ filename)


model.fit(x_train, y_train, epochs=50, batch_size=100, 
          validation_split=0.2, callbacks=[es,mcp], verbose=1)


# model.save(path +"keras31_dropout10_save_model.hdf5")

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


'''
cnn 했을 때 : loss :  0.26433998346328735
accuracy :  0.8913539052009583
  
  
  
  
dropout 했을 떄 loss :  0.5446786284446716
              accuracy :  0.7677512764930725





scaler 하기 전 : loss :  0.4136052131652832
                accuracy :  0.8242127895355225


scaler 한 후 :   
Minmax Scaler : loss :  1.2051825523376465
                accuracy :  0.4876036047935486

Standard Scaler : loss :  1.2037795782089233
                  accuracy :  0.4876036047935486
 
 





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




1.keras to_categorical
Epoch 00038: early stopping
걸린시간 :  773.5588660240173
3632/3632 [==============================] - 5s 1ms/step - loss: 0.4531 - accuracy: 0.7999
loss :  0.45311322808265686
accuracy :  0.7999104857444763
y_pred(예측값) : [1 1 1 ... 0 0 1]
y_test(원래값) :  [1 1 0 ... 1 0 1]
[1 1 1 ... 0 0 1]
(0.7999105014500486,)


'''