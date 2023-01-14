from tensorflow.keras.models import Sequential, Model, load_model    #모델을 붙이면 함수형으로 전환
from tensorflow.keras.layers import Dense, Input         # Input 붙이면 함수형으로 전환
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


path ='./_save/'
# path = '../_save/'
# path = 'c:/study4/_save/'  절대 경로


#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets['target']  #가격 데이터

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.8, shuffle=True, random_state=42)
  

scaler = MinMaxScaler()   #
#scaler = StandardScaler()   #

scaler.fit(x_train)
x_train = scaler.fit_transform(x_train)   #minmaxscaler  
x_test = scaler.transform(x_test)
# x_train = scaler.transform(x_train)     
# x_test = scaler.transform(x_test)


print(x)
print(type(x))  # <class 'numpy.ndarray'>

# print("최소값 : ", np.min(x))       #standardscaler 때는 최소값 최대값 필요없음 그래서 주석처리 
# print("최대값 : ",np.max(x))    



# print(x)
# print(x.shape)  # (506, 13)
# print(y)
# print(y.shape)  # (506,)

# print(dataset.feature_names)
# # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']
# print(dataset.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.8, shuffle=True, random_state=42)
  



# #2. 모델구성(함수형)
input1 = Input(shape=(13,))       #인풋레이어는 
dense1 = Dense(50, activation= 'relu')(input1)
dense2 = Dense(40, activation= 'sigmoid')(dense1)
dense3 = Dense(30, activation= 'relu')(dense2)
dense4 = Dense(20, activation= 'linear')(dense3)
output1 = Dense(1, activation= 'linear')(dense4)
model = Model(inputs=input1, outputs=output1)
model.summary()                     


# model.save(path +'keras29_1_save_model.h5')




#3. 컴파일, 훈련

model.compile(loss='mse', optimizer = 'adam', metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=20,
                              restore_best_weights=True,
                              verbose=1) 

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,
                      filepath= path + 'MCP/keras30_ModelCheckPoint3.hdf5')

hist = model.fit(x_train, y_train, epochs=5000, batch_size=1,     
                validation_split=0.2, callbacks=[es, mcp],
                verbose=1)  


model.save(path +"keras30_ModelCheckPoint3_save_model.hdf5")



          # 훈련을 시킨 다음 모델을 세이브 했으니 모델과 가중치 저장이 가능함 
#0.2552637561855202


# model = load_model(path + 'MCP/keras30_ModelCheckPoint1.hdf5')

#4. 평가, 예측
print("========================1.기본출력 =======================")
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))

# print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2스코어 : ", r2)



print("========================2.load model 출력 =======================")
model2 = load_model(path + 'keras30_ModelCheckPoint3_save_model.hdf5')

y_predict = model2.predict(x_test)

# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))

# print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2스코어 : ", r2)



print("========================3.ModelCheckPoint 출력 =======================")

model3 = load_model(path + 'MCP/keras30_ModelCheckPoint3.hdf5')

y_predict = model3.predict(x_test)

# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))

# print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2스코어 : ", r2)



"""
결과
========================1.기본출력 =======================
4/4 [==============================] - 0s 1ms/step - loss: 54.1879 - mae: 5.1368
loss :  [54.18791961669922, 5.13683557510376]
R2스코어 :  0.2610783062345804
========================2.load model 출력 =======================
R2스코어 :  0.2610783062345804
========================3.ModelCheckPoint 출력 =======================
R2스코어 :  0.2610783062345804


"""









