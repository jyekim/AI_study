from sklearn.datasets import load_iris   
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


#1. 데이터

datasets = load_iris()
# print(datasets.DESCR)      # pandas 에서는 .describe() 혹은 .info()
# print(datasets.feature_names)   #pandas .columns 으로 씀 괄호가 

x = datasets.data
y = datasets['target']
# print(x) 
# print(y)
# print(x.shape)
# print(y.shape)   #(150, 4),  (150, )

x_train , x_train, y_train, y_test = train_test_split(x, y, shuffle=False,
                                                     random_state=333, test_size=0.2)   #false의 문제점은?
print(y_train)
print(y_test)

#2. 모델구성
model= Sequential()
model.add(Dense(5),input_dim)
