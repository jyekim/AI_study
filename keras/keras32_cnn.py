from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten      #conv2D 이차원 이미지쪽은 이렇게 함 

model = Sequential()
                                        #(2, 2)는 조각을 내는것      #인풋은(60000, 5, 5, 1)= 데이터, 가로 ,세로, 컬러
model.add(Conv2D(filters=10, kernel_size=(2,2),                     
                 input_shape=(10, 10, 1)))              #(N, 4, 4, 10)  
model.add(Conv2D(5, kernel_size=(2,2)))       #(N, 3, 3, 5) # dense모델을 할때 히든레이어는 사람이 정하듯 output필터 역시 정하면됨  배치사이즈는 훈련의 단위 
model.add(Conv2D(7,(2,2)))       #(N, 3, 3, 5) # dense모델을 할때 히든레이어는 사람이 정하듯 output필터 역시 정하면됨  배치사이즈는 훈련의 단위 
model.add(Conv2D(6, 2))       #(N, 3, 3, 5) # dense모델을 할때 히든레이어는 사람이 정하듯 output필터 역시 정하면됨  배치사이즈는 훈련의 단위 
                   #(batch_size, rows, columns, channels)                                                        
model.add(Flatten())               #주요 특징만 추출되는데 추출된 주요 특징은 2차원 데이터로 이루어져 있지만 Dense와 같이 분류위한 학습레이어에서 21
                                  #(N, 45)
model.add(Dense(units=10))                                  #(N, 10)
                 #인풋은 (batch_size, input_dim) 10 column 특성의 사이즈라는 것을 알 수 있는것
model.add(Dense(4,activation= 'relu'))                    #(input_shape) cov2d 레이어에 입력되는 이미지 크키                #(N, 1)

model.summary()