import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from keras.preprocessing import image


x_train = np.load('C:/dogs-vs-cats/dvc_x_train.npy')
y_train = np.load('C:/dogs-vs-cats/dvc_y_train.npy')
x_test = np.load('C:/dogs-vs-cats/dvc_x_test.npy')
y_test = np.load('C:/dogs-vs-cats/dvc_y_test.npy')


print(x_train.shape, x_test.shape)  #(100, 150, 150, 3)
print(y_train.shape)  #(100,)


# 2 모델구성
model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(150, 150, 3)))
model.add(Conv2D(80, (3,3), activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Conv2D(50, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(30, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

#3 컴파일 훈련 
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'])

# hist = model.fit_generator(x_train, y_train, #steps_per_epoch=100,
#                            epochs=10,
#                            # batch_size=16, 
#                            validation_data=[x_test, y_test], 
#                            # validation_steps=4 
#                            )   

hist = model.fit(x_train, y_train, 
                 #steps_per_epoch=16, 
                 epochs=50, batch_size=10,
                 validation_data=[x_test, y_test],
                    # validation_steps=4,   
                    )

test_image = image.load_img('c:/dogs-vs-cats/test1/4.jpg', target_size =(64, 64))
test_image =image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis =0)
result = model.predict(test_image)
test_image.class_indices
if result[0][0] == 1:
  prediction = 'dog'
else:
  prediction = 'cat'

print(prediction)

# model.save('cat_and_dog.h5')

# accuracy = hist.history['acc']
# val_acc = hist.history['val_acc']
# loss = hist.history['loss']
# val_loss = hist.history['val_loss']

# print('loss :', loss[-1])    
# print('val_loss :', val_loss[-1])   
# print('accuracy :', accuracy[-1])
# print('val_acc :', val_acc[-1])
