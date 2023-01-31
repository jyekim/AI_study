import numpy as np  
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255)

test_datagen = ImageDataGenerator(
    rescale= 1./255)                    


xy_train = train_datagen.flow_from_directory(
    'D:/_data/dogs-vs-cats/train/',
    target_size=(150, 150), 
    batch_size=25000,       #배치를 크게 잡아주면 총 데이터수를 알 수 있게 된다. 
    class_mode='binary',
    # class_mode='categorical',   # binary가 아닌 categoricaldms 원핫이 된것. 
    color_mode='rgb', 
    shuffle=True)
    
    
xy_test = test_datagen.flow_from_directory(
  'D:/_data/dogs-vs-cats/test1/',
    target_size=(150, 150), 
    batch_size=12500,
    # class_mode='categorical',
    class_mode='binary',
    color_mode='rgb', 
    shuffle=True
    # Found 
)


print(xy_train)

np.save('C:/dogs-vs-cats/dvc_x_train.npy', arr=xy_train[0][0])
np.save('C:/dogs-vs-cats/dvc_y_train.npy', arr=xy_train[0][1])
np.save('C:/dogs-vs-cats/dvc_x_test.npy', arr=xy_train[0][0])
np.save('C:/dogs-vs-cats/dvc_y_test.npy', arr=xy_train[0][1])


