import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,Conv2D,MaxPool2D
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from class_data import data_all,r_square

#数据读取
my_data=data_all(r'data\train') #data_all将训练集数据集和标签集读入
data=[]
label=[]
for i in range(len(my_data)):
    data.append(my_data[i][0])
    label.append(my_data[i][1])

#数据预处理
for i in range(len(data)):
    data[i]=np.array(data[i])//200.0 #由于传输手段的问题导致出现噪音，这里将数据集的像素值除以200.0而不是255.0，能去除大部分噪音
    data[i]=1-data[i]

#打乱顺序
indexes = list(range(len(data)))
random.shuffle(indexes)
data = [data[i] for i in indexes]
label = [label[i] for i in indexes]

data = np.array(data)
data= np.expand_dims(data, axis=-1)
label=np.array(label)
label=to_categorical(label, num_classes=10)
data_train, data_test,label_train,label_test = train_test_split(data,label,test_size=0.2,random_state=42)

model=Sequential([
   Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(16, 16, 1),padding='same'),
   MaxPool2D(pool_size=(2, 2)),
   Dropout(0.5),
   Conv2D(128, kernel_size=(3, 3), activation='relu',padding='same'),
   MaxPool2D(pool_size=(2, 2)),
   Dropout(0.5),
   Conv2D(256, kernel_size=(3, 3), activation='relu',padding='same'),
   MaxPool2D(pool_size=(2, 2)),
   Dropout(0.5),

   Flatten(),
   Dense(128, activation='relu'),
   Dense(64, activation='relu'),
   Dense(32, activation='relu'),
   Dense(10,activation='softmax')
])

# 保存最佳模型
checkpoint_filepath =r'mymodel\best_model2.keras'
checkpoint_callback = ModelCheckpoint(checkpoint_filepath,monitor='val_loss',verbose=1,save_best_only=True,mode='min')
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(data_train,label_train,epochs=50,batch_size=16,validation_data=(data_test, label_test),callbacks=[checkpoint_callback])

#读取测试集数据并进行预处理
data_test=data_all(r"data\test")
data1=[]
label1=[]
for i in range(len(data_test)):
    data1.append(data_test[i][0])
    label1.append(data_test[i][1])
for i in range(len(data1)):
    data1[i]=np.array(data1[i])//200.0
    data1[i]=1-data1[i]

data1 = np.array(data1)
data1= np.expand_dims(data1, axis=-1)
label1=np.array(label1)
label1=to_categorical(label1, num_classes=10)

#测试模型
scores=model.evaluate(data1,label1,verbose=0)
print(scores)
r_square(data1,label1,model,types=1)
