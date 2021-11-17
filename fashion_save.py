#In[]
from os import times
from re import T
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os

# 分类名称
class_names = ["T恤/上衣","裤子","套头衫","连衣裙","外套","凉鞋","衬衫","运动鞋","包","短靴"]

# %% 读取数据集-构建网络模型-进行训练
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images/255.0
test_images = test_images/255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy']
)
model.fit(train_images, train_labels, epochs=20)

# %% 保存训练的模型结果
model_json = model.to_json()
with open('./save/model.json', 'w') as file:
    file.write(model_json)
model.save_weights('./save/model.h5')

# %% 读取训练的模型结果
with open('./save/model.json', 'r') as file:
    model_json_from = file.read()
new_model = keras.models.model_from_json(model_json_from)
new_model.load_weights('./save/model.h5')

# %% 进行预测
img =cv2.imread('./img/img_2.png',0) 
img = img/255.0
img = np.expand_dims(img, 0)
p = model.predict(img)
print(p)
print(class_names[np.argmax(p[0])])