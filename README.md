## 1. 为什么要保存模型数据？

人生重要的是积累，20岁到了什么程度，在此基础上30岁又达到什么境界，如此积累，不断进步。

你有没有想过，你花半天时间背诵了一页《三字经》，吃了个午饭后，全忘了。

于是，你加大投入，一天一夜背会了整篇《三字经》，结果睡了一觉后又全忘了。

是的，这肯定很痛苦。

同样，对于神经网络而言也一样。

刚刚耗费了200个小时，认识了30万张狗狗的图片，并计算出了他们的特征，能够轻松分辨出哈士奇和狼，结果计算机一断电，它又空白了。

这肯定不行。

因此，训练的结果要及时保存，保存的结果可以随时恢复。

当再次训练时，可以在上次成果的基础上继续累加。

就像一个人一样，研究学问到80岁，已经是满腹经纶了。

## 2. 训练过程的保存

### 2.1 新HelloWorld：fashion_mnist

我们拿人工智能的新HelloWorld来举例子。

原来人工智能的入门例子是mnist手写数据集。

后来改成fashion_mnist这个数据集了。

![fashion-mnist-sprite.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/4f3ee6b17a5f4fd0964915d83a201eb4~tplv-k3u1fbpfcp-watermark.image?)

这个数据集，主要是10个品类的时尚装饰。

标签 | 类     |
| -- | ----- |
| 0  | T恤/上衣 |
| 1  | 裤子    |
| 2  | 套头衫   |
| 3  | 连衣裙   |
| 4  | 外套    |
| 5  | 凉鞋    |
| 6  | 衬衫    |
| 7  | 运动鞋   |
| 8  | 包     |
| 9  | 短靴 |

通过训练它们，让神经网络认识他们，从而遇到一张图可以说出是衬衫还是裤子。

我们的重点主要是讲保存训练数据，所以训练相关的代码一笔带过。


```python
from os import times
from re import T
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2

# 分类名称
class_names = ["T恤/上衣","裤子","套头衫","连衣裙","外套","凉鞋","衬衫","运动鞋","包","短靴"]
# 读取数据集-构建网络模型-进行训练
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images/255.0
test_images = test_images/255.0
# 构建模型
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])
# 配置训练
model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy']
)
# 开始训练
model.fit(train_images, train_labels, epochs=50)
```

运行后是这样的效果：
```
Epoch 1/10
1875/1875 [======] - 1s 602us/step - loss: 0.5273 - accuracy: 0.8111
Epoch 2/10
1875/1875 [======] - 1s 594us/step - loss: 0.3994 - accuracy: 0.8554
Epoch 3/10
1875/1875 [======] - 1s 590us/step - loss: 0.3673 - accuracy: 0.8662
……
```

上面就训练完数据了。

### 2.2 内存中的数据

如果运行了上面的代码，那么其实两个关键数据（网络模型结构、训练的权重）就已经在内存中了，主要存在model中，因为刚刚就是训练的它，程序还没有关闭，它还是活的。

这时候，可以找一个验证数据试一试，这个验证数据是程序从没有见过的，比如说下面的img_2.png这个裤子。

![2021-11-17_103536.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/8050951aeb434b9b919339e6108bae49~tplv-k3u1fbpfcp-watermark.image?)

通过代码识别一下：
```python
img =cv2.imread('./img/img_2.png',0) 
img = img/255.0
img = np.expand_dims(img, 0)
p = model.predict(img)
print(p)
print(class_names[np.argmax(p[0])])
```
结果如下：
```
[[6.5590651e-11 1.0000000e+00 2.8622449e-13 6.2133689e-09 2.7508923e-10
  2.4884808e-19 7.4704088e-12 2.6084349e-25 3.8184945e-13 4.6570902e-23]]
裤子
```
上面输出了10个分类的可能性，其中第2个分类的可能性为100%，第2个分类索引为1。
`class_names = ["T恤/上衣","裤子","套头衫","连衣裙","外套","凉鞋","衬衫","运动鞋","包","短靴"]`

所以它是个裤子。

保存在内存中，很不稳定，关闭程序就丢失了。如果想要使用，必须重新进行训练。

### 2.3 模型保存为json、h5文件

一个训练完成的神经网络，包含结构和权重两个部分。

举个不恰当的例子，把下面这张图比喻成训练好的神经网络。其中的架子部分是每一层神经网络的结构。那么神经网络的权重，就表示里面存放的花盆等物品。

![置物架.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/7517e902b937489bb2937844597e6a07~tplv-k3u1fbpfcp-watermark.image?)

如果想从另一个地方、另一个时间还原上面的这个角落，就需要记录两个东西：一个是容纳物品的框架，另一个是被摆放的物品。对应到神经网络就是结构和权重。

神经网络的结构，可以通过json文件来存储。神经网络的权重，可以通过h5文件存储。

保存起来非常简单，代码如下：
```python
# 保存训练的模型
model_json = model.to_json()
with open('./save/model.json', 'w') as file:
    file.write(model_json)
# 保存训练的权重
model.save_weights('./save/model.h5')
```

当训练完成之后，所有的信息都保存在了model中，但此时只在内存里。通过`model.to_json()`以及`model.save_weights`可以把信息提取成数据持久化到文件里。

### 2.4 读取json、h5文件恢复模型

想使用的时候，也很简单，直接加载并使用就可以：

```python
# 读取训练的模型结果
with open('./save/model.json', 'r') as file:
    model_json_from = file.read()
new_model = keras.models.model_from_json(model_json_from)
new_model.load_weights('./save/model.h5')

# 进行预测
img =cv2.imread('./img/img_2.png',0) 
img = img/255.0
img = np.expand_dims(img, 0)
p = model.predict(img)
print(p)
print(class_names[np.argmax(p[0])])
```

输出的结果，也是：裤子。

同上面2.2章节中预测不同的是，你不用先执行训练的代码了。

这时，你可以新建一个文件单纯只做识别的任务，因为训练好的信息已经存储到json和h5里了，你读取出来使用就可以了。

### 2.5 恢复模型继续训练

如果某次你训练了2天数据，还没有结束，但是你要提着鸡蛋去走亲戚，因为没有人看护服务器，你需要先暂停。正好你看过上面的教程，先把模型保存成json和h5了。

去到亲戚家，住了5天，临走前亲戚送你一些小鸡仔，并且又给你了一些训练数据。你回到家，需要继续训练。

这时候怎么办？

我们看一下。

还是fashion_mnist的训练，我们训练了20轮，训练结果是这样的：

```
Epoch 1/20
1875/1875 [======] - 1s 579us/step - loss: 0.5342 - accuracy: 0.8096
……
Epoch 20/20
1875/1875 [======] - 1s 560us/step - loss: 0.2347 - accuracy: 0.9102
```
训练集的正确率从0.80到0.91。

假设我们中断了训练，并且依照2.3保存了模型。

后来，我们又想继续训练。


```python
# 读取训练的模型结果
with open('./save/model.json', 'r') as file:
    model_json_from = file.read()
new_model = keras.models.model_from_json(model_json_from)
new_model.load_weights('./save/model.h5')

# 训练模型
new_model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy']
)
new_model.fit(train_images, train_labels, epochs=10)
```
训练结果是这样的：

```
Epoch 1/10
1875/1875 [======] - 1s 604us/step - loss: 0.2313 - accuracy: 0.9128
Epoch 2/10
1875/1875 [======] - 1s 602us/step - loss: 0.2275 - accuracy: 0.9136
……
```
我们看到，本次训练第1轮准确率就是0.91。这和上次训练结束时的准确率是对应的。这说明本次是在一定准确率的基础上继续训练的。


以上就是关于模型的保存、恢复、继续训练。


