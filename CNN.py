from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras import layers

# 这个网络需要的东西：卷积层操作，最大池化，全连接网络分类器


# 输入到网络中的样本宽
# 高
# 最终分类的结果数量
width = 30
height = 30
result_num = 10

# 初始化网络模型，只是一个模板空壳。
model = Sequential()                 

# 第一个CNN网络块 
# 卷积层包含64个大小为3 * 3的filter(core)，使用relu为激活函数，输入大小为 （宽，长，RGB通道数）                
# relu 是最常用的一种激活函数，以前好像是sigmoid（S型的曲线），被relu代替了。                                    
# 池化层采用最大池化的方法，                                                        
# 将原样本的长和宽变为原来的1/2，深度不变。
# 池化可以理解为一种快速的特征提取（相对于卷积操作慢慢地缩减样本尺寸）。
# 有了池化之后，可以减小计算负担（我是这么理解的）
# 另一方面减少了参数数量，减少过拟合的情况。

"""
relu的函数 没记错的话长这个样子，
如果输入值是负值，那输出必为0
如果输入时正值，那通过一个线性函数计算后得到输出。
                       | y轴      +
                       |        +
                       |      +
                       |    +
                       |  +                    
                       |+
-----------------------——————————————————————————> x轴
                       |0
                       |
                       |
                       |

"""

model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(width, height, 1)))     
model.add(layers.MaxPool2D((2, 2)))                      

# 第二个CNN网络块
# 卷积层包含128个大小为3 * 3的filter(core)，使用relu为激活函数，
# 因为第一个CNN块中已经定义了输入大小，所以它之后的卷积层不再需要定义输入尺寸。
# 第二块网络与第一块还有一个区别是添加了一层Dropout。
# Dropout和Data Argumentation（添加数据样本）是处理过拟合的杀手锏，
# Dropout内的参数一般在0.1到0.5之间，
# 至于Dropout为什么可以缓解过拟合，众说纷纭；
model.add(layers.Dropout(0.3))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))                                   
model.add(layers.MaxPool2D((2, 2)))

# 第三个CNN网络块
# 第三个块中应该也可有一层池化
# 是否要添加池化看参数数量，如果数量不多，不加也可以。
model.add(layers.Conv2D(32, (3, 3), activation='relu'))             

# 利用全连接网络实现的分类器
# CNN会弱化网络的分类能力，而全连接网络可以很好地实现这一功能。
# Flatten会将一个多维的张量，压扁变成一个向量。
# 在这个例子中我们的输入时一个3D张量(image_w, image_h, image_channels),
# 经过Flatten的处理之后会变成一个1D张量（向量）
# 全连接网络用到的softmax 激活函数很重要，没有他是实现不了分类的。
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(result_num, activation='softmax'))

"""
与每层卷积层数目都相同的CNN结构相比：金字塔架构更能有效地利用计算资源。
因此，卷积层中的过滤器数量从少到多，能够发挥出更好地性能。
"""



# 配置模型的学习参数，
# optimizer：优化器
# loss：损失函数；
# metrics：设置训练时的输出参数。
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])



"""
参数变化表
原始输入大小:      (30, 30, 1)

                        Output Shape     Parameter Number

Conv2D 1                (28, 28, 16)        12544                   28 = 30 - 3 + 1
MaxPooling 1            (14, 14, 16)        3136                    12 = 28 / 2         MaxPooling 设置为（2,2）即长宽都除以2，求整，有一部分的数据被丢弃了。

Conv2D 2                (12, 12, 32)        4608                    12 = 14 - 3 + 1
MaxPooling 2            (6, 6, 32)          1152

Conv2D 3                (4, 4, 32)          512
MaxPooling 3            (2, 2, 32)          256

Flatten                 (128, )             128
Dense(64)               (64, )              64
Dense result_num        (10, )              10
"""
