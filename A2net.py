from keras.layers import Activation
from keras.layers import Conv2D
import keras.backend as K
import keras
from keras.layers import Reshape
"""
    Keras Implementation of Double Attention Network. NIPS 2018
"""
def A2net(input):
    channels = input._keras_shape[-1]
    intermediate_dim = channels // 2
    convA = Conv2D(intermediate_dim, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(input)
    convB = Conv2D(intermediate_dim, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(input)
    convV = Conv2D(intermediate_dim, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(input)
    (_, h, w, c) = K.int_shape(convB)
    feature_maps = Reshape((c, h * w))(convA)  # 对 A 进行reshape
    atten_map = Activation('softmax')(convB)
    atten_map = Reshape((h * w, c))(atten_map)# 对 B 进行reshape 生成 attention_aps
    global_descriptors = keras.layers.dot([feature_maps, atten_map], axes=[2,1], normalize=False)  # 特征图与attention_maps 相乘生成全局特征描述子
    atten_vectors = Activation('softmax')(convV)  # 生成 attention_vectors
    atten_vectors = Reshape((h*w, c))(atten_vectors)
    out = keras.layers.dot([atten_vectors, global_descriptors], axes=-1)# 注意力向量左乘全局特征描述子
    out = Reshape((c, h, w))(out)
    out = Reshape((h, w, c))(out)
    out = Conv2D(channels, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(out)
    return out

