import numpy as np
import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)
layers = tf.keras.layers


class ResBlock(tf.keras.Model):
    """
    Usual full pre-activation ResNet bottleneck block.
    """
    def __init__(self, outer_dim, inner_dim):
        super(ResBlock, self).__init__()
        data_format = 'channels_last'
        bn_axis = 1 if data_format == 'channels_first' else 3

        self.net = tf.keras.Sequential([
            layers.BatchNormalization(axis=-1),
            layers.LeakyReLU(),
            layers.Conv2D(inner_dim, (1, 1)),
            layers.BatchNormalization(axis=-1),
            layers.LeakyReLU(),
            layers.Conv2D(inner_dim, (3, 3), padding='same'),
            layers.BatchNormalization(axis=-1),
            layers.LeakyReLU(),
            layers.Conv2D(outer_dim, (1, 1), padding='same')])

    def call(self, x):
        return x + self.net(x)


class MLPBlock(tf.keras.Model):
    def __init__(self, inner_dim):
        super(MLPBlock, self).__init__()
        self.net = tf.keras.Sequential([
                layers.BatchNormalization(),
                layers.LeakyReLU(),
                layers.Conv2D(inner_dim, (1, 1))])

    def call(self, x):
        return x + self.net(x)


class MemoryLayer(tf.keras.Model):
    storage = {}

    def __init__(self, idx, output_bool=False, add_bool=False):
        super(MemoryLayer, self).__init__()
        self.idx = idx
        self.output_bool = output_bool
        self.add_bool = add_bool

    def call(self, x):
        if not self.output_bool:
            self.storage[self.idx] = x
            return x
        else:
            if self.idx not in self.storage:
                err = 'MemoryLayer: idx \'%s\' is not initialized. '
                raise ValueError(err)
            stored = self.storage[self.idx]
            if not self.add_bool:
                data = tf.concat([x, stored], axis=-1)
            else:
                data = x + stored
            return data


class ProposalNetwork(tf.keras.Model):
    def __init__(self):
        super(ProposalNetwork, self).__init__()
        self.net1 = tf.keras.Sequential([
            layers.Conv2D(8, 1),
            ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8)])

        self.net2 = tf.keras.Sequential([
            layers.AveragePooling2D(2, 2),
            ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8)])

        self.net3 = tf.keras.Sequential([
            layers.AveragePooling2D(2, 2), layers.Conv2D(16, 1),
            ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8)])

        self.net4 = tf.keras.Sequential([
            layers.AveragePooling2D(2, 2), layers.Conv2D(32, 1),
            ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16)])

        self.net5 = tf.keras.Sequential([
            layers.AveragePooling2D(2, 2), layers.Conv2D(64, 1),
            ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32)])

        self.net6 = tf.keras.Sequential([
            layers.AveragePooling2D(2, 2), layers.Conv2D(128, 1),
            ResBlock(128, 64), ResBlock(128, 64),
            ResBlock(128, 64), ResBlock(128, 64)])

        self.net7 = tf.keras.Sequential([
            layers.AveragePooling2D(2, 2), layers.Conv2D(256, 1),
            ResBlock(256, 128), ResBlock(256, 128),
            ResBlock(256, 128), ResBlock(256, 128)])

        self.net8 = tf.keras.Sequential([
            layers.AveragePooling2D(2, 2), layers.Conv2D(512, 1),
            MLPBlock(512), MLPBlock(512), MLPBlock(512), MLPBlock(512)])

    def call(self, x):
        # 当输入是 (None, 128, 128, 6), 各层的维度如下:
        x = self.net1(x)   # (16, 128, 128, 8)
        x = self.net2(x)   # (16, 64, 64, 8)
        x = self.net3(x)   # (16, 32, 32, 16)
        x = self.net4(x)   # (16, 16, 16, 32)
        x = self.net5(x)   # (16, 8, 8, 64)
        x = self.net6(x)   # (16, 4, 4, 128)
        x = self.net7(x)   # (16, 2, 2, 256)
        x = self.net8(x)   # (16, 1, 1, 512)
        return x


class PriorNetwork(tf.keras.Model):
    def __init__(self):
        super(PriorNetwork, self).__init__()
        self.net1 = tf.keras.Sequential([
            layers.Conv2D(8, 1),
            ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8)])

        self.net2 = tf.keras.Sequential([
            layers.AveragePooling2D(2, 2),
            ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8)])

        self.net3 = tf.keras.Sequential([
            layers.AveragePooling2D(2, 2), layers.Conv2D(16, 1),
            ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8)])

        self.net4 = tf.keras.Sequential([
            layers.AveragePooling2D(2, 2), layers.Conv2D(32, 1),
            ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16)])

        self.net5 = tf.keras.Sequential([
            layers.AveragePooling2D(2, 2), layers.Conv2D(64, 1),
            ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32)])

        self.net6 = tf.keras.Sequential([
            layers.AveragePooling2D(2, 2), layers.Conv2D(128, 1),
            ResBlock(128, 64), ResBlock(128, 64),
            ResBlock(128, 64), ResBlock(128, 64)])

        self.net7 = tf.keras.Sequential([
            layers.AveragePooling2D(2, 2), layers.Conv2D(256, 1),
            ResBlock(256, 128), ResBlock(256, 128),
            ResBlock(256, 128), ResBlock(256, 128)])

        self.net8 = tf.keras.Sequential([
            layers.AveragePooling2D(2, 2), layers.Conv2D(512, 1),
            MLPBlock(512), MLPBlock(512), MLPBlock(512), MLPBlock(512)])

        # memory layer
        self.mem0 = MemoryLayer(idx="#0", output_bool=False)
        self.mem1 = MemoryLayer(idx="#1", output_bool=False)
        self.mem2 = MemoryLayer(idx="#2", output_bool=False)
        self.mem3 = MemoryLayer(idx="#3", output_bool=False)
        self.mem4 = MemoryLayer(idx="#4", output_bool=False)
        self.mem5 = MemoryLayer(idx="#5", output_bool=False)
        self.mem6 = MemoryLayer(idx="#6", output_bool=False)
        self.mem7 = MemoryLayer(idx="#7", output_bool=False)

    def call(self, x):
        x = self.mem0(x)
        x = self.net1(x)
        x = self.mem1(x)

        x = self.net2(x)
        x = self.mem2(x)

        x = self.net3(x)
        x = self.mem3(x)

        x = self.net4(x)
        x = self.mem4(x)

        x = self.net5(x)
        x = self.mem5(x)

        x = self.net6(x)
        x = self.mem6(x)

        x = self.net7(x)
        x = self.mem7(x)

        x = self.net8(x)
        return x


class GenerativeNetwork(tf.keras.Model):
    def __init__(self):
        super(GenerativeNetwork, self).__init__()
        self.net1 = tf.keras.Sequential([
            layers.Conv2D(256, 1),
            MLPBlock(256), MLPBlock(256), MLPBlock(256), MLPBlock(256),
            layers.Conv2D(128, 1), layers.UpSampling2D((2, 2))])

        self.net2 = tf.keras.Sequential([
            layers.Conv2D(128, 1),
            ResBlock(128, 64), ResBlock(128, 64),
            ResBlock(128, 64), ResBlock(128, 64),
            layers.Conv2D(64, 1), layers.UpSampling2D((2, 2))])

        self.net3 = tf.keras.Sequential([
            layers.Conv2D(64, 1),
            ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32),
            layers.Conv2D(32, 1), layers.UpSampling2D((2, 2))])

        self.net4 = tf.keras.Sequential([
            layers.Conv2D(32, 1),
            ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16),
            layers.Conv2D(16, 1), layers.UpSampling2D((2, 2))])

        self.net5 = tf.keras.Sequential([
            layers.Conv2D(16, 1),
            ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8),
            layers.Conv2D(8, 1), layers.UpSampling2D((2, 2))])

        self.net6 = tf.keras.Sequential([
            layers.Conv2D(8, 1),
            ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
            layers.UpSampling2D((2, 2))])

        self.net7 = tf.keras.Sequential([
            layers.Conv2D(8, 1),
            ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
            layers.UpSampling2D((2, 2))])

        self.net8 = tf.keras.Sequential([
            layers.Conv2D(8, 1),
            ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8)])

        self.net9 = tf.keras.Sequential([
            layers.Conv2D(8, 1),
            ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
            layers.Conv2D(6, 1)])
        # all memory layers
        self.mem7 = MemoryLayer(idx="#7", output_bool=True)
        self.mem6 = MemoryLayer(idx="#6", output_bool=True)
        self.mem5 = MemoryLayer(idx="#5", output_bool=True)
        self.mem4 = MemoryLayer(idx="#4", output_bool=True)
        self.mem3 = MemoryLayer(idx="#3", output_bool=True)
        self.mem2 = MemoryLayer(idx="#2", output_bool=True)
        self.mem1 = MemoryLayer(idx="#1", output_bool=True)
        self.mem0 = MemoryLayer(idx="#0", output_bool=True)

    def call(self, x):
        x = self.net1(x)
        x = self.mem7(x)    # #7 concat

        x = self.net2(x)
        x = self.mem6(x)    # #6 concat

        x = self.net3(x)
        x = self.mem5(x)    # #5 concat

        x = self.net4(x)
        x = self.mem4(x)    # #4 concat

        x = self.net5(x)
        x = self.mem3(x)    # #3 concat

        x = self.net6(x)
        x = self.mem2(x)    # #2 concat

        x = self.net7(x)
        x = self.mem1(x)    # #1 concat

        x = self.net8(x)
        x = self.mem0(x)    # #0 concat

        x = self.net9(x)    # (10, 1024, 1024, 6)
        return x


if __name__ == '__main__':
    # Proposal
    print("Proposal network")
    x_np = np.array(np.random.random((10, 1024, 1024, 6)), dtype=np.float32)
    proposal_network = ProposalNetwork()
    x0 = tf.convert_to_tensor(x_np)
    y = proposal_network(x0)                # (10, 8, 8, 512)
    print(y.shape)
    proposal_network.summary()
    print("\n----------------")
    # Prior
    print("prior network")
    x_np = np.array(np.random.random((10, 1024, 1024, 6)), dtype=np.float32)
    prior_network = PriorNetwork()
    x1 = tf.convert_to_tensor(x_np)
    y = prior_network(x1)                   # (10, 8, 8, 512)
    print(y.shape)
    prior_network.summary()
    print("\n----------------")
    # Generative
    print("generative network")
    generative_network = GenerativeNetwork()
    x_np = np.array(np.random.random((10, 8, 8, 256)), dtype=np.float32)
    x2 = tf.convert_to_tensor(x_np)
    y = generative_network(x2)
    print(y.shape)
    generative_network.summary()

