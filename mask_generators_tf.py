import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)

# 包括各种产生 mask 的方式, mask 是一个与原图维度一致的 bool 矩阵. bool值为false地方不变，为true的地方置0
# 根据mask 用 new_img = tf.where(mask, tf.zeros_like(img), img) 可以将mask为true的区域像素值置0.


class MCARGenerator:
    """ 返回一个mask, 元素是0或1. 为0的概率为1-p, 为1的概率为p.
        每个位置单独采样, 每个位置元素采样至一个伯努利分布
        p 代表mask中元素为 true 的概率，代表未来将被去掉的像素比例.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, batch):
        assert batch.shape[-1] == 3
        nan_mask = tf.to_float(tf.is_nan(batch)).numpy()                  # 一般是一个全0矩阵
        # 构造一个 batch.size 的矩阵，元素是0或1. 为0的概率为1-p, 为1的概率为p
        bernoulli_mask = np.random.choice(
                        2, size=batch.shape.as_list(), p=[1 - self.p, self.p])
        mask = np.maximum(bernoulli_mask.astype(np.float32), nan_mask)
        assert list(mask.shape) == batch.shape.as_list()
        return mask


class ImageMCARGenerator:
    """ 产生一个通道的mask, shape=(batch,w,h,1), 随后复制到多个通道 -> (batch,w,h,3).
        用在图像中可以保证某个像素位 (1)取到RGB值 或 (2)RGB都置0.
        p 代表mask中元素为 true 的概率，代表未来将被去掉的像素比例.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, batch):
        assert batch.shape[-1] == 3
        bernoulli_mask = np.random.choice(
                2, size=batch.shape.as_list()[:-1], p=[1-self.p, self.p])  # (b, w, h)
        mask = np.stack([bernoulli_mask]*3, axis=-1).astype(np.float32)    # (b, w, h, 3)
        assert list(mask.shape) == batch.shape.as_list()
        return mask


class FixedRectangleGenerator:
    """ mask在矩阵区域 (x1, y1) and (x2, y2) 中的像素置 true. 其余位置为 false.
        在图像中应用该mask, 将使矩阵区域内的像素置0, 其余像素不变
    """
    def __init__(self, x1, y1, x2, y2):
        assert x1 < x2 and y1 < y2
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, batch):
        assert batch.shape[-1] == 3 and len(batch.shape.as_list()) == 4
        mask = np.zeros_like(batch)
        mask[:, self.x1: self.x2, self.y1: self.y2, :] = 1
        assert list(mask.shape) == batch.shape.as_list()
        return mask.astype(np.float32)


class GFCGenerator:
    """
    调用 FixedRectangleGenerator 产生多个Generator, 随后作为 MixtureMaskGenerator 的参数.
    这些 mask 的参数都是固定好的, 是针对 128*128 的人脸图像设计的.
    Generate equal-probably masks O1-O6 from the paper: Generative face completion. CVPR16
    """
    def __init__(self):
        gfc_o1 = FixedRectangleGenerator(52, 33, 116, 71)
        gfc_o2 = FixedRectangleGenerator(52, 57, 116, 95)
        gfc_o3 = FixedRectangleGenerator(52, 29, 74, 99)
        gfc_o4 = FixedRectangleGenerator(52, 29, 74, 67)
        gfc_o5 = FixedRectangleGenerator(52, 61, 74, 99)
        gfc_o6 = FixedRectangleGenerator(86, 40, 124, 88)

        # 第一个参数代表 6 个generator, 第二个参数代表每个 mask 的权重.
        self.generator = MixtureMaskGenerator(
            [gfc_o1, gfc_o2, gfc_o3, gfc_o4, gfc_o5, gfc_o6], [1] * 6)

    def __call__(self, batch):
        assert batch.shape[-1] == 3 and len(batch.shape.as_list()) == 4
        return self.generator(batch)


class MixtureMaskGenerator:
    """
        For each object firstly sample a generator according to their weights,
        and then sample a mask from the sampled generator.
    """
    def __init__(self, generators, weights):
        # weights的维度等于generator的个数
        self.generators = generators
        self.weights = np.array(weights, dtype='float')

    def __call__(self, batch):
        assert batch.shape[-1] == 3 and len(batch.shape.as_list()) == 4
        # 以 w 为权重, 产生等于 bach.shape[0] 即样本个数的随机数. 代表每个样本应该应用哪个 generator.
        w = self.weights / np.sum(self.weights)
        c_ids = np.random.choice(w.size, size=batch.shape[0], replace=True, p=w)
        mask = np.zeros_like(batch)
        for ix, gen in enumerate(self.generators):
            ids = np.where(c_ids == ix)[0]     # 找到 c_ids 等于 i 的序号
            # print("ix:", ix, "\n batch:", batch.shape, "\n w:", w, "\n c_ids:", c_ids, "\n ids:", ids)
            if len(ids) == 0:
                continue
            img_batch = tf.gather(batch, indices=ids, axis=0)
            samples = gen(img_batch)
            mask[ids] = samples
            # mask = tf.scatter_update(ref=tf.Variable(mask), indices=ids, updates=mask_batch)
        return mask


class RectangleGenerator:
    """  产生随机的方形区域 mask. 面积大小在  [min_rect_rel_square, max_rect_rel_square] 之间
    """
    def __init__(self, min_rect_rel_square=0.3, max_rect_rel_square=1):
        self.min_rect_rel_square = min_rect_rel_square
        self.max_rect_rel_square = max_rect_rel_square

    def gen_coordinates(self, width, height):
        # 产生4个坐标
        x1, x2 = np.random.randint(low=0, high=width, size=2)
        y1, y2 = np.random.randint(low=0, high=height, size=2)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        return int(x1), int(y1), int(x2), int(y2)

    def __call__(self, batch):
        assert batch.shape[-1] == 3 and len(batch.shape.as_list()) == 4
        batch_size, width, height, _ = batch.shape.as_list()
        mask = np.zeros_like(batch)
        for ix in range(batch_size):
            x1, y1, x2, y2 = self.gen_coordinates(width, height)
            sqr = width * height
            # 确保产生的 mask 的面积要在限定的范围内
            while not (self.min_rect_rel_square * sqr <=
                       (x2 - x1 + 1) * (y2 - y1 + 1) <=
                       self.max_rect_rel_square * sqr):
                x1, y1, x2, y2 = self.gen_coordinates(width, height)
            mask[ix, x1: x2 + 1, y1: y2 + 1, :] = 1

        return mask


class RandomPattern:
    def __init__(self, max_size=10000, resolution=0.06, density=0.25, update_freq=1, seed=239):
        self.max_size = max_size
        self.resolution = resolution
        self.density = density
        self.update_freq = update_freq
        self.rng = np.random.RandomState(seed)
        self.regenerate_cache()

    def regenerate_cache(self):
        """
        Resamples the big matrix and resets the counter of the total
        number of elements in the returned masks.
        """
        low_size = int(self.resolution * self.max_size)
        low_pattern = self.rng.uniform(0, 1, size=(low_size, low_size)) * 255
        low_pattern = torch.from_numpy(low_pattern.astype('float32'))
        pattern = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize(self.max_size, Image.BICUBIC),
                        transforms.ToTensor(),
        ])(low_pattern[None])[0]
        pattern = torch.lt(pattern, self.density).byte()
        self.pattern = pattern.byte()
        self.points_used = 0

    def __call__(self, batch, density_std=0.05):
        batch_size, width, height, num_channels = batch.shape.as_list()
        res = np.zeros_like(batch)
        idx = list(range(batch_size))
        while idx:
            nw_idx = []
            x = self.rng.randint(0, self.max_size - width + 1, size=len(idx))
            y = self.rng.randint(0, self.max_size - height + 1, size=len(idx))
            for i, lx, ly in zip(idx, x, y):
                res[i] = self.pattern[lx: lx + width, ly: ly + height][:, :, None]
                coverage = float(res[i, 0].mean())
                if not (self.density - density_std < coverage < self.density + density_std):
                    nw_idx.append(i)
            idx = nw_idx
        self.points_used += batch_size * width * height
        if self.update_freq * (self.max_size ** 2) < self.points_used:
            self.regenerate_cache()
        return res


class SIIDGMGenerator:
    """
    Generate equiprobably masks from the paper
    Yeh, R. A., Chen, C., Yian Lim, T., Schwing,
    A. G., Hasegawa-Johnson, M., & Do, M. N.
    Semantic Image Inpainting with Deep Generative Models.
    Conference on Computer Vision and Pattern Recognition, 2017.
    ArXiv link: https://arxiv.org/abs/1607.07539

    Note, that this generator works as supposed only for 128x128 images.
    In the paper authors used 64x64 images, but here for the demonstration
    purposes we adapted their masks to 128x128 images.
    """
    def __init__(self):
        # the resolution parameter differs from the original paper because of
        # the image size change from 64x64 to 128x128 in order to preserve
        # the typical mask shapes
        random_pattern = RandomPattern(max_size=10000, resolution=0.03)
        # the number of missing pixels is also increased from 80% to 95%
        # in order not to increase the amount of the observable information
        # for the inpainting method with respect to the original paper
        # with 64x64 images
        mcar = ImageMCARGenerator(0.95)
        # 切分中心或切分1/4
        center = FixedRectangleGenerator(32, 32, 96, 96)
        half_01 = FixedRectangleGenerator(0, 0, 128, 64)
        half_02 = FixedRectangleGenerator(0, 0, 64, 128)
        half_03 = FixedRectangleGenerator(0, 64, 128, 128)
        half_04 = FixedRectangleGenerator(64, 0, 128, 128)

        self.generator = MixtureMaskGenerator([
            center, random_pattern, mcar, half_01, half_02, half_03, half_04
        ], [2, 2, 2, 1, 1, 1, 1])

    def __call__(self, batch):
        return self.generator(batch)


class ImageMaskGenerator:
    """
    Note, that this generator works as supposed only for 128x128 images.
    """
    def __init__(self):
        siidgm = SIIDGMGenerator()
        gfc = GFCGenerator()
        common = RectangleGenerator()
        self.generator = MixtureMaskGenerator([siidgm, gfc, common], [1, 1, 2])

    def __call__(self, batch):
        return self.generator(batch)


if __name__ == '__main__':
    from datasets_tf import build_dataset
    train_data, _, _ = build_dataset()
    mask_generator = ImageMaskGenerator()

    for i, batch in enumerate(train_data):
        print(batch)
        # mask_generator = ImageMaskGenerator()
        # mask_generator = RandomPattern()
        mask = mask_generator(batch)       # (16, 128, 128, 3)
        print("mask:", mask.shape)
        for j in range(2):
            plt.imshow(mask[j])
            plt.title(str(i)+"+"+str(j))
            plt.show()
            plt.close()
