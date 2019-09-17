import tensorflow as tf
import tensorflow_probability as tfp
import os
import numpy as np
tfd = tfp.distributions
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)


def normal_parse_params(params, min_sigma=0.0):
    """
    将输入拆分成两份, 分别代表 mean 和 std.
    min_sigma 是对 sigma 最小值的限制
    """
    n = params.shape[0]
    d = params.shape[-1]                    # channel
    mu = params[..., :d // 2]               # 最后一维的通道分成两份, 分别为均值和标准差
    sigma_params = params[..., d // 2:]
    sigma = tf.math.softplus(sigma_params)
    sigma = tf.clip_by_value(t=sigma, clip_value_min=min_sigma, clip_value_max=1e5)
    # 此处是否应该为多元高斯 ?
    distr = tfd.Normal(loc=mu, scale=sigma)
    # distr = tfd.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
    return distr     # proposal 网络的输出 (None,512), mu.shape=(None,256), sigma.shape=(None,256)


def rec_log_prob(ground_truth, distr_params, mask, min_sigma=1e-2):
    # 计算重建误差. distr_params 包含了均值和标准差参数. ground_truth为图像. mask掩码
    # ground_truth.shape=(None,128,128,3),  distr_params.shape=(None,128,128,6)
    distr = normal_parse_params(distr_params, min_sigma)
    probs = distr.prob(ground_truth)
    log_prob = distr.log_prob(ground_truth) * mask
    # print("REC**", tf.reduce_mean(distr.log_prob(ground_truth)), "\n", mask.shape, "\n", tf.reduce_mean(log_prob))
    return tf.reduce_sum(tf.reshape(log_prob, (ground_truth.shape[0], -1)), axis=-1)


# class GaussianLoss(tf.keras.Model):
#     """
#     计算重建误差.
#     Compute reconstruction log probability of groundtruth given
#     a tensor of Gaussian distribution parameters and a mask.
#     Gaussian distribution parameters are output of a neural network
#     without any restrictions, the minimal sigma value is clipped
#     from below to min_sigma (default: 1e-2) in order not to overfit
#     network on some exact pixels.
#
#     The first half of channels corresponds to mean, the second half
#     corresponds to std. See normal_parse_parameters for more info.
#     This layer doesn't work with NaNs in the data, it is used for
#     inpainting. Roughly speaking, this loss is similar to L2 loss.
#     Returns a vector of log probabilities for each object of the batch.
#     """
#     def __init__(self, min_sigma=1e-2):
#         super().__init__()
#         self.min_sigma = min_sigma
#
#     def prob(self, ground_truth, distr_params, mask):
#         # ground_truth.shape=(None,128,128,3),  distr_params.shape=(None,128,128,6)
#         distr = normal_parse_params(distr_params, self.min_sigma)
#         log_prob = distr.log_prob(ground_truth) * mask
#         # print("REC**", tf.reduce_mean(distr.log_prob(ground_truth)), tf.reduce_mean(log_prob))
#         return tf.reduce_sum(tf.reshape(log_prob, (ground_truth.shape[0], -1)), axis=-1)


