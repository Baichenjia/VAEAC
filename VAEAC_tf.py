from prob_utils_tf import normal_parse_params, rec_log_prob
from celeba_model.model_tf import ProposalNetwork, PriorNetwork, GenerativeNetwork
from mask_generators_tf import ImageMaskGenerator
from datasets_tf import build_dataset
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


class VAEAC(tf.keras.Model):
    """
    核心模块, 具备以下性质:
    + The batch of objects and the mask of unobserved features have the same shape.
    + prior 和 proposal 的分布在隐变量空间中应该为独立多元正态分布
    The constructor takes:
    + Prior network: p(z|x_{1-b}, b),   Proposal network: q(z|x, b)
      将条件中的二者连接作为输入, 两个网络的输出层不需要进行限制, 因为sigma稍后会由soft-plus函数约束
    + Generative network: p_theta(x_b | z, x_{1 - b}, b)
      采样的 z 作为输入; x_{1-b}和b不作为输入, 原因是二者的信息可以从 prior network 通过memory传递过来.
    + 重建误差. Reconstruction log probability. rec_log_prob is a callable
      which takes (groundtruth, distr_params, mask) as an input
      and return vector of differentiable log probabilities
      p_theta(x_b | z, x_{1 - b}, b) for each object of the batch.
    + Sigma_mu and sigma_sigma are the coefficient of the regularization
      in the hidden space. The default values correspond to a very weak,
      almost disappearing regularization, which is suitable for all
      experimental setups the model was tested on.
    """
    def __init__(self, proposal_network, prior_network,
                 generative_network, sigma_mu=1e4, sigma_sigma=1e-4):
        super().__init__()
        self.proposal_network = proposal_network
        self.prior_network = prior_network
        self.generative_network = generative_network
        self.sigma_mu = sigma_mu
        self.sigma_sigma = sigma_sigma

    def make_observed(self, batch, mask):
        """ 将 batch 中 mask 为 True 的区域置 0.
        """
        mask = tf.cast(mask, tf.bool)
        observed = tf.where(mask, tf.zeros_like(batch), batch)
        return observed

    def make_latent_distributions(self, batch, mask, no_proposal=False):
        """
        根据 batch, mask 输出 proposal 网络和 prior 网络的输出
        No no_proposal is True, return None instead of proposal distribution.
        """
        assert batch.numpy().shape[-1] == 3
        observed = self.make_observed(batch, mask)
        if no_proposal:
            proposal = None
        else:
            # Proposal 网络输入是 原始图像 和 mask
            full_info = tf.concat([batch, mask], axis=-1)        # 在通道上进行连接 (None,128,128,6)
            proposal_params = self.proposal_network(full_info)   # (None,1,1,512)
            proposal = normal_parse_params(proposal_params, 1e-3)
        # Prior 网络输入是 mask之后的图像 和 mask
        prior_params = self.prior_network(tf.concat([observed, mask], axis=-1))  # 在通道上进行连接
        prior = normal_parse_params(prior_params, 1e-3)
        return proposal, prior

    def prior_regularization(self, prior):
        """
            对 prior network 输出的分布进行约束. 在没有该约束的情况下, 模型一般也不会发散.
            该正则项对原损失函数的影响很小, 几乎不影响学习的过程, 推荐使用. 对应于论文 4.3.2 内容
        """
        # print("先验分布均值 shape=", prior.mean().shape)    # (batch_size, 256)
        num_objects = prior.mean().shape[0]
        mu = tf.reshape(prior.mean(), (num_objects, -1))
        sigma = tf.reshape(prior.stddev(), (num_objects, -1))
        mu_regularise = - tf.reduce_sum(mu ** 2, axis=-1) / (2 * (self.sigma_mu ** 2))
        sigma_regularise = tf.reduce_sum(tf.math.log(sigma)-sigma, axis=-1) * self.sigma_sigma
        return mu_regularise + sigma_regularise     # shape=(batch,)

    def reparameterize(self, proposal_dist):
        eps = tf.random.normal(shape=proposal_dist.mean().shape)
        return eps * proposal_dist.stddev() + proposal_dist.mean()

    def batch_vlb(self, batch, mask):
        """ 输出 variational lower bound, 训练目标是最大化该值. 输出维度 (batch,)
        """
        proposal, prior = self.make_latent_distributions(batch, mask)
        prior_regularization = self.prior_regularization(prior)      # (batch,)
        latent = self.reparameterize(proposal)                       # (batch,1,1,256) 重参数化并采样
        rec_params = self.generative_network(latent)                 # (batch,128,128,6)
        rec_loss = rec_log_prob(batch, rec_params, mask)             # (batch,)
        kl = tfp.distributions.kl_divergence(proposal, prior)        # (batch,1,1,256)
        kl = tf.reduce_sum(tf.reshape(kl, (batch.shape[0], -1)), axis=-1)  # (batch,)
        info = {"rec_loss": -rec_loss, "kl_loss": kl, "prior_reg_loss": -prior_regularization}
        return rec_loss - kl + prior_regularization, info             # (batch,)

    def batch_iwae(self, batch, mask, k):
        """ 从 proposal 中采样, 计算似然概率, 减去 KL-divergence, 得到 ELBO.
        """
        proposal, prior = self.make_latent_distributions(batch, mask)
        estimates = []
        for ix in range(k):
            latent = self.reparameterize(proposal)                 # (batch,1,1,256) 重参数化并采样
            rec_params = self.generative_network(latent)           # (batch,128,128,6)
            rec_loss = rec_log_prob(batch, rec_params, mask)       # (batch,)

            prior_log_prob = prior.log_prob(latent)                # (batch,1,1,256)
            prior_log_prob = tf.reshape(prior_log_prob, (batch.shape[0], -1))  # (batch,256)
            prior_log_prob = tf.reduce_sum(prior_log_prob, axis=-1)            # (batch,)

            proposal_log_prob = proposal.log_prob(latent)           # (batch,1,1,256)
            proposal_log_prob = tf.reshape(proposal_log_prob, (batch.shape[0], -1))  # (batch,256)
            proposal_log_prob = tf.reduce_sum(proposal_log_prob, axis=-1)  # (batch,)

            estimate = rec_loss + prior_log_prob - proposal_log_prob   # (batch_size,) elbo=rec-KL
            estimates.append(estimate)

        estimates_tensor = tf.stack(estimates, axis=1)     # (batch_size, k)
        assert len(estimates_tensor.shape) == 2 and estimates_tensor.shape[1] == k
        # 操作相当于在log内除以k, 输出 shape=(batch_size,)
        return tf.math.reduce_logsumexp(estimates_tensor, axis=1) - tf.math.log(float(k))

    def generate_samples_params(self, batch, mask, k=1):
        """
            k 代表采样的个数. 从 prior network 输出分布中采样, 随后输入到 generative network 中采样
        """
        _, prior = self.make_latent_distributions(batch, mask)
        samples_params = []
        for i in range(k):
            latent = self.reparameterize(prior)                # (batch,1,1,256) 重参数化并采样
            sample_params = self.generative_network(latent)    # (batch,128,128,6)
            samples_params.append(sample_params)
        return tf.stack(samples_params, axis=-1)               # (batch,128,128,6,k)

    def generate_reconstructions_params(self, batch, mask, k=1):
        """
        Generate parameters of generative distributions for reconstructions
        from the given batch.
        It makes K latent representation for each object from the batch
        and generate samples from them.
        The second axis is used to index samples for an object, i. e.
        if the batch shape is [n x D1 x D2], then the result shape is
        [n x K x D1 x D2].
        """
        _, prior = self.make_latent_distributions(batch, mask)
        reconstructions_params = []
        for ix in range(k):
            latent = self.reparameterize(prior)                # (batch,1,1,256) 重参数化并采样
            rec_params = self.generative_network(latent)       # (batch,128,128,6)
            reconstructions_params.append(rec_params)
        return tf.stack(reconstructions_params, axis=-1)       # (batch,128,128,6,k)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    model = VAEAC(proposal_network=ProposalNetwork(),
                  prior_network=PriorNetwork(),
                  generative_network=GenerativeNetwork())

    # Dataset
    train_data, valid_data, test_data = build_dataset()

    for i, batch in enumerate(train_data):
        print(i, batch.shape)            # (16, 128, 128, 3)
        # Generate mask
        mask_generator = ImageMaskGenerator()
        mask = mask_generator(batch)
        assert list(mask.shape) == batch.shape.as_list()

        # mask
        # mask = tf.cast(mask, tf.bool)
        # proposal, prior = model.make_latent_distributions(batch, mask)
        # prior_reg_loss = model.prior_regularization(prior)
        vlb, _ = model.batch_vlb(batch, mask)
        print("vlb:", vlb)
        # iwae = model.batch_iwae(batch, mask, k=2)
        # print("iwae:", iwae)

        # img_gen = model.generate_samples_params(batch, mask)
        # print(img_gen[0, :, :, :3, 0].numpy())

        # print(observed.numpy().max(), observed.numpy().min())
        # for ix in range(batch.shape[0]):
        #     plt.subplot(121)
        #     plt.imshow(batch[ix] * 0.5 + 0.5)
        #     plt.subplot(122)
        #     plt.imshow(observed[ix] * 0.5 + 0.5)
        #     plt.title(str(ix))
        #     plt.show()
        #     plt.close()

        break
