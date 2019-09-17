from argparse import ArgumentParser
from VAEAC_tf import VAEAC
from celeba_model.model_tf import ProposalNetwork, PriorNetwork, GenerativeNetwork
from mask_generators_tf import ImageMaskGenerator
from datasets_tf import build_dataset
import tensorflow as tf
import os
import numpy as np
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)

parser = ArgumentParser(description='Train VAEAC to in-paint.')
parser.add_argument('--model_dir', type=str, default='celeba_model')
parser.add_argument('--epochs', type=int, default=40, help='number of epochs.')
args = parser.parse_args()

# build VAEAC on top of the imported networks
model = VAEAC(proposal_network=ProposalNetwork(),
              prior_network=PriorNetwork(),
              generative_network=GenerativeNetwork())

# build optimizer and import its parameters
learning_rate = tf.Variable(2e-4, name="learning_rate")
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
batch_size = 32
vlb_scale_factor = 128 ** 2

# import mask generator
mask_generator = ImageMaskGenerator()

# load train and validation datasets
train_data, valid_data, test_data = build_dataset()

# a list of validation IWAE estimates
validation_iwae = []
train_vlb = []


def get_validation_iwae(m, num_samples=10):
    """ 在验证集上计算 IWAE.
    """
    cum_size, avg_iwae = 0., 0.
    for ix, batch in enumerate(valid_data):
        mask = mask_generator(batch)
        iwae = m.batch_iwae(batch, mask, num_samples)  # (batch_size,)
        # 滑动平均. 对 avg 的权重较大, 对当前批量的权值小.
        avg_iwae = avg_iwae * (cum_size / (cum_size + float(iwae.shape.as_list()[0]))) + \
                   tf.reduce_sum(iwae).numpy() / (cum_size + float(iwae.shape.as_list()[0]))
        cum_size += float(iwae.shape.as_list()[0])
        # print("avg_iwae", avg_iwae.numpy(), ", iwae:", iwae.shape, "cum_size:", cum_size)
        if cum_size >= 1024:      # 验证集只使用1024个样本.
            break
    return float(avg_iwae)


# main train loop
loss_best = 1e7
for epoch in range(args.epochs):
    avg_vlb, losses, valid_iwae = [], [], []
    print("\n---------\nEpoch:", epoch)
    for i, batch in enumerate(train_data):
        with tf.GradientTape() as tape:     # train
            mask = mask_generator(batch)
            vlb, info = model.batch_vlb(batch, mask)
            loss = tf.reduce_mean(-1.0*vlb) / vlb_scale_factor  # 损失为 -elbo/factor

        gradients = tape.gradient(loss, model.trainable_variables)
        # gradients, _ = tf.clip_by_global_norm(gradients, 1.)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        if i % 10 == 0:
            print("  i = ", i, ",loss: ", loss.numpy(), ", rec:", tf.reduce_mean(info["rec_loss"]).numpy(),
                  ", kl:", tf.reduce_mean(info["kl_loss"]).numpy(),
                  ", prior_regularization:", tf.reduce_mean(info["prior_reg_loss"]).numpy())

        # validation
        if i % (162770 / 2) == 162770 / 2 - 1:        # 每个epoch执行5次
            print("valid..", end=".", flush=True)
            val_iwae = get_validation_iwae(model)
            valid_iwae.append(val_iwae)
            print("done.")

        # update running variational lower bound average
        avg_vlb.append(float(tf.reduce_mean(vlb).numpy()))
        losses.append(loss.numpy())
        if epoch == 0 and i == 0:    # total parameters: 5,509,054
            print("total parameters:", np.sum([np.prod(v.shape.as_list()) for v in model.trainable_variables]))

    print("Avg train vlb:", np.mean(avg_vlb), ", avg loss:", np.mean(losses), ', valid iwae:', np.mean(valid_iwae))
    if np.mean(losses) < loss_best:
        loss_best = np.mean(losses)
        model.save_weights(os.path.join("celeba_model/weights", "model_best.h5"))
        print("new best loss:", loss_best)
    if epoch % 5 == 4:
        model.save_weights(os.path.join("celeba_model/weights", "model_"+str(epoch)+".h5"))

