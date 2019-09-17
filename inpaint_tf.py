from argparse import ArgumentParser
from VAEAC_tf import VAEAC
from celeba_model.model_tf import ProposalNetwork, PriorNetwork, GenerativeNetwork
from mask_generators_tf import ImageMaskGenerator
from datasets_tf import build_dataset
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)


parser = ArgumentParser(description='Inpaint images using a given model.')
parser.add_argument('--model_dir', type=str, default="result/")
parser.add_argument('--num_samples', type=int, default=3)
args = parser.parse_args()

# build VAEAC on top of the imported networks
model = VAEAC(proposal_network=ProposalNetwork(),
              prior_network=PriorNetwork(),
              generative_network=GenerativeNetwork())


# load model and test_dataset
batch_size = 32
model_path = os.path.join("celeba_model/weights", "model_39.h5")
assert os.path.exists(model_path)
mask_generator = ImageMaskGenerator()

_, valid_data, test_data = build_dataset()
test_iterator = valid_data.make_one_shot_iterator()
batch = test_iterator.get_next()
mask = mask_generator(batch)
vlb, _ = model.batch_vlb(batch, mask)
print("test vlb:", tf.reduce_mean(vlb).numpy())
print("load weights.")
model.load_weights(model_path)
print("load done.")
vlb, _ = model.batch_vlb(batch, mask)
print("test vlb:", tf.reduce_mean(vlb).numpy())

show_samples = 3
plt.figure(figsize=(12, 8))

for idx, batch in enumerate(valid_data):
    ground_truth = batch.numpy()               # (b, 128, 128, 3)
    mask = mask_generator(batch)               # (b, 128, 128, 3)
    img_samples_params = model.generate_samples_params(
                batch, mask, args.num_samples).numpy()  # shape=(b, 128, 128, 6, k)

    for ix in range(batch_size):
        # 输出 ground truth
        ground_truth_img = ground_truth[ix] * 0.5 + 0.5         # (128, 128, 3)
        ground_truth_img_mask = ground_truth_img.copy()         # (128, 128, 3)
        ground_truth_img_mask[mask[ix].astype(np.bool)] = 0.    # (128, 128, 3)

        # 输出 sample
        # ground truth
        ax = plt.subplot(show_samples, args.num_samples+2, (args.num_samples+2)*ix+1)
        ax.set_title("Ground Truth")
        ax.imshow(ground_truth_img)
        # mask
        # ax.imshow(mask[ix])
        ax = plt.subplot(show_samples, args.num_samples+2, (args.num_samples+2)*ix+2)
        ax.set_title("ground_truth_img_mask")
        ax.imshow(ground_truth_img_mask)

        for jx in range(args.num_samples):
            sample_img = img_samples_params[ix, :, :, :3, jx] * 0.5 + 0.5   # 提取均值作为采样值 (128,128,3)
            sample_img[(1.0-mask[ix]).astype(np.bool)] = 0.0    # 将图中 ground truth 部分置0
            sample_img += ground_truth_img_mask                 # (28, 28, 1)
            print(ground_truth_img.shape, ground_truth_img_mask.shape, sample_img.shape)
            # Generate
            ax = plt.subplot(show_samples, args.num_samples+2, (args.num_samples+2)*ix+3+jx)
            ax.set_title("Generative")
            plt.imshow(sample_img)

        if ix == 2:
            plt.suptitle("batch:" + str(idx))
            plt.savefig("result/"+str(idx)+".jpg")
            # plt.show()
            plt.close()
            break

    plt.figure(figsize=(12, 8))
    show_sample = 1

    if idx == 10:
        break

