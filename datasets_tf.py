import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
# 配置
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)
layers = tf.keras.layers


def load_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    image = tf.cast(image, tf.float32)     # 真实图, shape=(218,178,3)
    return image


def center_crop(image, source=(218, 178, 3), target=128):
    """
    Generate center offset for crop window
    :return: Crop window offsets in form of (from_H, to_H, from_W, to_W), example: (8, 120, 29, 141)
    """
    height, width, channel = source

    off_h = np.ceil((height - target) / 2).astype(int)
    off_w = np.ceil((width - target) / 2).astype(int)
    return image[off_h: off_h+target, off_w: off_w+target, :]


def process_normalize(image):
    """规约到 -1到1 之间"""
    return ((image / 255.0) - 0.5) * 2.0


def process_train_valid_test(image_file):
    process_img = load_image(image_file)
    process_img = center_crop(process_img)
    process_img = process_normalize(process_img)
    return process_img


def build_dataset():
    root_dir = 'data_celeba/'
    img_dir = os.path.join(root_dir, 'img_align_celeba')
    partition_file = os.path.join(root_dir, 'list_eval_partition.txt')

    partition = {'train': [], 'valid': [], 'test': []}
    part = {'0': 'train', '1': 'valid', '2': 'test'}

    # 分割数据集 train/val/test
    for line in open(partition_file):
        if not line.strip():
            continue
        filename, part_id = line.strip().split(' ')
        partition[part[part_id]].append(os.path.join(img_dir, filename))
    print("data: Train:", len(partition['train']), "test:", len(partition['test']), "valid:", len(partition['valid']))

    # train dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(partition['train'])
    train_dataset = train_dataset.shuffle(buffer_size=len(partition['train']))
    train_dataset = train_dataset.map(process_train_valid_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size=32)

    # valid dataset
    valid_dataset = tf.data.Dataset.from_tensor_slices(partition['valid'])
    valid_dataset = valid_dataset.shuffle(buffer_size=len(partition['valid']))
    valid_dataset = valid_dataset.map(process_train_valid_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    valid_dataset = valid_dataset.batch(batch_size=32)

    # test dataset
    test_dataset = tf.data.Dataset.from_tensor_slices(partition['test'])
    test_dataset = test_dataset.shuffle(buffer_size=len(partition['test']))
    test_dataset = test_dataset.map(process_train_valid_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size=32)

    return train_dataset, valid_dataset, test_dataset


if __name__ == '__main__':
    train_data, valid_data, test_data = build_dataset()
    test_iterator = test_data.make_one_shot_iterator()
    batch = test_iterator.get_next()
    print(batch)

    # for i, img in enumerate(valid_data):
    #     print(i, img.shape)                      # (batch, 128, 128, 3)
    #     print(img.numpy())
    #     plt.imshow(img[0].numpy() * 0.5 + 0.5)
    #     plt.show()
    #     break















