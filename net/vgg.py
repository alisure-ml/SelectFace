# encoding: UTF-8
import os
import time
import shutil
import zipfile
import argparse
import numpy as np
from PIL import Image
from glob import glob
import tensorflow as tf
import tensorflow.contrib.layers as tcl


class Tools:
    def __init__(self):
        pass

    @staticmethod
    def print_info(info):
        print(time.strftime("%H:%M:%S", time.localtime()), info)
        pass

    # 新建目录
    @staticmethod
    def new_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    pass


class PreData:

    def __init__(self, zip_file, ratio=2):
        data_path = zip_file.split(".zip")[0]
        self.train_path = os.path.join(data_path, "train")
        self.test_path = os.path.join(data_path, "test")

        if not os.path.exists(data_path):
            f = zipfile.ZipFile(zip_file, "r")
            f.extractall(data_path)

            all_image = self.get_all_images(os.path.join(data_path, data_path.split("/")[-1]))
            self.get_data_result(all_image, ratio, Tools.new_dir(self.train_path), Tools.new_dir(self.test_path))
        else:
            Tools.print_info("data is exists")
        pass

    # 生成测试集和训练集
    @staticmethod
    def get_data_result(all_image, ratio, train_path, test_path):
        train_list = []
        test_list = []

        # 遍历
        Tools.print_info("bian")
        for now_type in range(len(all_image)):
            now_images = all_image[now_type]
            for now_image in now_images:
                # 划分
                if np.random.randint(0, ratio) == 0:  # 测试数据
                    test_list.append((now_type, now_image))
                else:
                    train_list.append((now_type, now_image))
            pass

        # 打乱
        Tools.print_info("shuffle")
        np.random.shuffle(train_list)
        np.random.shuffle(test_list)

        # 提取训练图片和标签
        Tools.print_info("train")
        for index in range(len(train_list)):
            now_type, image = train_list[index]
            shutil.copyfile(image, os.path.join(train_path,
                                                str(np.random.randint(0, 1000000)) + "-" + str(now_type) + ".jpg"))

        # 提取测试图片和标签
        Tools.print_info("test")
        for index in range(len(test_list)):
            now_type, image = test_list[index]
            shutil.copyfile(image, os.path.join(test_path,
                                                str(np.random.randint(0, 1000000)) + "-" + str(now_type) + ".jpg"))

        pass

    # 所有的图片
    @staticmethod
    def get_all_images(images_path):
        all_image = []
        all_path = os.listdir(images_path)
        for one_type_path in all_path:
            now_path = os.path.join(images_path, one_type_path)
            if os.path.isdir(now_path):
                now_images = glob(os.path.join(now_path, '*.jpg'))
                all_image.append(now_images)
            pass
        return all_image

    # 生成数据
    @staticmethod
    def main(zip_file, ratio=2):
        pre_data = PreData(zip_file, ratio)
        return pre_data.train_path, pre_data.test_path

    pass


class Data:
    def __init__(self, batch_size, type_number, image_size, image_channel, train_path, test_path):
        self.batch_size = batch_size

        self.type_number = type_number
        self.image_size = image_size
        self.image_channel = image_channel

        self._train_images = glob(os.path.join(train_path, "*.jpg"))
        self._test_images = glob(os.path.join(test_path, "*.jpg"))

        self.test_batch_number = len(self._test_images) // self.batch_size
        pass

    def next_train(self):
        begin = np.random.randint(0, len(self._train_images) - self.batch_size)
        return self.norm_image_label(self._train_images[begin: begin + self.batch_size])

    def next_test(self, batch_count):
        begin = self.batch_size * (0 if batch_count >= self.test_batch_number else batch_count)
        return self.norm_image_label(self._test_images[begin: begin + self.batch_size])

    def norm_image_label(self, images_list):
        images = []
        for image_path in images_list:
            image_data = np.array(Image.open(image_path).resize((self.image_size, self.image_size)))
            if len(image_data.shape) == 2:
                image_data = np.asarray([image_data, image_data,
                                         image_data]).reshape([self.image_size, self.image_size, 3])
            images.append(image_data.astype(np.float) / 255.0)
        labels = [int(image_path.split("-")[1].split(".")[0]) for image_path in images_list]
        return images, labels

    pass


class CNNNet:
    def __init__(self, type_number, image_size, image_channel, batch_size):
        self._type_number = type_number
        self._image_size = image_size
        self._image_channel = image_channel
        self._batch_size = batch_size
        pass

    # 网络
    def cnn_5(self, input_op, **kw):
        weight_1 = tf.Variable(tf.truncated_normal(shape=[5, 5, self._image_channel, 64], stddev=5e-2))
        kernel_1 = tf.nn.conv2d(input_op, weight_1, [1, 1, 1, 1], padding="SAME")
        bias_1 = tf.Variable(tf.constant(0.0, shape=[64]))
        conv_1 = tf.nn.relu(tf.nn.bias_add(kernel_1, bias_1))
        pool_1 = tf.nn.max_pool(conv_1, ksize=[1, 5, 5, 1], strides=[1, 4, 4, 1], padding="SAME")
        norm_1 = tf.nn.lrn(pool_1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

        weight_2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 64, 128], stddev=5e-2))
        kernel_2 = tf.nn.conv2d(norm_1, weight_2, [1, 1, 1, 1], padding="SAME")
        bias_2 = tf.Variable(tf.constant(0.1, shape=[128]))
        conv_2 = tf.nn.relu(tf.nn.bias_add(kernel_2, bias_2))
        norm_2 = tf.nn.lrn(conv_2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        pool_2 = tf.nn.max_pool(norm_2, ksize=[1, 5, 5, 1], strides=[1, 4, 4, 1], padding="SAME")

        weight_23 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 256], stddev=5e-2))
        kernel_23 = tf.nn.conv2d(pool_2, weight_23, [1, 2, 2, 1], padding="SAME")
        bias_23 = tf.Variable(tf.constant(0.1, shape=[256]))
        conv_23 = tf.nn.relu(tf.nn.bias_add(kernel_23, bias_23))
        norm_23 = tf.nn.lrn(conv_23, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        pool_23 = tf.nn.max_pool(norm_23, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding="SAME")

        reshape = tf.reshape(pool_23, [self._batch_size, -1])
        dim = reshape.get_shape()[1].value

        weight_4 = tf.Variable(tf.truncated_normal(shape=[dim, 192 * 2], stddev=0.04))
        bias_4 = tf.Variable(tf.constant(0.1, shape=[192 * 2]))
        local_4 = tf.nn.relu(tf.matmul(reshape, weight_4) + bias_4)

        weight_5 = tf.Variable(tf.truncated_normal(shape=[192 * 2, self._type_number], stddev=1 / 192.0))
        bias_5 = tf.Variable(tf.constant(0.0, shape=[self._type_number]))
        logits = tf.add(tf.matmul(local_4, weight_5), bias_5)

        softmax = tf.nn.softmax(logits)
        prediction = tf.argmax(softmax, 1)

        return logits, softmax, prediction

    pass


class VGGNet:

    def __init__(self, type_number, image_size, image_channel, batch_size):
        self._type_number = type_number
        self._image_size = image_size
        self._image_channel = image_channel
        self._batch_size = batch_size
        pass

    # 网络
    # keep_prob=0.7
    def vgg_10(self, input_op, **kw):
        first_out = 16

        conv_1_1 = self._conv_op(input_op, "conv_1_1", 3, 3, n_out=first_out, stripe_height=1, stripe_width=1)
        pool_1 = self._max_pool_op(conv_1_1, "pool_1", 2, 2, stripe_height=2, stripe_width=2)

        conv_2_1 = self._conv_op(pool_1, "conv_2_1", 3, 3, n_out=first_out * 2, stripe_height=1, stripe_width=1)
        pool_2 = self._max_pool_op(conv_2_1, "pool_2", 2, 2, stripe_height=2, stripe_width=2)

        conv_3_1 = self._conv_op(pool_2, "conv_3_1", 3, 3, n_out=first_out * 4, stripe_height=1, stripe_width=1)
        pool_3 = self._max_pool_op(conv_3_1, "pool_3", 2, 2, stripe_height=2, stripe_width=2)

        conv_4_1 = self._conv_op(pool_3, "conv_4_1", 3, 3, n_out=first_out * 4, stripe_height=1, stripe_width=1)
        conv_4_2 = self._conv_op(conv_4_1, "conv_4_2", 3, 3, n_out=first_out * 4, stripe_height=1, stripe_width=1)
        pool_4 = self._max_pool_op(conv_4_2, "pool_4", 2, 2, stripe_height=2, stripe_width=2)

        conv_5_1 = self._conv_op(pool_4, "conv_5_1", 3, 3, n_out=first_out * 4, stripe_height=1, stripe_width=1)
        conv_5_2 = self._conv_op(conv_5_1, "conv_5_2", 3, 3, n_out=first_out * 4, stripe_height=1, stripe_width=1)
        pool_5 = self._max_pool_op(conv_5_2, "pool_5", 2, 2, stripe_height=2, stripe_width=2)

        shp = pool_5.get_shape()
        flattened_shape = shp[1].value * shp[2].value * shp[3].value
        reshape_pool_5 = tf.reshape(pool_5, [-1, flattened_shape], name="reshape_pool_5")

        fc_6 = self._fc_op(reshape_pool_5, name="fc_6", n_out=128)
        fc_6_drop = tf.nn.dropout(fc_6, keep_prob=kw["keep_prob"], name="fc_6_drop")

        fc_7 = self._fc_op(fc_6_drop, name="fc_7", n_out=64)
        fc_7_drop = tf.nn.dropout(fc_7, keep_prob=kw["keep_prob"], name="fc_7_drop")

        fc_8 = self._fc_op(fc_7_drop, name="fc_8", n_out=self._type_number)
        softmax = tf.nn.softmax(fc_8)
        prediction = tf.argmax(softmax, 1)

        return fc_8, softmax, prediction

    # 创建卷积层
    @staticmethod
    def _conv_op(input_op, name, kernel_height, kernel_width, n_out, stripe_height, stripe_width):
        n_in = input_op.get_shape()[-1].value
        with tf.name_scope(name=name) as scope:
            kernel = tf.get_variable(scope + "w", shape=[kernel_height, kernel_width, n_in, n_out], dtype=tf.float32,
                                     initializer=tcl.xavier_initializer_conv2d())
            conv = tf.nn.conv2d(input_op, filter=kernel, strides=(1, stripe_height, stripe_width, 1), padding="SAME")
            biases = tf.Variable(tf.constant(0.0, shape=[n_out], dtype=tf.float32), trainable=True, name="b")
            activation = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)
            return activation
        pass

    # 创建全连接层
    @staticmethod
    def _fc_op(input_op, name, n_out):
        n_in = input_op.get_shape()[-1].value
        with tf.name_scope(name) as scope:
            kernel = tf.get_variable(scope + "w", shape=[n_in, n_out], dtype=tf.float32,
                                     initializer=tcl.xavier_initializer())
            biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name="b")
            activation = tf.nn.relu_layer(x=input_op, weights=kernel, biases=biases, name=scope)
            return activation
        pass

    # 最大池化层
    @staticmethod
    def _max_pool_op(input_op, name, kernel_height, kernel_width, stripe_height, stripe_width):
        return tf.nn.max_pool(input_op, ksize=[1, kernel_height, kernel_width, 1],
                              strides=[1, stripe_height, stripe_width, 1], padding="SAME", name=name)

    pass

