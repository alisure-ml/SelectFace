import os
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf

from net.vgg import Tools, VGGNet


class Runner:

    def __init__(self, type_number, image_size, image_channel, classifies, **kw):
        self._type_number = type_number
        self._image_size = image_size
        self._image_channel = image_channel
        self._classifies = classifies

        input_shape = [1, self._image_size, self._image_size, self._image_channel]
        self._images = tf.placeholder(shape=input_shape, dtype=tf.float32)
        self._prediction = classifies(self._images, **kw)[2]

        self._saver = tf.train.Saver()
        self._sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        pass

    # 推理
    def inference(self, ckpt_path, image_path, result_txt):
        self._sess.run(tf.global_variables_initializer())
        tf.train.Saver(var_list=tf.trainable_variables()).restore(self._sess, save_path=ckpt_path)

        # 0:face, 1:no, 2:error, 3:init
        images_info = self._get_image_info(image_path)

        for image_index, image_info in enumerate(images_info):
            try:
                image_data = np.array(Image.open(image_info[1]).resize((self._image_size, self._image_size)))
                if len(image_data.shape) < 3 or image_data.shape[2] is not 3:
                    raise IOError("shape is less 3")
                images = [image_data.astype(np.float) / 255.0]
                result_type = self._sess.run(fetches=[self._prediction], feed_dict={self._images: images})[0][0]
                images_info[image_index][2] = result_type
                Tools.print_info("image {} type is {}".format(image_info[1], result_type))
            except IOError as e:
                images_info[image_index][2] = 2
                Tools.print_info("...... image {} is error".format(image_info[1]))
                pass
            pass

        with open(result_txt, "w") as f:
            f.writelines(["{},{},{}\n".format(s1, s2, s3) for s1, s2, s3 in images_info])
            pass

        for image_index, image_info in enumerate(images_info):
            Tools.print_info("{} {} {} {}".format(image_index, image_info[2], image_info[0], image_info[1]))
            pass

        Tools.print_info("result in {}".format(result_txt))
        pass

    # 得到所有的图片：目录，地址，类别
    def _get_image_info(self, now_path):
        image_info = []
        image_path_files = os.listdir(now_path)
        for path_file in image_path_files:
            now_path_file = os.path.join(now_path, path_file)
            if os.path.isdir(now_path_file):  # 是目录
                image_info.extend(self._get_image_info(now_path_file))  # 递归
            elif ".jpg" in now_path_file or ".png" in now_path_file:  # 是图片
                image_info.append([os.path.basename(now_path), now_path_file, 3])
            pass
        return image_info

    pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-name", type=str, default="vgg_small", help="name")
    parser.add_argument("-type_number", type=int, default=2, help="type number")
    parser.add_argument("-image_size", type=int, default=128, help="image size")
    parser.add_argument("-image_channel", type=int, default=3, help="image channel")
    parser.add_argument("-keep_prob", type=float, default=0.7, help="keep prob")
    args = parser.parse_args()

    output_param = "name={},type_number={},image_size={},image_channel={},keep_prob={}"
    Tools.print_info(output_param.format(args.name, args.type_number, args.image_size,
                                         args.image_channel, args.keep_prob))

    now_net = VGGNet(args.type_number, args.image_size, args.image_channel, 1)

    runner = Runner(args.type_number, args.image_size, args.image_channel, now_net.vgg_10, keep_prob=args.keep_prob)
    runner.inference(ckpt_path="model/vgg_small/vgg_small.ckpt-9999", image_path="images",
                     result_txt="result/result.txt")

    pass
