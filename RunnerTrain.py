import argparse
import numpy as np
import tensorflow as tf

from net.vgg import Tools, VGGNet, Data, PreData


class Runner:

    def __init__(self, data, classifies, learning_rate, **kw):
        self._data = data
        self._type_number = self._data.type_number
        self._image_size = self._data.image_size
        self._image_channel = self._data.image_channel
        self._batch_size = self._data.batch_size
        self._classifies = classifies

        input_shape = [self._batch_size, self._image_size, self._image_size, self._image_channel]
        self._images = tf.placeholder(shape=input_shape, dtype=tf.float32)
        self._labels = tf.placeholder(dtype=tf.int32, shape=[self._batch_size])

        self._logits, self._softmax, self._prediction = classifies(self._images, **kw)
        self._entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._labels, logits=self._logits)
        self._loss = tf.reduce_mean(self._entropy)
        self._solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5).minimize(self._loss)

        self._saver = tf.train.Saver()
        self._sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        pass

    # 训练网络
    def train(self, epochs, save_model, min_loss, print_loss, test, save):
        self._sess.run(tf.global_variables_initializer())
        epoch = 0
        for epoch in range(epochs):
            images, labels = self._data.next_train()
            try:
                loss, _, softmax = self._sess.run(fetches=[self._loss, self._solver, self._softmax],
                                                  feed_dict={self._images: images, self._labels: labels})
                if epoch % print_loss == 0:
                    Tools.print_info("{}: loss {}".format(epoch, loss))
                    pass
            except ValueError as e:
                print(e)
                pass
            # if loss < min_loss:
            #     break
            if epoch % test == 0:
                self.test()
                pass
            if epoch % save == 0:
                self._saver.save(self._sess, save_path=save_model, global_step=epoch)
            pass
        self._saver.save(self._sess, save_path=save_model, global_step=epoch)
        Tools.print_info("{}: train end".format(epoch))
        self.test()
        Tools.print_info("test end")
        pass

    # 测试网络
    def test(self):
        all_ok = 0
        test_epoch = self._data.test_batch_number
        for now in range(test_epoch):
            images, labels = self._data.next_test(now)
            prediction = self._sess.run(fetches=self._prediction, feed_dict={self._images: images})
            all_ok += np.sum(np.equal(labels, prediction))
        all_number = test_epoch * self._batch_size
        Tools.print_info("the result is {} ({}/{})".format(all_ok / (all_number * 1.0), all_ok, all_number))
        pass

    pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-name", type=str, default="vgg_small", help="name")
    parser.add_argument("-epochs", type=int, default=10000, help="train epoch number")
    parser.add_argument("-batch_size", type=int, default=16, help="batch size")
    parser.add_argument("-type_number", type=int, default=2, help="type number")
    parser.add_argument("-image_size", type=int, default=128, help="image size")
    parser.add_argument("-image_channel", type=int, default=3, help="image channel")
    parser.add_argument("-keep_prob", type=float, default=0.7, help="keep prob")
    parser.add_argument("-zip_file", type=str, default="data/face.zip", help="zip file path")
    args = parser.parse_args()

    output_param = "name={},epochs={},batch_size={},type_number={}," \
                   "image_size={},image_channel={},zip_file={},keep_prob={}"
    Tools.print_info(output_param.format(args.name, args.epochs, args.batch_size, args.type_number,
                                         args.image_size, args.image_channel, args.zip_file, args.keep_prob))

    now_train_path, now_test_path = PreData.main(zip_file=args.zip_file, ratio=2)
    now_data = Data(batch_size=args.batch_size, type_number=args.type_number, image_size=args.image_size,
                    image_channel=args.image_channel, train_path=now_train_path, test_path=now_test_path)

    now_net = VGGNet(now_data.type_number, now_data.image_size, now_data.image_channel, now_data.batch_size)

    runner = Runner(data=now_data, classifies=now_net.vgg_10, learning_rate=0.0001, keep_prob=args.keep_prob)
    runner.train(epochs=args.epochs, save_model=Tools.new_dir("model/" + args.name) + "/" + args.name + ".ckpt",
                 min_loss=1e-6, print_loss=100, test=1000, save=2000)

    pass
