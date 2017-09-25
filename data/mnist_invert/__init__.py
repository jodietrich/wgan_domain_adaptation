import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./data/mnist_invert/raw')

class SourceDataSampler(object):
    def __init__(self):
        self.shape = [28, 28, 1]

    def __call__(self, batch_size):

        return self.data2img(mnist.train.next_batch(batch_size)[0])

    def get_validation_batch(self, batch_size):
        img = self.data2img(mnist.validation.next_batch(batch_size)[0])
        return img

    def data2img(self, data):
        return np.reshape(data, [data.shape[0]] + self.shape)


class TargetDataSampler(object):
    def __init__(self):
        self.shape = [28, 28, 1]

    def __call__(self, batch_size):

        img = self.data2img(mnist.test.next_batch(batch_size)[0])
        img = self.convert_to_target(img)
        return img


    def get_validation_batch(self, batch_size):

        img = self.data2img(mnist.validation.next_batch(batch_size)[0])
        img = self.convert_to_target(img)
        return img

    def convert_to_target(self, img):

        img = -1.0*img + 1  # invert
        noise = np.random.uniform(0.0, 0.2, img.shape)
        img -= noise
        img[img < 0.0] = 0.0

        return img

    def data2img(self, data):
        return np.reshape(data, [data.shape[0]] + self.shape)
