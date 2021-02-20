import tensorflow as tf
import numpy as np
import random

class Buffer:
    def __init__(self, x, y, buffer_size, reinit_freq):
        _, indices = tf.math.top_k(y[:, 0], k=128)
        self.y = tf.gather(y, indices, axis=0)
        self.x = tf.gather(x, indices, axis=0)

        self.buffer_size = buffer_size
        self.reinit_freq = reinit_freq
        self.elems = []
        init_x = self.get_init_x(num_samples = self.buffer_size)
        self.add(init_x)

    def __len__(self):
        return len(self.elems)

    def full(self):
        return self.__len__() == self.buffer_size

    def empty(self):
        return self.__len__() == 0

    def add(self, x):
        new_elems = np.split(x, x.shape[0], axis=0)
        self.elems.extend(new_elems)

    def pop(self, num_samples):
        perm = list(range(self.__len__()))
        random.shuffle(perm)
        self.elems = [self.elems[idx] for idx in perm]

        sampled_elems = self.elems[:num_samples]
        self.elems = self.elems[num_samples:]

        sampled_x = np.concatenate(sampled_elems, axis=0)

        reinit_mask = np.random.binomial(1, self.reinit_freq, num_samples)
        sampled_x[reinit_mask > 0.5] = self.get_init_x(num_samples=np.sum(reinit_mask))

        return sampled_x

    def get_init_x(self, num_samples):
        indices = tf.random.uniform(shape=[num_samples], maxval=self.x.shape[0], dtype=tf.int32)
        init_x = tf.gather(self.x, indices, axis=0)
        return init_x

    def get_topk_x(self, num_samples):
        indices = tf.math.top_k(self.y[:, 0], k=num_samples)[1]
        init_x = tf.gather(self.x, indices, axis=0)
        return init_x