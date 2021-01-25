import tensorflow as tf
import tensorflow.keras.layers as tfkl


class ForwardModel(tf.keras.Model):
    def __init__(self, input_shape, hidden):
        super(ForwardModel, self).__init__()
        self.flatten0 = tfkl.Flatten(input_shape=input_shape)
        self.concat0 = tfkl.Concatenate(axis=1)
        self.layer0 = tfkl.Dense(hidden, activation=tfkl.LeakyReLU())
        self.layer1 = tfkl.Dense(hidden, activation=tfkl.LeakyReLU())
        self.layer2 = tfkl.Dense(1)

    def __call__(self, x, y):
        out = self.flatten0(x)
        out = self.concat0([out, y])
        out = self.layer0(out)
        out = self.layer1(out)
        out = self.layer2(out)
        return out
