import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Concatenate, Layer


class SqueezeLayer(Layer):
    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.squeeze(inputs, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config
