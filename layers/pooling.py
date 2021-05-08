"""
References:
https://github.com/tensorflow/models/blob/414b7b7442b58ef9c04d92c23c0f9e8bccfd2ec1/research/delf/delf/python/training/model/resnet50.py#L462-L482
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers.pooling import GlobalPooling2D


def gem_pooling(feature_map, axis, power, threshold=1e-6):
    """Performs GeM (Generalized Mean) pooling.

    See https://arxiv.org/abs/1711.02512 for a reference.

    Args:
        feature_map: Tensor of shape [batch, height, width, channels] for
            the "channels_last" format or [batch, channels, height, width] for the
            "channels_first" format.
        axis: Dimensions to reduce.
        power: Float, GeM power parameter.
        threshold: Optional float, threshold to use for activations.

    Returns:
        pooled_feature_map: Tensor of shape [batch, channels].
    """
    return tf.pow(
        tf.reduce_mean(
            tf.pow(tf.maximum(feature_map, threshold), power), axis=axis, keepdims=False
        ),
        1.0 / power,
    )


class GlobalGeMPooling2D(GlobalPooling2D):

    def __init__(self, initial_power=1.0, trainable_power=True, **kwargs):
        super(GlobalGeMPooling2D, self).__init__(**kwargs)
        self.initial_power = initial_power
        self.trainable_power = trainable_power
        self.reduction_indices = tf.constant(
            [1, 2] if self.data_format == "channels_last" else [2, 3]
        )

    def build(self, input_shape):
        self.power = self.add_weight(  # pylint: disable=attribute-defined-outside-init
            name="power",
            initializer=Constant(self.initial_power),
            trainable=self.trainable_power,
        )
        super(GlobalGeMPooling2D, self).build(input_shape)

    def call(self, inputs):
        return gem_pooling(
            feature_map=inputs, axis=self.reduction_indices, power=self.power
        )

    def get_config(self):
        config = {
            "initial_power": self.initial_power,
            "trainable_power": self.trainable_power,
        }
        base_config = super(GlobalGeMPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class InspectGeMPoolingParameters(Callback):

    def forward(self, model, logs):
        for item in model.layers:
            if isinstance(item, Model):
                self.forward(item, logs)
                continue
            if not isinstance(item, GlobalGeMPooling2D):
                continue
            variable_name = "power"
            logs["{}_{}".format(item.name, variable_name)] = K.get_value(
                getattr(item, variable_name)
            )

    def on_epoch_end(self, epoch, logs=None):  # @UnusedVariable
        self.forward(self.model, logs)
