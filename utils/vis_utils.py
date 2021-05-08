import os

from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model


def summarize_model(model):
    # Summarize the model at hand
    identifier = "{}_{}".format(model.name, id(model))
    print("Summarizing {} ...".format(identifier))
    model.summary()

    # Summarize submodels
    for item in model.layers:
        if isinstance(item, Model):
            summarize_model(item)


def visualize_model(model, output_folder_path):
    # https://github.com/tensorflow/tensorflow/issues/38988#issuecomment-627982453
    model._layers = [  # pylint: disable=protected-access
        item
        for item in model._layers  # pylint: disable=protected-access
        if isinstance(item, Layer)
    ]

    # Visualize the model at hand
    identifier = "{}_{}".format(model.name, id(model))
    identifier = identifier.replace(os.sep, "_")
    print("Visualizing {} ...".format(identifier))
    plot_model(
        model,
        show_shapes=True,
        show_layer_names=True,
        expand_nested=True,
        to_file=os.path.join(output_folder_path, "{}.png".format(identifier)),
    )

    # Visualize submodels
    for item in model.layers:
        if isinstance(item, Model):
            visualize_model(item, output_folder_path)
