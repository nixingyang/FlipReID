import os
import urllib
from inspect import isclass, isfunction

import cv2
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.applications.imagenet_utils import (
    decode_predictions,
    preprocess_input,
)
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Input,
    Lambda,
    Layer,
    Softmax,
)
from tensorflow.keras.models import Model, load_model

import applications.common as common
import applications.ibnresnet as ibnresnet
import applications.resnesta as resnesta
import applications.resnet as resnet


class Applications(object):

    def __init__(self, modules=(common, ibnresnet, resnesta, resnet)):
        self.modules = modules
        self.preprocess_input_mode = "torch"

    def get_model_name_to_model_function(self):
        model_name_to_model_function = {}
        for module in self.modules:
            names = module.__all__
            for name in names:
                item = getattr(module, name)
                if isfunction(item):
                    model_name_to_model_function[name] = item
        return model_name_to_model_function

    def get_custom_objects(self):
        custom_objects = {}
        for module in self.modules:
            names = dir(module)
            for name in names:
                item = getattr(module, name)
                if isclass(item) and issubclass(item, Layer):
                    custom_objects[name] = item
        return custom_objects

    def get_model_in_blocks(self, model_function, include_top=True):
        # Load the vanilla model
        model = model_function(pretrained=True)

        # Instantiate consecutive blocks
        blocks = []

        # Add preprocess_input
        blocks.append(
            Lambda(
                lambda x: preprocess_input(x, mode=self.preprocess_input_mode),
                name="preprocess_input",
            )
        )

        # Discard the last pooling layer
        blocks += model.features.children[:-1]

        if include_top:
            # Add a GlobalAveragePooling2D layer
            blocks.append(GlobalAveragePooling2D())

            # Add the dense layer with softmax
            blocks += [model.output1, Softmax()]

        return blocks

    @staticmethod
    def wrap_block(block, x):
        input_tensor = Input(shape=K.int_shape(x)[1:])
        output_tensor = block.call(input_tensor)
        model = Model(inputs=input_tensor, outputs=output_tensor, name=block.name)
        return model


def sanity_check(
    image_url="https://upload.wikimedia.org/wikipedia/commons/3/37/African_Bush_Elephant.jpg",
    input_shape=(224, 224, 3),
):
    # Load image
    response = urllib.request.urlopen(image_url)
    image_content = np.asarray(bytearray(response.read()), dtype="uint8")
    image_content = cv2.imdecode(image_content, cv2.IMREAD_COLOR)
    image_content = cv2.resize(image_content, input_shape[:2][::-1])
    image_content = cv2.cvtColor(image_content, cv2.COLOR_BGR2RGB)
    image_content_array = np.expand_dims(image_content, axis=0)

    root_folder_path = "/tmp"
    applications_instance = Applications()
    model_name_to_model_function = (
        applications_instance.get_model_name_to_model_function()
    )
    custom_objects = applications_instance.get_custom_objects()
    for model_name, model_function in model_name_to_model_function.items():
        try:
            # Get model in blocks
            blocks = applications_instance.get_model_in_blocks(
                model_function=model_function, include_top=True
            )

            # Define the model
            input_tensor = Input(shape=input_shape)
            output_tensor = input_tensor
            for block in blocks:
                block = Applications.wrap_block(block, output_tensor)
                output_tensor = block(output_tensor)
            model = Model(inputs=input_tensor, outputs=output_tensor, name=model_name)

            # Miscellaneous tests
            model_file_path = os.path.join(root_folder_path, "{}.h5".format(model_name))
            model.save_weights(model_file_path)
            model.load_weights(model_file_path)
            model.save(model_file_path)
            model = load_model(model_file_path, custom_objects=custom_objects)
            model.summary()

            # Generate predictions
            predictions = model.predict_on_batch(image_content_array)
            print(model_name, decode_predictions(predictions, top=3)[0])
        except Exception as exception:  # pylint: disable=broad-except
            print(exception)

    print("All done!")


if __name__ == "__main__":
    sanity_check()
