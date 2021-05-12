import numpy as np
from tensorflow.keras.layers import BatchNormalization, Dense
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.layers.convolutional import Conv


def replicate_model(model, suffix):
    # https://github.com/tensorflow/tensorflow/issues/37541
    vanilla_weights = model.get_weights()
    model = clone_model(model)
    model = Model(
        inputs=model.input,
        outputs=model.output,
        name="{}_{}".format(model.name, suffix),
    )
    model.set_weights(vanilla_weights)
    # https://github.com/tensorflow/tensorflow/issues/46871#issuecomment-789623719
    for weight in model.weights:
        weight._handle_name = "{}/{}".format(
            suffix, weight.name
        )  # pylint: disable=protected-access
    return model


def gather_items(model):
    # Get the model and all its children
    items = None
    # https://github.com/tensorflow/tensorflow/blob/v2.2.2/tensorflow/python/keras/engine/base_layer.py#L1120
    if hasattr(model, "_gather_unique_layers"):
        items = model._gather_unique_layers()  # pylint: disable=protected-access
    # https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/python/keras/engine/base_layer.py#L1409
    if hasattr(model, "_flatten_layers"):
        items = [*model._flatten_layers()]  # pylint: disable=protected-access
    assert items is not None

    # Separate models and layers
    models, layers = [], []
    for item in items:
        # A model should have submodules
        if len(item.submodules) > 0:
            models.append(item)
        else:
            layers.append(item)

    return models, layers


def specify_regularizers(
    model,
    kernel_regularization_factor=0,
    bias_regularization_factor=0,
    gamma_regularization_factor=0,
    beta_regularization_factor=0,
):

    def _assign_regularizer(
        layer, attribute_name, switch_name, regularization_factor, summary
    ):
        # regularization factor should be a positive number
        # layer should have the attribute
        if regularization_factor <= 0 or not hasattr(layer, attribute_name):
            return

        # Disable the additional check if switch_name is None
        # (only relevant to the kernel parameter in Conv and Dense layers)
        # Skip if the switch_name attribute is False
        # (the parameter of interest is not enabled)
        if switch_name is not None and not getattr(layer, switch_name):
            return

        # Set the regularizer in layer
        # (only relevant if one needs to save configurations of the model)
        regularizer = l2(l=regularization_factor)
        setattr(layer, attribute_name, regularizer)

        # Add the regularization loss term
        # https://github.com/tensorflow/tensorflow/blob/v2.2.2/tensorflow/python/keras/engine/base_layer.py#L578-L585
        variable = getattr(layer, attribute_name.split("_")[0])
        name_in_scope = variable.name[: variable.name.find(":")]
        layer._handle_weight_regularization(  # pylint: disable=protected-access
            name_in_scope, variable, regularizer
        )

        # Update summary
        key = "{}:{}".format(layer.__class__.__name__, attribute_name)
        if key not in summary:
            summary[key] = 0
        summary[key] += 1

    # Assign regularizers
    summary = {}
    losses_num = len(model.losses)
    print("Specifying regularizers for {} ...".format(model.name))
    _, layers = gather_items(model)
    for layer in layers:
        if isinstance(layer, (Conv, Dense)):
            _assign_regularizer(
                layer=layer,
                attribute_name="kernel_regularizer",
                switch_name=None,
                regularization_factor=kernel_regularization_factor,
                summary=summary,
            )
            _assign_regularizer(
                layer=layer,
                attribute_name="bias_regularizer",
                switch_name="use_bias",
                regularization_factor=bias_regularization_factor,
                summary=summary,
            )
        elif isinstance(layer, BatchNormalization):
            _assign_regularizer(
                layer=layer,
                attribute_name="gamma_regularizer",
                switch_name="scale",
                regularization_factor=gamma_regularization_factor,
                summary=summary,
            )
            _assign_regularizer(
                layer=layer,
                attribute_name="beta_regularizer",
                switch_name="center",
                regularization_factor=beta_regularization_factor,
                summary=summary,
            )
    print(summary)

    # Sanity check
    if summary:
        added_losses_num = np.sum(list(summary.values()))
        print("Added {} regularization terms.".format(added_losses_num))
        assert len(model.losses) == losses_num + added_losses_num


def specify_trainable(model, trainable, keywords=None):

    def _specify_trainable(item, trainable):
        if hasattr(item, "trainable") and item.trainable != trainable:
            print(
                "Change the trainable property of {} to {} ...".format(
                    item.name, trainable
                )
            )
            item.trainable = trainable

    match_function = (
        lambda item: np.sum([item.name.startswith(keyword) for keyword in keywords]) > 0
    )
    models, layers = gather_items(model)
    items = models + layers
    for item in items:
        if keywords is None or match_function(item):
            _specify_trainable(item, trainable)
