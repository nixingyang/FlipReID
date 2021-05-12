import os
import shutil
import sys
import time

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from absl import app, flags
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import GroupShuffleSplit
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Input,
    Lambda,
)
from tensorflow.keras.losses import categorical_crossentropy, mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence

from applications import Applications
from augmentation import ImageAugmentor
from callbacks import HistoryLogger
from datasets import load_accumulated_info_of_dataset
from evaluation.metrics import compute_CMC_mAP
from evaluation.post_processing.re_ranking_ranklist import re_ranking
from layers.pooling import GlobalGeMPooling2D, InspectGeMPoolingParameters
from metric_learning.triplet_hermans import batch_hard, cdist
from utils.model_utils import replicate_model, specify_regularizers, specify_trainable
from utils.vis_utils import summarize_model, visualize_model

flags.DEFINE_string(
    "root_folder_path",
    os.path.expanduser("~/Documents/Local Storage/Dataset"),
    "Folder path of the dataset.",
)
flags.DEFINE_string("dataset_name", "Market1501", "Name of the dataset.")
# ["Market1501", "DukeMTMC_reID", "MSMT17"]
flags.DEFINE_string("backbone_model_name", "resnet50", "Name of the backbone model.")
# ["resnet50", "ibn_resnet50", "resnesta50"]
flags.DEFINE_integer(
    "freeze_backbone_for_N_epochs",
    20,
    "Freeze layers in the backbone model for N epochs.",
)
flags.DEFINE_integer("image_width", 128, "Width of the images.")
flags.DEFINE_integer("image_height", 384, "Height of the images.")
flags.DEFINE_integer("region_num", 2, "Number of regions in the regional branch.")
flags.DEFINE_float(
    "kernel_regularization_factor", 0.0005, "Regularization factor of kernel."
)
flags.DEFINE_float(
    "bias_regularization_factor", 0.0005, "Regularization factor of bias."
)
flags.DEFINE_float(
    "gamma_regularization_factor", 0.0005, "Regularization factor of gamma."
)
flags.DEFINE_float(
    "beta_regularization_factor", 0.0005, "Regularization factor of beta."
)
flags.DEFINE_string("pooling_mode", "GeM", "Mode of the pooling layer.")
# ["Average", "Max", "GeM"]
flags.DEFINE_float("min_value", 0.0, "Minimum value of feature embeddings.")
flags.DEFINE_float("max_value", 1.0, "Maximum value of feature embeddings.")
flags.DEFINE_float(
    "testing_size", 1.0, "Proportion or absolute number of testing groups."
)
flags.DEFINE_integer(
    "evaluate_testing_every_N_epochs",
    10,
    "Evaluate the performance on testing samples every N epochs.",
)
flags.DEFINE_integer("identity_num_per_batch", 16, "Number of identities in one batch.")
flags.DEFINE_integer("image_num_per_identity", 4, "Number of images of one identity.")
flags.DEFINE_string(
    "learning_rate_mode", "default", "Mode of the learning rate scheduler."
)
# ["constant", "linear", "cosine", "warmup", "default"]
flags.DEFINE_float("learning_rate_start", 2e-4, "Starting learning rate.")
flags.DEFINE_float("learning_rate_end", 2e-4, "Ending learning rate.")
flags.DEFINE_float("learning_rate_base", 2e-4, "Base learning rate.")
flags.DEFINE_integer(
    "learning_rate_warmup_epochs", 10, "Number of epochs to warmup the learning rate."
)
flags.DEFINE_integer(
    "learning_rate_steady_epochs",
    30,
    "Number of epochs to keep the learning rate steady.",
)
flags.DEFINE_float(
    "learning_rate_drop_factor", 10, "Factor to decrease the learning rate."
)
flags.DEFINE_float(
    "learning_rate_lower_bound", 2e-6, "Lower bound of the learning rate."
)
flags.DEFINE_integer("steps_per_epoch", 200, "Number of steps per epoch.")
flags.DEFINE_integer("epoch_num", 1000000, "Number of epochs.")
flags.DEFINE_integer("workers", 5, "Number of processes to spin up for data generator.")
flags.DEFINE_float(
    "patch_replacement_probability", 0.8, "Probability of applying patch replacement."
)
flags.DEFINE_float(
    "random_erasing_probability", 0.8, "Probability of applying random erasing."
)
flags.DEFINE_bool(
    "use_data_augmentation_in_training", True, "Use data augmentation in training."
)
flags.DEFINE_bool(
    "use_horizontal_flipping_inside_model",
    False,
    "Use horizontal flipping inside model.",
)
flags.DEFINE_bool(
    "use_horizontal_flipping_in_evaluation",
    False,
    "Use horizontal flipping in evaluation.",
)
flags.DEFINE_bool("use_re_ranking", False, "Use the re-ranking method.")
flags.DEFINE_bool("evaluation_only", False, "Only perform evaluation.")
flags.DEFINE_bool(
    "save_data_to_disk",
    False,
    "Save image features, identity ID and camera ID to disk.",
)
flags.DEFINE_string(
    "pretrained_model_file_path", "", "File path of the pretrained model."
)
flags.DEFINE_string(
    "output_folder_path",
    os.path.abspath(os.path.join(__file__, "../output")),
    "Path to directory to output files.",
)
FLAGS = flags.FLAGS


def apply_groupshufflesplit(groups, test_size, random_state=0):
    groupshufflesplit_instance = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_indexes, test_indexes = next(
        groupshufflesplit_instance.split(np.arange(len(groups)), groups=groups)
    )
    return train_indexes, test_indexes


def init_model(
    backbone_model_name,
    freeze_backbone_for_N_epochs,
    input_shape,
    region_num,
    attribute_name_to_label_encoder_dict,
    kernel_regularization_factor,
    bias_regularization_factor,
    gamma_regularization_factor,
    beta_regularization_factor,
    pooling_mode,
    min_value,
    max_value,
    use_horizontal_flipping,
):

    def _add_pooling_module(input_tensor):
        # Add a global pooling layer
        output_tensor = input_tensor
        if len(K.int_shape(output_tensor)) == 4:
            if pooling_mode == "Average":
                output_tensor = GlobalAveragePooling2D()(output_tensor)
            elif pooling_mode == "Max":
                output_tensor = GlobalMaxPooling2D()(output_tensor)
            elif pooling_mode == "GeM":
                output_tensor = GlobalGeMPooling2D()(output_tensor)
            else:
                assert False, "{} is an invalid argument!".format(pooling_mode)

        # Add the clipping operation
        if min_value is not None and max_value is not None:
            output_tensor = Lambda(
                lambda x: K.clip(x, min_value=min_value, max_value=max_value)
            )(output_tensor)

        return output_tensor

    def _add_classification_module(input_tensor):
        # Add a batch normalization layer
        output_tensor = input_tensor
        output_tensor = BatchNormalization(epsilon=2e-5)(output_tensor)

        # Add a dense layer with softmax activation
        label_encoder = attribute_name_to_label_encoder_dict["identity_ID"]
        class_num = len(label_encoder.classes_)
        output_tensor = Dense(
            units=class_num,
            use_bias=False,
            kernel_initializer=RandomNormal(mean=0.0, stddev=0.001),
        )(output_tensor)
        output_tensor = Activation("softmax")(output_tensor)

        return output_tensor

    def _triplet_hermans_loss(y_true, y_pred, metric="euclidean", margin="soft"):
        # Create the loss in two steps:
        # 1. Compute all pairwise distances according to the specified metric.
        # 2. For each anchor along the first dimension, compute its loss.
        dists = cdist(y_pred, y_pred, metric=metric)
        loss = batch_hard(dists=dists, pids=tf.argmax(y_true, axis=-1), margin=margin)
        return loss

    # Initiation
    miscellaneous_output_tensor_list = []

    # Initiate the early blocks
    applications_instance = Applications()
    model_name_to_model_function = (
        applications_instance.get_model_name_to_model_function()
    )
    assert (
        backbone_model_name in model_name_to_model_function.keys()
    ), "Backbone {} is not supported.".format(backbone_model_name)
    model_function = model_name_to_model_function[backbone_model_name]
    blocks = applications_instance.get_model_in_blocks(
        model_function=model_function, include_top=False
    )
    vanilla_input_tensor = Input(shape=input_shape)
    intermediate_output_tensor = vanilla_input_tensor
    for block in blocks[:-1]:
        block = Applications.wrap_block(block, intermediate_output_tensor)
        intermediate_output_tensor = block(intermediate_output_tensor)

    # Initiate the last blocks
    last_block = Applications.wrap_block(blocks[-1], intermediate_output_tensor)
    last_block_for_global_branch_model = replicate_model(
        model=last_block, suffix="global_branch"
    )
    last_block_for_regional_branch_model = replicate_model(
        model=last_block, suffix="regional_branch"
    )

    # Add the global branch
    miscellaneous_output_tensor = _add_pooling_module(
        input_tensor=last_block_for_global_branch_model(intermediate_output_tensor)
    )
    miscellaneous_output_tensor_list.append(miscellaneous_output_tensor)

    # Add the regional branch
    if region_num > 0:
        # Process each region
        regional_branch_output_tensor = last_block_for_regional_branch_model(
            intermediate_output_tensor
        )
        total_height = K.int_shape(regional_branch_output_tensor)[1]
        region_size = total_height // region_num
        for region_index in np.arange(region_num):
            # Get a slice of feature maps
            start_index = region_index * region_size
            end_index = (region_index + 1) * region_size
            if region_index == region_num - 1:
                end_index = total_height
            sliced_regional_branch_output_tensor = Lambda(
                lambda x, start_index=start_index, end_index=end_index: x[
                    :, start_index:end_index
                ]
            )(regional_branch_output_tensor)

            # Downsampling
            sliced_regional_branch_output_tensor = Conv2D(
                filters=K.int_shape(sliced_regional_branch_output_tensor)[-1]
                // region_num,
                kernel_size=3,
                padding="same",
            )(sliced_regional_branch_output_tensor)
            sliced_regional_branch_output_tensor = Activation("relu")(
                sliced_regional_branch_output_tensor
            )

            # Add the regional branch
            miscellaneous_output_tensor = _add_pooling_module(
                input_tensor=sliced_regional_branch_output_tensor
            )
            miscellaneous_output_tensor_list.append(miscellaneous_output_tensor)

    # Define the model used in inference
    inference_model = Model(
        inputs=[vanilla_input_tensor],
        outputs=miscellaneous_output_tensor_list,
        name="inference_model",
    )
    specify_regularizers(
        inference_model,
        kernel_regularization_factor,
        bias_regularization_factor,
        gamma_regularization_factor,
        beta_regularization_factor,
    )

    # Define the model used in classification
    classification_input_tensor_list = [
        Input(shape=K.int_shape(item)[1:]) for item in miscellaneous_output_tensor_list
    ]
    classification_output_tensor_list = []
    for classification_input_tensor in classification_input_tensor_list:
        classification_output_tensor = _add_classification_module(
            input_tensor=classification_input_tensor
        )
        classification_output_tensor_list.append(classification_output_tensor)
    classification_model = Model(
        inputs=classification_input_tensor_list,
        outputs=classification_output_tensor_list,
        name="classification_model",
    )
    specify_regularizers(
        classification_model,
        kernel_regularization_factor,
        bias_regularization_factor,
        gamma_regularization_factor,
        beta_regularization_factor,
    )

    # Define the model used in training
    expand = lambda x: x if isinstance(x, list) else [x]
    vanilla_input_tensor = Input(shape=K.int_shape(inference_model.input)[1:])
    vanilla_feature_tensor_list = expand(inference_model(vanilla_input_tensor))
    if use_horizontal_flipping:
        flipped_input_tensor = tf.image.flip_left_right(vanilla_input_tensor)
        flipped_feature_tensor_list = expand(inference_model(flipped_input_tensor))
        merged_feature_tensor_list = [
            sum(item_tuple) / 2
            for item_tuple in zip(
                vanilla_feature_tensor_list, flipped_feature_tensor_list
            )
        ]
    else:
        merged_feature_tensor_list = vanilla_feature_tensor_list
    miscellaneous_output_tensor_list = merged_feature_tensor_list
    classification_output_tensor_list = expand(
        classification_model(merged_feature_tensor_list)
    )
    training_model = Model(
        inputs=[vanilla_input_tensor],
        outputs=miscellaneous_output_tensor_list + classification_output_tensor_list,
        name="training_model",
    )

    # Add the flipping loss
    if use_horizontal_flipping:
        flipping_loss_list = [
            K.mean(mean_squared_error(*item_tuple))
            for item_tuple in zip(
                vanilla_feature_tensor_list, flipped_feature_tensor_list
            )
        ]
        flipping_loss = sum(flipping_loss_list)
        training_model.add_metric(flipping_loss, name="flipping", aggregation="mean")
        training_model.add_loss(1.0 * flipping_loss)

    # Compile the model
    triplet_hermans_loss_function = lambda y_true, y_pred: 1.0 * _triplet_hermans_loss(
        y_true, y_pred
    )
    miscellaneous_loss_function_list = [triplet_hermans_loss_function] * len(
        miscellaneous_output_tensor_list
    )
    categorical_crossentropy_loss_function = (
        lambda y_true, y_pred: 1.0
        * categorical_crossentropy(
            y_true, y_pred, from_logits=False, label_smoothing=0.0
        )
    )
    classification_loss_function_list = [categorical_crossentropy_loss_function] * len(
        classification_output_tensor_list
    )
    training_model.compile_kwargs = {
        "optimizer": Adam(),
        "loss": miscellaneous_loss_function_list + classification_loss_function_list,
    }
    if freeze_backbone_for_N_epochs > 0:
        specify_trainable(
            model=training_model,
            trainable=False,
            keywords=[block.name for block in blocks],
        )
    training_model.compile(**training_model.compile_kwargs)

    # Print the summary of the training model
    summarize_model(training_model)

    return training_model, inference_model


def read_image_file(image_file_path, input_shape):
    # Read image file
    image_content = cv2.imread(image_file_path)

    # Resize the image
    image_content = cv2.resize(image_content, input_shape[:2][::-1])

    # Convert from BGR to RGB
    image_content = cv2.cvtColor(image_content, cv2.COLOR_BGR2RGB)

    return image_content


class TrainDataSequence(Sequence):

    def __init__(
        self,
        accumulated_info_dataframe,
        attribute_name_to_label_encoder_dict,
        input_shape,
        image_augmentor,
        use_data_augmentation,
        label_repetition_num,
        identity_num_per_batch,
        image_num_per_identity,
        steps_per_epoch,
    ):
        super(TrainDataSequence, self).__init__()

        # Save as variables
        (
            self.accumulated_info_dataframe,
            self.attribute_name_to_label_encoder_dict,
            self.input_shape,
        ) = (
            accumulated_info_dataframe,
            attribute_name_to_label_encoder_dict,
            input_shape,
        )
        self.image_augmentor, self.use_data_augmentation = (
            image_augmentor,
            use_data_augmentation,
        )
        self.label_repetition_num = label_repetition_num
        (
            self.identity_num_per_batch,
            self.image_num_per_identity,
            self.steps_per_epoch,
        ) = (identity_num_per_batch, image_num_per_identity, steps_per_epoch)

        # Unpack image_file_path and identity_ID
        self.image_file_path_array, self.identity_ID_array = (
            self.accumulated_info_dataframe[
                ["image_file_path", "identity_ID"]
            ].values.transpose()
        )
        self.image_file_path_to_record_index_dict = dict(
            [
                (image_file_path, record_index)
                for record_index, image_file_path in enumerate(
                    self.image_file_path_array
                )
            ]
        )
        self.batch_size = identity_num_per_batch * image_num_per_identity
        self.image_num_per_epoch = self.batch_size * steps_per_epoch

        # Initiation
        self.image_file_path_list_generator = self._get_image_file_path_list_generator()
        self.image_file_path_list = next(self.image_file_path_list_generator)

    def _get_image_file_path_list_generator(self):
        # Map identity ID to image file paths
        identity_ID_to_image_file_paths_dict = {}
        for image_file_path, identity_ID in zip(
            self.image_file_path_array, self.identity_ID_array
        ):
            if identity_ID not in identity_ID_to_image_file_paths_dict:
                identity_ID_to_image_file_paths_dict[identity_ID] = []
            identity_ID_to_image_file_paths_dict[identity_ID].append(image_file_path)

        image_file_path_list = []
        while True:
            # Split image file paths into multiple sections
            identity_ID_to_image_file_paths_in_sections_dict = {}
            for identity_ID in identity_ID_to_image_file_paths_dict:
                image_file_paths = np.array(
                    identity_ID_to_image_file_paths_dict[identity_ID]
                )
                if len(image_file_paths) < self.image_num_per_identity:
                    continue
                np.random.shuffle(image_file_paths)
                section_num = int(len(image_file_paths) / self.image_num_per_identity)
                image_file_paths = image_file_paths[
                    : section_num * self.image_num_per_identity
                ]
                image_file_paths_in_sections = np.split(image_file_paths, section_num)
                identity_ID_to_image_file_paths_in_sections_dict[identity_ID] = (
                    image_file_paths_in_sections
                )

            while (
                len(identity_ID_to_image_file_paths_in_sections_dict)
                >= self.identity_num_per_batch
            ):
                # Choose identity_num_per_batch identity_IDs
                identity_IDs = np.random.choice(
                    list(identity_ID_to_image_file_paths_in_sections_dict.keys()),
                    size=self.identity_num_per_batch,
                    replace=False,
                )
                for identity_ID in identity_IDs:
                    # Get one section
                    image_file_paths_in_sections = (
                        identity_ID_to_image_file_paths_in_sections_dict[identity_ID]
                    )
                    image_file_paths = image_file_paths_in_sections.pop(-1)
                    if len(image_file_paths_in_sections) == 0:
                        del identity_ID_to_image_file_paths_in_sections_dict[
                            identity_ID
                        ]

                    # Add the entries
                    image_file_path_list += image_file_paths.tolist()

                if len(image_file_path_list) == self.image_num_per_epoch:
                    yield image_file_path_list
                    image_file_path_list = []

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, index):
        label_encoder = self.attribute_name_to_label_encoder_dict["identity_ID"]
        image_content_list, one_hot_encoding_list = [], []
        image_file_path_list = self.image_file_path_list[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        for image_file_path in image_file_path_list:
            # Read image
            image_content = read_image_file(image_file_path, self.input_shape)
            image_content_list.append(image_content)

            # Get current record from accumulated_info_dataframe
            record_index = self.image_file_path_to_record_index_dict[image_file_path]
            accumulated_info = self.accumulated_info_dataframe.iloc[record_index]
            assert image_file_path == accumulated_info["image_file_path"]

            # Get the one hot encoding vector
            identity_ID = accumulated_info["identity_ID"]
            one_hot_encoding = np.zeros(len(label_encoder.classes_), dtype=np.float32)
            one_hot_encoding[label_encoder.transform([identity_ID])[0]] = 1
            one_hot_encoding_list.append(one_hot_encoding)

        # Construct image_content_array
        image_content_array = np.array(image_content_list)
        if self.use_data_augmentation:
            # Apply data augmentation
            image_content_array = self.image_augmentor.apply_augmentation(
                image_content_array
            )

        # Construct one_hot_encoding_array_list
        one_hot_encoding_array = np.array(one_hot_encoding_list)
        one_hot_encoding_array_list = [
            one_hot_encoding_array
        ] * self.label_repetition_num

        return image_content_array, one_hot_encoding_array_list

    def on_epoch_end(self):
        self.image_file_path_list = next(self.image_file_path_list_generator)


class TestDataSequence(Sequence):

    def __init__(self, accumulated_info_dataframe, input_shape, batch_size):
        super(TestDataSequence, self).__init__()

        # Save as variables
        self.accumulated_info_dataframe, self.input_shape = (
            accumulated_info_dataframe,
            input_shape,
        )

        # Unpack image_file_path and identity_ID
        self.image_file_path_array = self.accumulated_info_dataframe[
            "image_file_path"
        ].values
        self.batch_size = batch_size
        self.steps_per_epoch = int(
            np.ceil(len(self.image_file_path_array) / self.batch_size)
        )

        # Initiation
        self.image_file_path_list = self.image_file_path_array.tolist()
        self.use_horizontal_flipping = False

    def enable_horizontal_flipping(self):
        self.use_horizontal_flipping = True

    def disable_horizontal_flipping(self):
        self.use_horizontal_flipping = False

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, index):
        image_content_list = []
        image_file_path_list = self.image_file_path_list[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        for image_file_path in image_file_path_list:
            # Read image
            image_content = read_image_file(image_file_path, self.input_shape)
            if self.use_horizontal_flipping:
                image_content = cv2.flip(image_content, 1)
            image_content_list.append(image_content)

        # Construct image_content_array
        image_content_array = np.array(image_content_list)

        return image_content_array


class Evaluator(Callback):

    def __init__(
        self,
        inference_model,
        split_name,
        query_accumulated_info_dataframe,
        gallery_accumulated_info_dataframe,
        input_shape,
        use_horizontal_flipping,
        use_re_ranking,
        batch_size,
        workers,
        use_multiprocessing,
        rank_list=(1, 5, 10, 20),
        every_N_epochs=1,
        epoch_num=1,
        output_folder_path=None,
    ):
        super(Evaluator, self).__init__()
        if hasattr(self, "_supports_tf_logs"):
            self._supports_tf_logs = True

        self.callback_disabled = (
            query_accumulated_info_dataframe is None
            or gallery_accumulated_info_dataframe is None
        )
        if self.callback_disabled:
            return

        self.inference_model = inference_model
        self.split_name = split_name
        self.query_generator = TestDataSequence(
            query_accumulated_info_dataframe, input_shape, batch_size
        )
        self.gallery_generator = TestDataSequence(
            gallery_accumulated_info_dataframe, input_shape, batch_size
        )
        self.query_identity_ID_array, self.query_camera_ID_array = (
            query_accumulated_info_dataframe[
                ["identity_ID", "camera_ID"]
            ].values.transpose()
        )
        self.gallery_identity_ID_array, self.gallery_camera_ID_array = (
            gallery_accumulated_info_dataframe[
                ["identity_ID", "camera_ID"]
            ].values.transpose()
        )
        self.input_shape, self.use_horizontal_flipping = (
            input_shape,
            use_horizontal_flipping,
        )
        self.use_re_ranking, self.batch_size = use_re_ranking, batch_size
        self.workers, self.use_multiprocessing = workers, use_multiprocessing
        self.rank_list, self.every_N_epochs, self.epoch_num = (
            rank_list,
            every_N_epochs,
            epoch_num,
        )
        self.output_file_path = (
            None
            if output_folder_path is None
            else os.path.join(output_folder_path, "{}.npz".format(split_name))
        )

        self.metrics = ["cosine"]

    def extract_features(self, data_generator):
        apply_stacking = lambda x: np.hstack(x) if isinstance(x, list) else x
        data_generator.disable_horizontal_flipping()
        feature_array = apply_stacking(
            self.inference_model.predict(
                x=data_generator,
                workers=self.workers,
                use_multiprocessing=self.use_multiprocessing,
            )
        )
        if self.use_horizontal_flipping:
            data_generator.enable_horizontal_flipping()
            feature_array += apply_stacking(
                self.inference_model.predict(
                    x=data_generator,
                    workers=self.workers,
                    use_multiprocessing=self.use_multiprocessing,
                )
            )
            feature_array /= 2
        return feature_array

    def compute_distance_matrix(
        self, query_image_features, gallery_image_features, metric, use_re_ranking
    ):
        # Compute the distance matrix
        query_gallery_distance = pairwise_distances(
            query_image_features, gallery_image_features, metric=metric
        )
        distance_matrix = query_gallery_distance

        # Use the re-ranking method
        if use_re_ranking:
            query_query_distance = pairwise_distances(
                query_image_features, query_image_features, metric=metric
            )
            gallery_gallery_distance = pairwise_distances(
                gallery_image_features, gallery_image_features, metric=metric
            )
            distance_matrix = re_ranking(
                query_gallery_distance, query_query_distance, gallery_gallery_distance
            )

        return distance_matrix

    def on_epoch_end(self, epoch, logs=None):
        if self.callback_disabled:
            return
        if not (
            (epoch + 1) % self.every_N_epochs == 0 or (epoch + 1) == self.epoch_num
        ):
            return

        # Extract features
        feature_extraction_start = time.time()
        query_image_features_array = self.extract_features(self.query_generator)
        gallery_image_features_array = self.extract_features(self.gallery_generator)
        feature_extraction_end = time.time()
        feature_extraction_speed = (
            len(query_image_features_array) + len(gallery_image_features_array)
        ) / (feature_extraction_end - feature_extraction_start)
        print(
            "Speed of feature extraction: {:.2f} images per second.".format(
                feature_extraction_speed
            )
        )

        # Save image features, identity ID and camera ID to disk
        if self.output_file_path is not None:
            np.savez(
                self.output_file_path,
                query_image_features_array=query_image_features_array,
                gallery_image_features_array=gallery_image_features_array,
                query_identity_ID_array=self.query_identity_ID_array,
                gallery_identity_ID_array=self.gallery_identity_ID_array,
                query_camera_ID_array=self.query_camera_ID_array,
                gallery_camera_ID_array=self.gallery_camera_ID_array,
            )

        for metric in self.metrics:
            # Compute distance matrix
            distance_matrix = self.compute_distance_matrix(
                query_image_features_array,
                gallery_image_features_array,
                metric,
                self.use_re_ranking,
            )

            # Compute the CMC and mAP scores
            CMC_score_array, mAP_score = compute_CMC_mAP(
                distmat=distance_matrix,
                q_pids=self.query_identity_ID_array,
                g_pids=self.gallery_identity_ID_array,
                q_camids=self.query_camera_ID_array,
                g_camids=self.gallery_camera_ID_array,
            )

            # Append the CMC and mAP scores
            logs[
                "{}_{}_{}_rank_to_accuracy_dict".format(
                    self.split_name, metric, self.use_re_ranking
                )
            ] = dict(
                [
                    ("rank-{} accuracy".format(rank), CMC_score_array[rank - 1])
                    for rank in self.rank_list
                ]
            )
            logs[
                "{}_{}_{}_mAP_score".format(
                    self.split_name, metric, self.use_re_ranking
                )
            ] = mAP_score


def learning_rate_scheduler(
    epoch_index,
    epoch_num,
    learning_rate_mode,
    learning_rate_start,
    learning_rate_end,
    learning_rate_base,
    learning_rate_warmup_epochs,
    learning_rate_steady_epochs,
    learning_rate_drop_factor,
    learning_rate_lower_bound,
):
    learning_rate = None
    if learning_rate_mode == "constant":
        assert (
            learning_rate_start == learning_rate_end
        ), "starting and ending learning rates should be equal!"
        learning_rate = learning_rate_start
    elif learning_rate_mode == "linear":
        learning_rate = (learning_rate_end - learning_rate_start) / (
            epoch_num - 1
        ) * epoch_index + learning_rate_start
    elif learning_rate_mode == "cosine":
        assert (
            learning_rate_start > learning_rate_end
        ), "starting learning rate should be higher than ending learning rate!"
        learning_rate = (learning_rate_start - learning_rate_end) / 2 * np.cos(
            np.pi * epoch_index / (epoch_num - 1)
        ) + (learning_rate_start + learning_rate_end) / 2
    elif learning_rate_mode == "warmup":
        learning_rate = (learning_rate_end - learning_rate_start) / (
            learning_rate_warmup_epochs - 1
        ) * epoch_index + learning_rate_start
        learning_rate = np.min((learning_rate, learning_rate_end))
    elif learning_rate_mode == "default":
        if epoch_index < learning_rate_warmup_epochs:
            learning_rate = (learning_rate_base - learning_rate_lower_bound) / (
                learning_rate_warmup_epochs - 1
            ) * epoch_index + learning_rate_lower_bound
        else:
            if learning_rate_drop_factor == 0:
                learning_rate_drop_factor = np.exp(
                    learning_rate_steady_epochs
                    / (epoch_num - learning_rate_warmup_epochs * 2)
                    * np.log(learning_rate_base / learning_rate_lower_bound)
                )
            learning_rate = learning_rate_base / np.power(
                learning_rate_drop_factor,
                int(
                    (epoch_index - learning_rate_warmup_epochs)
                    / learning_rate_steady_epochs
                ),
            )
    else:
        assert False, "{} is an invalid argument!".format(learning_rate_mode)
    learning_rate = np.max((learning_rate, learning_rate_lower_bound))
    return learning_rate


def main(_):
    print("Getting hyperparameters ...")
    print("Using command {}".format(" ".join(sys.argv)))
    flag_values_dict = FLAGS.flag_values_dict()
    for flag_name in sorted(flag_values_dict.keys()):
        flag_value = flag_values_dict[flag_name]
        print(flag_name, flag_value)
    root_folder_path, dataset_name = FLAGS.root_folder_path, FLAGS.dataset_name
    backbone_model_name, freeze_backbone_for_N_epochs = (
        FLAGS.backbone_model_name,
        FLAGS.freeze_backbone_for_N_epochs,
    )
    image_height, image_width = FLAGS.image_height, FLAGS.image_width
    input_shape = (image_height, image_width, 3)
    region_num = FLAGS.region_num
    kernel_regularization_factor = FLAGS.kernel_regularization_factor
    bias_regularization_factor = FLAGS.bias_regularization_factor
    gamma_regularization_factor = FLAGS.gamma_regularization_factor
    beta_regularization_factor = FLAGS.beta_regularization_factor
    pooling_mode = FLAGS.pooling_mode
    min_value, max_value = FLAGS.min_value, FLAGS.max_value
    testing_size = FLAGS.testing_size
    testing_size = int(testing_size) if testing_size > 1 else testing_size
    use_testing = testing_size != 0
    evaluate_testing_every_N_epochs = FLAGS.evaluate_testing_every_N_epochs
    identity_num_per_batch, image_num_per_identity = (
        FLAGS.identity_num_per_batch,
        FLAGS.image_num_per_identity,
    )
    batch_size = identity_num_per_batch * image_num_per_identity
    learning_rate_mode, learning_rate_start, learning_rate_end = (
        FLAGS.learning_rate_mode,
        FLAGS.learning_rate_start,
        FLAGS.learning_rate_end,
    )
    learning_rate_base, learning_rate_warmup_epochs, learning_rate_steady_epochs = (
        FLAGS.learning_rate_base,
        FLAGS.learning_rate_warmup_epochs,
        FLAGS.learning_rate_steady_epochs,
    )
    learning_rate_drop_factor, learning_rate_lower_bound = (
        FLAGS.learning_rate_drop_factor,
        FLAGS.learning_rate_lower_bound,
    )
    steps_per_epoch = FLAGS.steps_per_epoch
    epoch_num = FLAGS.epoch_num
    workers = FLAGS.workers
    use_multiprocessing = workers > 1
    patch_replacement_probability, random_erasing_probability = (
        FLAGS.patch_replacement_probability,
        FLAGS.random_erasing_probability,
    )
    use_data_augmentation_in_training = FLAGS.use_data_augmentation_in_training
    use_horizontal_flipping_inside_model, use_horizontal_flipping_in_evaluation = (
        FLAGS.use_horizontal_flipping_inside_model,
        FLAGS.use_horizontal_flipping_in_evaluation,
    )
    use_re_ranking = FLAGS.use_re_ranking
    evaluation_only, save_data_to_disk = FLAGS.evaluation_only, FLAGS.save_data_to_disk
    pretrained_model_file_path = FLAGS.pretrained_model_file_path

    output_folder_path = os.path.abspath(
        os.path.join(
            FLAGS.output_folder_path, "{}_{}".format(dataset_name, backbone_model_name)
        )
    )
    shutil.rmtree(output_folder_path, ignore_errors=True)
    os.makedirs(output_folder_path)
    print("Recreating the output folder at {} ...".format(output_folder_path))

    print("Loading the annotations of the {} dataset ...".format(dataset_name))
    (
        train_and_valid_accumulated_info_dataframe,
        test_query_accumulated_info_dataframe,
        test_gallery_accumulated_info_dataframe,
        train_and_valid_attribute_name_to_label_encoder_dict,
    ) = load_accumulated_info_of_dataset(
        root_folder_path=root_folder_path, dataset_name=dataset_name
    )

    if use_testing:
        if testing_size != 1:
            print("Using a subset from the testing dataset ...")
            test_accumulated_info_dataframe = pd.concat(
                [
                    test_query_accumulated_info_dataframe,
                    test_gallery_accumulated_info_dataframe,
                ],
                ignore_index=True,
            )
            test_identity_ID_array = test_accumulated_info_dataframe[
                "identity_ID"
            ].values
            _, test_query_and_gallery_indexes = apply_groupshufflesplit(
                groups=test_identity_ID_array, test_size=testing_size
            )
            test_query_mask = test_query_and_gallery_indexes < len(
                test_query_accumulated_info_dataframe
            )
            test_gallery_mask = np.logical_not(test_query_mask)
            test_query_indexes, test_gallery_indexes = (
                test_query_and_gallery_indexes[test_query_mask],
                test_query_and_gallery_indexes[test_gallery_mask],
            )
            test_query_accumulated_info_dataframe = (
                test_accumulated_info_dataframe.iloc[test_query_indexes]
            )
            test_gallery_accumulated_info_dataframe = (
                test_accumulated_info_dataframe.iloc[test_gallery_indexes]
            )
    else:
        (
            test_query_accumulated_info_dataframe,
            test_gallery_accumulated_info_dataframe,
        ) = (None, None)

    print("Initiating the model ...")
    training_model, inference_model = init_model(
        backbone_model_name=backbone_model_name,
        freeze_backbone_for_N_epochs=freeze_backbone_for_N_epochs,
        input_shape=input_shape,
        region_num=region_num,
        attribute_name_to_label_encoder_dict=train_and_valid_attribute_name_to_label_encoder_dict,
        kernel_regularization_factor=kernel_regularization_factor,
        bias_regularization_factor=bias_regularization_factor,
        gamma_regularization_factor=gamma_regularization_factor,
        beta_regularization_factor=beta_regularization_factor,
        pooling_mode=pooling_mode,
        min_value=min_value,
        max_value=max_value,
        use_horizontal_flipping=use_horizontal_flipping_inside_model,
    )
    visualize_model(model=training_model, output_folder_path=output_folder_path)

    print("Initiating the image augmentor ...")
    image_augmentor = ImageAugmentor(
        image_height=image_height,
        image_width=image_width,
        patch_replacement_probability=patch_replacement_probability,
        random_erasing_probability=random_erasing_probability,
    )

    print("Perform training ...")
    train_generator = TrainDataSequence(
        accumulated_info_dataframe=train_and_valid_accumulated_info_dataframe,
        attribute_name_to_label_encoder_dict=train_and_valid_attribute_name_to_label_encoder_dict,
        input_shape=input_shape,
        image_augmentor=image_augmentor,
        use_data_augmentation=use_data_augmentation_in_training,
        label_repetition_num=len(training_model.outputs),
        identity_num_per_batch=identity_num_per_batch,
        image_num_per_identity=image_num_per_identity,
        steps_per_epoch=steps_per_epoch,
    )
    test_evaluator_callback = Evaluator(
        inference_model=inference_model,
        split_name="test",
        query_accumulated_info_dataframe=test_query_accumulated_info_dataframe,
        gallery_accumulated_info_dataframe=test_gallery_accumulated_info_dataframe,
        input_shape=input_shape,
        use_horizontal_flipping=use_horizontal_flipping_in_evaluation,
        use_re_ranking=use_re_ranking,
        batch_size=batch_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        every_N_epochs=evaluate_testing_every_N_epochs,
        epoch_num=epoch_num,
        output_folder_path=output_folder_path if save_data_to_disk else None,
    )
    inspect_gempooling_parameters_callback = InspectGeMPoolingParameters()
    optimal_model_file_path = os.path.join(output_folder_path, "training_model.h5")
    modelcheckpoint_monitor = "test_cosine_False_mAP_score"
    modelcheckpoint_callback = ModelCheckpoint(
        filepath=optimal_model_file_path,
        monitor=modelcheckpoint_monitor,
        mode="max",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )
    learningratescheduler_callback = LearningRateScheduler(
        schedule=lambda epoch_index: learning_rate_scheduler(
            epoch_index=epoch_index,
            epoch_num=epoch_num,
            learning_rate_mode=learning_rate_mode,
            learning_rate_start=learning_rate_start,
            learning_rate_end=learning_rate_end,
            learning_rate_base=learning_rate_base,
            learning_rate_warmup_epochs=learning_rate_warmup_epochs,
            learning_rate_steady_epochs=learning_rate_steady_epochs,
            learning_rate_drop_factor=learning_rate_drop_factor,
            learning_rate_lower_bound=learning_rate_lower_bound,
        ),
        verbose=1,
    )
    if len(pretrained_model_file_path) > 0:
        assert os.path.isfile(pretrained_model_file_path)
        print("Loading weights from {} ...".format(pretrained_model_file_path))
        # Load weights from the pretrained model
        training_model.load_weights(pretrained_model_file_path)
        # Save the inference model
        inference_model_file_path = os.path.abspath(
            os.path.join(pretrained_model_file_path, "..", "inference_model.h5")
        )
        if not os.path.isfile(inference_model_file_path):
            inference_model.save(inference_model_file_path)
    if evaluation_only:
        print("Freezing the whole model in the evaluation_only mode ...")
        specify_trainable(model=training_model, trainable=False)
        training_model.compile(**training_model.compile_kwargs)

        assert testing_size == 1, "Use all testing samples for evaluation!"
        historylogger_callback = HistoryLogger(
            output_folder_path=os.path.join(output_folder_path, "evaluation")
        )
        training_model.fit(
            x=train_generator,
            steps_per_epoch=1,
            callbacks=[
                inspect_gempooling_parameters_callback,
                test_evaluator_callback,
                historylogger_callback,
            ],
            epochs=1,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            verbose=2,
        )
    else:
        if freeze_backbone_for_N_epochs > 0:
            print(
                "Freeze layers in the backbone model for {} epochs.".format(
                    freeze_backbone_for_N_epochs
                )
            )
            historylogger_callback = HistoryLogger(
                output_folder_path=os.path.join(output_folder_path, "training_A")
            )
            training_model.fit(
                x=train_generator,
                steps_per_epoch=steps_per_epoch,
                callbacks=[
                    test_evaluator_callback,
                    learningratescheduler_callback,
                    historylogger_callback,
                ],
                epochs=freeze_backbone_for_N_epochs,
                workers=workers,
                use_multiprocessing=use_multiprocessing,
                verbose=2,
            )

            print("Unfreeze layers in the backbone model.")
            specify_trainable(model=training_model, trainable=True)
            training_model.compile(**training_model.compile_kwargs)

        print("Perform conventional training for {} epochs.".format(epoch_num))
        historylogger_callback = HistoryLogger(
            output_folder_path=os.path.join(output_folder_path, "training_B")
        )
        training_model.fit(
            x=train_generator,
            steps_per_epoch=steps_per_epoch,
            callbacks=[
                inspect_gempooling_parameters_callback,
                test_evaluator_callback,
                modelcheckpoint_callback,
                learningratescheduler_callback,
                historylogger_callback,
            ],
            epochs=epoch_num,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            verbose=2,
        )

        if not os.path.isfile(optimal_model_file_path):
            print("Saving model to {} ...".format(optimal_model_file_path))
            training_model.save(optimal_model_file_path)

    print("All done!")


if __name__ == "__main__":
    app.run(main)
