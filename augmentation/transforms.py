import cv2
import numpy as np
from albumentations import ImageOnlyTransform


def apply_random_erasing(
    image_content,
    sl=0.02,
    sh=0.4,
    r1=0.3,
    mean=(123.68, 116.779, 103.939),
    max_attempt_num=100,
):
    """
    References:
    https://arxiv.org/abs/1708.04896
    https://github.com/zhunzhong07/Random-Erasing/blob/master/transforms.py
    """
    image_content = image_content.copy()
    image_height, image_width = image_content.shape[:-1]
    image_area = image_height * image_width
    for _ in range(max_attempt_num):
        target_area = np.random.uniform(sl, sh) * image_area
        aspect_ratio = np.random.uniform(r1, 1 / r1)
        erasing_height = int(np.round(np.sqrt(target_area * aspect_ratio)))
        erasing_width = int(np.round(np.sqrt(target_area / aspect_ratio)))
        if erasing_width < image_width and erasing_height < image_height:
            starting_height = np.random.randint(0, image_height - erasing_height)
            starting_width = np.random.randint(0, image_width - erasing_width)
            image_content[
                starting_height : starting_height + erasing_height,
                starting_width : starting_width + erasing_width,
            ] = mean
            break
    return image_content


def apply_random_grayscale_patch_replacement(
    image_content, sl=0.02, sh=0.4, r1=0.3, max_attempt_num=100
):
    """
    References:
    https://arxiv.org/abs/2101.08533
    https://github.com/finger-monkey/Data-Augmentation/blob/main/trans_gray.py
    """
    image_content = image_content.copy()
    image_height, image_width = image_content.shape[:-1]
    image_area = image_height * image_width
    for _ in range(max_attempt_num):
        target_area = np.random.uniform(sl, sh) * image_area
        aspect_ratio = np.random.uniform(r1, 1 / r1)
        erasing_height = int(np.round(np.sqrt(target_area * aspect_ratio)))
        erasing_width = int(np.round(np.sqrt(target_area / aspect_ratio)))
        if erasing_width < image_width and erasing_height < image_height:
            starting_height = np.random.randint(0, image_height - erasing_height)
            starting_width = np.random.randint(0, image_width - erasing_width)
            patch_in_RGB = image_content[
                starting_height : starting_height + erasing_height,
                starting_width : starting_width + erasing_width,
            ]
            patch_in_GRAY = cv2.cvtColor(patch_in_RGB, cv2.COLOR_RGB2GRAY)
            for index in range(3):
                image_content[
                    starting_height : starting_height + erasing_height,
                    starting_width : starting_width + erasing_width,
                    index,
                ] = patch_in_GRAY
            break
    return image_content


class RandomErasing(ImageOnlyTransform):  # pylint: disable=abstract-method

    def apply(self, img, **params):
        return apply_random_erasing(image_content=img)


class RandomGrayscalePatchReplacement(
    ImageOnlyTransform
):  # pylint: disable=abstract-method

    def apply(self, img, **params):
        return apply_random_grayscale_patch_replacement(image_content=img)
