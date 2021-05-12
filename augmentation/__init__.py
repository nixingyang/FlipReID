from urllib.request import urlopen

import cv2
import numpy as np
from albumentations import Compose, HorizontalFlip, PadIfNeeded, RandomCrop

from augmentation.transforms import RandomErasing, RandomGrayscalePatchReplacement


class ImageAugmentor(object):

    def __init__(
        self,
        image_height=224,
        image_width=224,
        padding_length=0,
        padding_ratio=0.05,
        horizontal_flip_probability=0.5,
        patch_replacement_probability=0.5,
        random_erasing_probability=0.5,
    ):
        # Initiation
        transforms = []

        # Pad side of the image and crop a random part of it
        if padding_length > 0 or padding_ratio > 0:
            min_height = image_height + int(
                max(padding_length, image_height * padding_ratio)
            )
            min_width = image_width + int(
                max(padding_length, image_width * padding_ratio)
            )
            transforms.append(
                PadIfNeeded(
                    min_height=min_height,
                    min_width=min_width,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=(123.68, 116.779, 103.939),
                )
            )
            transforms.append(RandomCrop(height=image_height, width=image_width))

        # Add HorizontalFlip
        transforms.append(HorizontalFlip(p=horizontal_flip_probability))

        # Add RandomGrayscalePatchReplacement
        transforms.append(
            RandomGrayscalePatchReplacement(p=patch_replacement_probability)
        )

        # Add RandomErasing
        transforms.append(RandomErasing(p=random_erasing_probability))

        # Compose transforms
        self.transformer = Compose(transforms=transforms)

    def apply_augmentation(self, image_content_array):
        transformed_image_content_list = []
        for image_content in image_content_array:
            transformed_image_content = self.transformer(image=image_content)["image"]
            transformed_image_content_list.append(transformed_image_content)
        return np.array(transformed_image_content_list, dtype=np.uint8)


def example():
    print("Loading the image content ...")
    raw_data = urlopen(url="https://avatars3.githubusercontent.com/u/15064790").read()
    raw_data = np.frombuffer(raw_data, np.uint8)
    image_content = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
    image_content = cv2.cvtColor(image_content, cv2.COLOR_BGR2RGB)
    image_height, image_width = image_content.shape[:2]

    print("Initiating the image augmentor ...")
    image_augmentor = ImageAugmentor(image_height=image_height, image_width=image_width)

    print("Generating the batch ...")
    image_content_list = [image_content] * 8
    image_content_array = np.array(image_content_list)

    print("Applying data augmentation ...")
    image_content_array = image_augmentor.apply_augmentation(image_content_array)

    print("Visualization ...")
    for image_index, image_content in enumerate(image_content_array, start=1):
        image_content = cv2.cvtColor(image_content, cv2.COLOR_RGB2BGR)
        cv2.imshow("image {}".format(image_index), image_content)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("All done!")


if __name__ == "__main__":
    example()
