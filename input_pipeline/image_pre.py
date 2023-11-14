import gin
from random import randint
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def crop_image_from_gray(img, tol=7):
    # It aims to remove the black or dark background from an image by cropping it down to the main content
    # check the dimension of the image
    if img.ndim == 2:  # if the image is grayscale
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]  # return the image without the black background
    elif img.ndim == 3:  # if the image is RGB
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # convert the image to grayscale
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1, img2, img3], axis=-1)
        #         print(img.shape)
        return img

def Ben_preprocess(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  #  Converts the image from BGR (Blue, Green, Red) color space
    # to RGB color space.

    image = cv2.resize(image, (256, 256))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 10), -4, 128)
    # The function cv2.GaussianBlur() is used to blur the image. This function takes the following arguments:
    # src: It is the image whose Gaussian Blur has to be calculated.
    # GaussianBlur is help to remove the noise from the image and make it smooth.
    image = crop_image_from_gray(image)
    #  Calls the crop_image_from_gray function previously explained to crop out dark areas from the image,
    #  which is useful for focusing on the main content and removing irrelevant background.

    return image


def Ben_preprocess_circle(img):
    img = np.array(img)
    """
    Create circular crop around image centre
    """
    img = crop_image_from_gray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Calculate Radius for Circular Crop:
    height, width, depth = img.shape

    x = 1980  # int(width / 2)
    y = 1424  # int(height / 2)
    r = np.amin((x, y))  # Return the minimum of an array or minimum along an axis.

    circle_img = np.zeros((height, width), np.uint8)  # create a black image
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)  # draw a circle on the black image
    img = cv2.bitwise_and(img, img, mask=circle_img)  # keep only the part of the image inside the circle
    img = crop_image_from_gray(img)
    img = cv2.resize(img, (280, 280))
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), 10), -4, 128)  # 128
    img = img / 255
    shape = img.shape
    img = img.reshape([1, shape[0], shape[1], shape[2]])
    img = tf.image.crop_and_resize(
        img,
        [[12 / 280, 12 / 280, 268 / 280, 268 / 280]],
        box_indices=[0],
        crop_size=(256, 256),
    )  # crop the useless parts of the image  and add  black parts to make it a square then resize
    img = tf.squeeze(img)
    img = tf.keras.preprocessing.image.img_to_array(img)

    # plt.imshow(img)
    # plt.show()
    b = np.zeros(img.shape) # create a black image
    cv2.circle(b, (128, 128), int(64), (1, 1, 1), 128) # draw a circle on the black image
    img = img * b + 0.5 * (1 - b)  # keep only the part of the image inside the circle and add a gray background
    b, g, r = cv2.split(img)  # split the image into its three channels
    img = cv2.merge([r, g, b])  # merge the three channels to get the final image
    return img


"""Data augmentation"""

# Create a generator.
rng = tf.random.Generator.from_seed(123)


# Adjust the brightness of images by a random factor deterministically.
def random_brightness(image):
    img = tf.image.stateless_random_brightness(image, 0.1, seed=rng.make_seeds(2)[0])
    return img


# Adjust the contrast of images by a random factor deterministically.
def random_contrast(image):
    img = tf.image.stateless_random_contrast(image, 0.7, 0.9, seed=rng.make_seeds(2)[0])
    return img


# Adjust the hue of RGB images by a random factor deterministically.
def random_hue(image):
    img = tf.image.stateless_random_hue(image, 0.1, seed=rng.make_seeds(2)[0])
    return img


# Adjust the saturation of RGB images by a random factor deterministically.
def random_saturation(image):
    img = tf.image.stateless_random_saturation(image, 0.8, 1.0, seed=rng.make_seeds(2)[0])
    return img


# Randomly flip an image horizontally (left to right) deterministically.
def random_flip_left_right(image):
    img = tf.image.stateless_random_flip_left_right(image, seed=rng.make_seeds(2)[0])
    return img


# Randomly flip an image vertically (upside down) deterministically.
def random_flip_up_down(image):
    img = tf.image.stateless_random_flip_up_down(image, seed=rng.make_seeds(2)[0])
    return img


# Randomly rotate an image.
def random_rotate(image, center=None, scale=1.05):
    image = np.array(image)
    (h, w) = image.shape[:2]
    # if the center is None, initialize it as the center of the image
    if center is None:
        center = (w / 2, h / 2)  # perform the rotation
    M = cv2.getRotationMatrix2D(center, randint(1, 359), scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def random_augment(image, func):
    if tf.random.uniform([]) < 0.99:
        img = func(image)
    else:
        img = image
    return img


@gin.configurable
def augment(image):
    """Data augmentation"""
    image = random_augment(image, random_flip_left_right)
    image = random_augment(image, random_flip_up_down)
    image = random_augment(image, random_rotate)

    return image

@gin.configurable
def preprocess(image, label, img_height, img_width):
    """Dataset preprocessing: Normalizing and resizing"""

    # Normalize image: `uint8` -> `float32`.
    tf.cast(image, tf.float32) / 255.

    # Resize image
    image = tf.image.resize(image, size=(img_height, img_width))

    return image, label


