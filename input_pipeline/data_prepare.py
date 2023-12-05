import csv
import os
import shutil
import cv2
import gin
import numpy as np
import pandas as pd
from image_pre import augment, Ben_preprocess_circle


def setDir(filepath):
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        # to delete the existed files
        for filename in os.listdir(filepath):
            file_path = os.path.join(filepath, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


def get_image_names_labels(path):
    labels = pd.read_csv(path)
    labels = np.asarray(labels)
    return labels


# Using this function to write binary labels and get oversampled, preprocessed and augmented image data
@gin.configurable
def processing_augmentation_oversampling(lb_path, save_path, img_path, amount):
    # train=True
    # if test dataset; train==0 without augmentation # amount :the wanted number of images per class
    # multiplier of the number of pictures in training set

    setDir(os.path.join(save_path, 'images'))
    setDir(os.path.join(save_path, 'images', 'train'))
    setDir(os.path.join(save_path, 'images', 'test'))

    title = ['name', 'label']

    with open(os.path.join(save_path, 'test.csv'), 'a', newline='', encoding='UTF-8') as f2:
        writer = csv.writer(f2)
        writer.writerow(title)

    label_imagename = get_image_names_labels(os.path.join(lb_path, 'test.csv'))

    for i in range(len(label_imagename)):
        file_path = os.path.join(save_path, 'images', 'test', label_imagename[i, 0] + ".jpg")
        image = cv2.imread(os.path.join(img_path, 'test', label_imagename[i, 0] + '.jpg'))

        if label_imagename[i, 1] <= 1:
            add = [label_imagename[i, 0], 0]
            with open(os.path.join(save_path, 'test.csv'), 'a', newline='', encoding='UTF-8') as f2:
                writer = csv.writer(f2)
                writer.writerow(add)

            image1 = Ben_preprocess_circle(image)
            result = cv2.imwrite(file_path, image1 * 255)
            print('saving image', label_imagename[i, 0],result)
        else:
            add = [label_imagename[i, 0], 1]
            with open(os.path.join(save_path, 'test.csv'), 'a', newline='', encoding='UTF-8') as f2:
                writer = csv.writer(f2)
                writer.writerow(add)

            image1 = Ben_preprocess_circle(image)
            result = cv2.imwrite(file_path, image1 * 255)
            print('saving image', label_imagename[i, 0], result)


    with open(os.path.join(save_path, 'train.csv'), 'a', newline='', encoding='UTF-8') as f3:
        writer = csv.writer(f3)
        writer.writerow(title)

#now start generate train dataset
    label_imagename = get_image_names_labels(os.path.join(lb_path, 'train.csv'))

    k = 1
    count0 = 0
    count1 = 0
    i = 0
    print('...........................')
    while (count1 < amount) or (count0 < amount):
        if (label_imagename[i, 1] <= 1) & (count0 < amount):
            image = cv2.imread(os.path.join(img_path, 'train', label_imagename[i, 0] + '.jpg'))
            if k > 413:
                image = augment(image)  # augumentation
            image = Ben_preprocess_circle(image)
            image = np.asarray(image)
            a = str(k).zfill(3)
            cv2.imwrite(os.path.join(save_path, 'images', 'train', 'IDRiD_' + a + '.jpg'), image * 255)
            print('saving image', "IDRiD_", a)
            add = ['IDRiD_' + a, 0]
            with open(save_path + 'train.csv', 'a', newline='', encoding='UTF-8') as f3:
                writer = csv.writer(f3)
                writer.writerow(add)
            count0 += 1
            i += 1
            k += 1
            if i == 413:
                i = 0
        elif (label_imagename[i, 1] > 1) & (count1 < amount):
            image = cv2.imread(os.path.join(img_path, 'train', label_imagename[i, 0] + '.jpg'))
            if k > 413:
                image = augment(image)  # augumentation
            image = Ben_preprocess_circle(image)
            image = np.asarray(image)
            a = str(k).zfill(3)
            cv2.imwrite(os.path.join(save_path, 'images', 'train', 'IDRiD_' + a + '.jpg'), image * 255)
            print('saving image', "IDRiD_", a)
            add = ['IDRiD_' + a, 1]
            with open(save_path + 'train.csv', 'a', newline='', encoding='UTF-8') as f3:
                writer = csv.writer(f3)
                writer.writerow(add)
            count1 += 1
            i += 1
            k += 1
            if i == 413:
                i = 0
        else:
            i += 1
            if i == 413:
                i = 0



# if __name__ == '__main__':
#
#     gin.parse_config_file('D:\\DL_Lab_P1\\config.gin')
#
#     processing_augmentation_oversampling()