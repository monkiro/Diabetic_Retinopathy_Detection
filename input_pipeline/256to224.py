import os
import cv2


def resize_images(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all files in the input folder
    file_list = os.listdir(input_folder)

    for file_name in file_list:
        # Construct the full path of the input image
        input_path = os.path.join(input_folder, file_name)

        # Read the image
        image = cv2.imread(input_path)

        # Check if the image is valid
        if image is not None:
            # Resize the image to (224, 224)
            resized_image = cv2.resize(image, (224, 224))

            # Construct the full path of the output image
            output_path = os.path.join(output_folder, file_name)

            # Save the resized image
            cv2.imwrite(output_path, resized_image)

if __name__ == "__main__":
    # Replace 'input_folder' and 'output_folder' with the actual paths
    input_folder = 'D:\\backups\\dataset_processed1\\images\\train'
    output_folder = 'D:\\backups\\dataset_processed1\\images\\train1'

    resize_images(input_folder, output_folder)
