import os
import cv2



def crop_image(image_file_path, save_file_path, crop_width, crop_height):
    image = cv2.imread(image_file_path)
    cropImg = image[:crop_width,:crop_height]
    cv2.imwrite(save_file_path,cropImg)


if __name__ == '__main__':
    file_fold = r'C:\Users\LSC-110\Desktop\Images'
    save_fold = r'C:\Users\LSC-110\Desktop\Images\crop'
    crop_width = 736
    crop_height = 736
    if not os.path.exists(save_fold):
        os.mkdir(save_fold)
    for file_name in os.listdir(file_fold):
        if os.path.isfile(os.path.join(file_fold, file_name)):
            crop_image(os.path.join(file_fold, file_name),
                       os.path.join(save_fold, file_name),
                       crop_width,
                       crop_height)