import math
from data_generation.generate_inhibits import *
import os
from label_file import LabelFile
from keras.preprocessing.image import ImageDataGenerator
import shutil


def translate(image, x, y):
  M = np.float32([[1, 0, x], [0, 1, y]])
  shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
  cv2.affine
  return shifted


def rotate(image, angle, center=None, scale=1.0):
  (h, w) = image.shape[:2]

  if center is None:
    center = (w / 2, h / 2)

  M = cv2.getRotationMatrix2D(center, angle, scale)
  rotated = cv2.warpAffine(image, M, (w, h))

  return rotated, center


def getPointAffinedPos(src_points, center, angle):
  dx = src_points[0] - center[0]
  dy = src_points[1] - center[1]
  func = lambda x: int(x + 0.5) if x > 0 else int(x - 0.5)
  tempx = dx * math.cos(math.pi / 2) + dy * math.sin(math.pi / 2) + center[0]
  tempy = -dx * math.sin(math.pi / 2) + dy * math.cos(math.pi / 2) + center[1]
  dst_x = func(tempx)
  dst_y = func(tempy)
  return [dst_x, dst_y]


def gettrans_Point(point, rotate_img, tr):
  (h, w) = rotate_img.shape[:2]
  # center = (w / 2, h / 2)
  dx = point[0]
  dy = point[1]
  if tr == 'flip_horizontal':
    dst_x = w - dx
    dst_y = dy
  if tr == 'flip_vertical':
    dst_x = dx
    dst_y = h - dy

  return [int(dst_x), int(dst_y)]



def data_augment(img_file, txt_file, rotate_folder):
  img = cv2.imread(img_file)
  rotate_label_data = LabelFile(txt_file)
  category_list, coords_list, file_name = rotate_label_data.load_json_label(txt_file)
  generator = ImageDataGenerator()
  transform_list = ['flip_horizontal', 'flip_vertical']
  filename = os.path.split(img_file)[-1]
  suffixlen = len(img_file.split('.')[-1]) + 1
  ### copy original img files to rotate_folder
  shutil.copy(img_file, os.path.join(rotate_folder, filename))
  shutil.copy(txt_file,
              os.path.join(rotate_folder, filename[:-suffixlen] + ".json"))
  # cp txt_file, totate_folder
  for brightness_loop in range(2):
    if brightness_loop == 0:
      brightness = 1
    else:
      brightness = np.random.uniform(low=0.6, high=1, size=1)[
        0]  # only pick one range from 0.6 to 1

    for transform_type in transform_list:

      rotate_img_file = transform_type + "_bright" + "%.2f" % brightness + filename  # single file

      rotate_txt_file = rotate_img_file[:-suffixlen] + ".json"
      rotate_img = generator.apply_transform(img, {transform_type: True,
                                                   'brightness': brightness})
      cv2.imwrite(os.path.join(rotate_folder, rotate_img_file), rotate_img)

      rotate_coord_list = []
      for coord in coords_list:
        rotate_point = []
        for i in range(4):
          rotate_point.append(
            gettrans_Point(coord[i], rotate_img, transform_type))

        rotate_coord_list.append(rotate_point)

      (imageHeight, imageWidth) = rotate_img.shape[:2]
      save_rotate_json_label(os.path.join(rotate_folder, rotate_txt_file),
                             rotate_img_file, imageHeight, imageWidth,
                             rotate_label_data, rotate_coord_list)


def save_rotate_json_label(rotate_path, rotate_img_file, imageHeight,
                           imageWidth, rotate_label_data, rotate_coord_list):
  for i in range(len(rotate_coord_list)):
    rotate_label_data.shapes[i]['points'] = rotate_coord_list[i]
    rotate_label_data.shapes[i]['shape_type'] = 'polygon'

  # shapes=[]
  # for shape in rotate_label_data.shapes:
  #    shapes.append(shape)

  rotate_label_data.save(rotate_path, rotate_label_data.shapes, rotate_img_file,
                         imageHeight, imageWidth)
  # rotate_label_data.save(rotate_txt_file,rotate_label_data.shapes,rotate_img_file,imageHeight,imageWidth)


if __name__ == "__main__":
  ####test single picture######
  # img_fold = r'../Duolin/done/'
  # image = r'_pmc_articles_instance_3845601_bin_cjc-32-07-380-g002.jpg'
  # img_file = img_fold+image
  # txt_file = r'../Duolin/done/_pmc_articles_instance_3845601_bin_cjc-32-07-380-g002.json'
  # rotate_folder = r'../Duolin/test_image_augment/'
  # sim_folder = r'../Duolin/test_image_inhibit/'
  # data_augment(img_file,txt_file,rotate_folder)
  # simulate_inhibit_perimg(img_fold,image,sim_folder)

  img_folder = r'../Duolin/done/'
  rotate_folder = r'../Duolin/test_image_augment/'
  sim_folder = r'../Duolin/test_image_inhibit/'
  # sim_thres = 0.6
  # generate all the augment files first
  for image in os.listdir(img_folder):
    if ".json" in image:
      continue

    img_file = os.path.join(img_folder, image)
    suffixlen = len(image.split('.')[-1]) + 1
    txt_file = os.path.join(img_folder, image[:-suffixlen] + '.json')
    try:
      data_augment(img_file, txt_file, rotate_folder)
    except:
      if not os.path.exists(txt_file):
        print('%s there is no json file' % txt_file)
      else:
        print('%s something wrong with file' % img_file)

  # simulate inhibits on all the augment files
  for image in os.listdir(rotate_folder):
    if ".json" in image:
      continue

    simulate_inhibit_perimg(rotate_folder, image, sim_folder)


