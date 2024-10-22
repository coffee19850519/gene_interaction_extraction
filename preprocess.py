import numpy as np
from PIL import Image, ImageDraw
import os,label_file
import random
from tqdm import tqdm

import cfg
from label import shrink


def batch_reorder_vertexes(xy_list_array):
    reorder_xy_list_array = np.zeros_like(xy_list_array)
    for xy_list, i in zip(xy_list_array, range(len(xy_list_array))):
        reorder_xy_list_array[i] = reorder_vertexes(xy_list)
    return reorder_xy_list_array


def reorder_vertexes(xy_list):
    reorder_xy_list = np.zeros_like(xy_list)
    # determine the first point with the smallest x,
    # if two has same x, choose that with smallest y,
    ordered = np.argsort(xy_list, axis=0)
    xmin1_index = ordered[0, 0]
    xmin2_index = ordered[1, 0]
    if xy_list[xmin1_index, 0] == xy_list[xmin2_index, 0]:
        if xy_list[xmin1_index, 1] <= xy_list[xmin2_index, 1]:
            reorder_xy_list[0] = xy_list[xmin1_index]
            first_v = xmin1_index
        else:
            reorder_xy_list[0] = xy_list[xmin2_index]
            first_v = xmin2_index
    else:
        reorder_xy_list[0] = xy_list[xmin1_index]
        first_v = xmin1_index
    # connect the first point to others, the third point on the other side of
    # the line with the middle slope
    others = list(range(4))
    others.remove(first_v)
    k = np.zeros((len(others),))
    for index, i in zip(others, range(len(others))):
        k[i] = (xy_list[index, 1] - xy_list[first_v, 1]) \
                    / (xy_list[index, 0] - xy_list[first_v, 0] + cfg.epsilon)
    k_mid = np.argsort(k)[1]
    third_v = others[k_mid]
    reorder_xy_list[2] = xy_list[third_v]
    # determine the second point which on the bigger side of the middle line
    others.remove(third_v)
    b_mid = xy_list[first_v, 1] - k[k_mid] * xy_list[first_v, 0]
    second_v, fourth_v = 0, 0
    for index, i in zip(others, range(len(others))):
        # delta = y - (k * x + b)
        delta_y = xy_list[index, 1] - (k[k_mid] * xy_list[index, 0] + b_mid)
        if delta_y > 0:
            second_v = index
        else:
            fourth_v = index
    reorder_xy_list[1] = xy_list[second_v]
    reorder_xy_list[3] = xy_list[fourth_v]
    # compare slope of 13 and 24, determine the final order
    k13 = k[k_mid]
    k24 = (xy_list[second_v, 1] - xy_list[fourth_v, 1]) / (
                xy_list[second_v, 0] - xy_list[fourth_v, 0] + cfg.epsilon)
    if k13 < k24:
        tmp_x, tmp_y = reorder_xy_list[3, 0], reorder_xy_list[3, 1]
        for i in range(2, -1, -1):
            reorder_xy_list[i + 1] = reorder_xy_list[i]
        reorder_xy_list[0, 0], reorder_xy_list[0, 1] = tmp_x, tmp_y
    return reorder_xy_list


def resize_image(im, max_img_size=cfg.max_train_img_size):
    im_width = np.minimum(im.width, max_img_size)
    if im_width == max_img_size < im.width:
        im_height = int((im_width / im.width) * im.height)
    else:
        im_height = im.height
    o_height = np.minimum(im_height, max_img_size)
    if o_height == max_img_size < im_height:
        o_width = int((o_height / im_height) * im_width)
    else:
        o_width = im_width
    d_wight = o_width - (o_width % 32)
    d_height = o_height - (o_height % 32)
    return d_wight, d_height


def preprocess():
    # calculate number of text, arrow, nock
    num_text = 0
    num_arrow = 0
    num_nock = 0
    nock_samples = []

    data_dir = cfg.train_data_dir
    origin_image_dir = os.path.join(data_dir, cfg.origin_image_dir_name)
    origin_txt_dir = os.path.join(data_dir, cfg.origin_txt_dir_name)
    train_image_dir = os.path.join(data_dir, cfg.train_image_dir_name)
    train_label_dir = os.path.join(data_dir, cfg.train_label_dir_name)
    if not os.path.exists(train_image_dir):
        os.mkdir(train_image_dir)
    if not os.path.exists(train_label_dir):
        os.mkdir(train_label_dir)
    show_gt_image_dir = os.path.join(data_dir, cfg.show_gt_image_dir_name)
    if not os.path.exists(show_gt_image_dir):
        os.mkdir(show_gt_image_dir)
    show_act_image_dir = os.path.join(cfg.train_data_dir, cfg.show_act_image_dir_name)
    if not os.path.exists(show_act_image_dir):
        os.mkdir(show_act_image_dir)

    o_img_list = os.listdir(origin_image_dir)
    print('found %d origin images.' % len(o_img_list))
    train_val_set = []
    for o_img_fname, _ in zip(o_img_list, tqdm(range(len(o_img_list)))):
        with Image.open(os.path.join(origin_image_dir, o_img_fname)) as im:
            # d_wight, d_height = resize_image(im)
            d_wight, d_height = cfg.max_train_img_size, cfg.max_train_img_size
            scale_ratio_w = d_wight / im.width
            scale_ratio_h = d_height / im.height
            im = im.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
            show_gt_im = im.copy()
            # draw on the img
            draw = ImageDraw.Draw(show_gt_im)
            # if o_img_fname[:3] != 'cin':
            #     preffix, suffix = o_img_fname.split('_')
            #     suffix = str(suffix).replace('png', 'txt')
            #     with open(os.path.join(origin_txt_dir,
            #                            preffix + '_label_' + suffix), 'r') as f:
            #         # load corresponding label file
            #         anno_list = f.readlines()
            #
            #     xy_list_array = np.zeros((len(anno_list), 5, 2))
            #     for anno, i in zip(anno_list, range(len(anno_list))):
            #         category, anno_colums = anno.strip().split('\t')
            #         anno_colums = anno_colums.strip().split(',')
            #         anno_array = np.array(anno_colums)
            #
            #         xy_list = anno_array.astype(float).reshape((4, 2))
            #
            #         xy_list[:, 0] = xy_list[:, 0] * scale_ratio_w
            #         xy_list[:, 1] = xy_list[:, 1] * scale_ratio_h
            #         xy_list = reorder_vertexes(xy_list)
            #         # set the class label, needed one-hot encoding
            #         if category == 'text':
            #             num_text = num_text + 1
            #             xy_list_array[i] = np.r_[xy_list, np.array([[0, 0]])]
            #             # xy_list_array[i] = xy_list
            #             _, shrink_xy_list, _ = shrink(xy_list, cfg.shrink_ratio)
            #             shrink_1, _, long_edge = shrink(xy_list,
            #                                             cfg.shrink_side_ratio)
            #         elif category == 'nock':
            #             num_nock = num_nock + 1
            #             xy_list_array[i] = np.r_[xy_list, np.array([[0, 1]])]
            #             _, shrink_xy_list, _ = shrink(xy_list, 0)
            #             shrink_1, _, long_edge = shrink(xy_list,
            #                                             0.5)
            #             # nock_samples.append(current_file)
            #         elif category == 'arrow':
            #             num_arrow = num_arrow + 1
            #             xy_list_array[i] = np.r_[xy_list, np.array([[1, 0]])]
            #             _, shrink_xy_list, _ = shrink(xy_list, 0)
            #             shrink_1, _, long_edge = shrink(xy_list,
            #                                             0.5)
            #         # else:
            #         #     #category == 'predict'
            #         #     xy_list_array[i] = np.r_[xy_list, np.array([[1, 1]])]
            #         #     _, shrink_xy_list, _ = shrink(xy_list, cfg.shrink_ratio)
            #         #     shrink_1, _, long_edge = shrink(xy_list,
            #         #                                     cfg.shrink_side_ratio)
            #
            #         if cfg.DEBUG:
            #             draw.line([tuple(xy_list[0]), tuple(xy_list[1]),
            #                        tuple(xy_list[2]), tuple(xy_list[3]),
            #                        tuple(xy_list[0])
            #                        ],
            #                       width=2, fill='green')
            #             draw.line([tuple(shrink_xy_list[0]),
            #                        tuple(shrink_xy_list[1]),
            #                        tuple(shrink_xy_list[2]),
            #                        tuple(shrink_xy_list[3]),
            #                        tuple(shrink_xy_list[0])
            #                        ],
            #                       width=2, fill='blue')
            #             vs = [[[0, 0, 3, 3, 0], [1, 1, 2, 2, 1]],
            #                   [[0, 0, 1, 1, 0], [2, 2, 3, 3, 2]]]
            #             for q_th in range(2):
            #                 draw.line([tuple(xy_list[vs[long_edge][q_th][0]]),
            #                            tuple(shrink_1[vs[long_edge][q_th][1]]),
            #                            tuple(shrink_1[vs[long_edge][q_th][2]]),
            #                            tuple(xy_list[vs[long_edge][q_th][3]]),
            #                            tuple(xy_list[vs[long_edge][q_th][4]])],
            #                           width=3, fill='yellow')
            #     if cfg.DEBUG:
            #         im.save(os.path.join(train_image_dir, o_img_fname))
            #     np.save(os.path.join(
            #         train_label_dir,
            #         o_img_fname[:-4] + '.npy'),
            #         xy_list_array)
            #     if cfg.DEBUG:
            #         show_gt_im.save(
            #             os.path.join(show_gt_image_dir, o_img_fname))
            #     train_val_set.append('{},{},{}\n'.format(o_img_fname,
            #                                              d_wight,
            #                                              d_height))
            # else:
            # load corresponding json label file
            category_list, coords_list, current_file = load_json_label(
                os.path.join(
                    origin_txt_dir, o_img_fname[:-4] + '.json'))

            xy_list_array = np.zeros((len(category_list), 5, 2))

            for category, coord, i in zip(category_list, coords_list,
                                          range(len(category_list))):
                xy_list = coord.astype(float)
                xy_list[:, 0] = xy_list[:, 0] * scale_ratio_w
                xy_list[:, 1] = xy_list[:, 1] * scale_ratio_h
                xy_list = reorder_vertexes(xy_list)
                # set the class label, needed one-hot encoding
                if category == 'text':
                    num_text = num_text + 1
                    xy_list_array[i] = np.r_[xy_list, np.array([[0, 0]])]
                    # xy_list_array[i] = xy_list
                    _, shrink_xy_list, _ = shrink(xy_list, cfg.shrink_ratio)
                    shrink_1, _, long_edge = shrink(xy_list,
                                                    cfg.shrink_side_ratio)
                elif category == 'nock':
                    num_nock = num_nock + 1
                    xy_list_array[i] = np.r_[xy_list, np.array([[0, 1]])]
                    _, shrink_xy_list, _ = shrink(xy_list, 0)
                    shrink_1, _, long_edge = shrink(xy_list,
                                                    0.5)
                    # nock_samples.append(current_file)
                elif category == 'arrow':
                    num_arrow = num_arrow + 1
                    xy_list_array[i] = np.r_[xy_list, np.array([[1, 0]])]
                    _, shrink_xy_list, _ = shrink(xy_list, 0)
                    shrink_1, _, long_edge = shrink(xy_list,
                                                    0.5)
                # else:
                #     #category == 'predict'
                #     xy_list_array[i] = np.r_[xy_list, np.array([[1, 1]])]
                #     _, shrink_xy_list, _ = shrink(xy_list, cfg.shrink_ratio)
                #     shrink_1, _, long_edge = shrink(xy_list,
                #                                     cfg.shrink_side_ratio)

                if cfg.DEBUG:
                    draw.line([tuple(xy_list[0]), tuple(xy_list[1]),
                               tuple(xy_list[2]), tuple(xy_list[3]),
                               tuple(xy_list[0])
                               ],
                              width=2, fill='green')
                    draw.line([tuple(shrink_xy_list[0]),
                               tuple(shrink_xy_list[1]),
                               tuple(shrink_xy_list[2]),
                               tuple(shrink_xy_list[3]),
                               tuple(shrink_xy_list[0])
                               ],
                              width=2, fill='blue')
                    vs = [[[0, 0, 3, 3, 0], [1, 1, 2, 2, 1]],
                          [[0, 0, 1, 1, 0], [2, 2, 3, 3, 2]]]
                    for q_th in range(2):
                        draw.line([tuple(xy_list[vs[long_edge][q_th][0]]),
                                   tuple(shrink_1[vs[long_edge][q_th][1]]),
                                   tuple(shrink_1[vs[long_edge][q_th][2]]),
                                   tuple(xy_list[vs[long_edge][q_th][3]]),
                                   tuple(xy_list[vs[long_edge][q_th][4]])],
                                  width=3, fill='yellow')
            if cfg.DEBUG:
                im.save(os.path.join(train_image_dir, o_img_fname))
            np.save(os.path.join(
                train_label_dir,
                o_img_fname[:-4] + '.npy'),
                xy_list_array)
            if cfg.DEBUG:
                show_gt_im.save(
                    os.path.join(show_gt_image_dir, o_img_fname))
            train_val_set.append('{},{},{}\n'.format(o_img_fname,
                                                     d_wight,
                                                     d_height))
            # del category_list, coords_list

    train_img_list = os.listdir(train_image_dir)
    print('found %d train images.' % len(train_img_list))
    train_label_list = os.listdir(train_label_dir)
    print('found %d train labels.' % len(train_label_list))

    random.shuffle(train_val_set)
    val_count = int(cfg.validation_split_ratio * len(train_val_set))
    with open(os.path.join(data_dir, cfg.val_fname), 'w') as f_val:
        f_val.writelines(train_val_set[:val_count])
    with open(os.path.join(data_dir, cfg.train_fname), 'w') as f_train:
        f_train.writelines(train_val_set[val_count:])

    return num_text, num_nock, num_arrow, nock_samples

def load_json_label(json_path):
    label_data = label_file.LabelFile(json_path)
    category_list = []
    coords_list = []
    file_name = label_data.filename
    for shape in label_data.shapes:
        try:
            if shape['shape_type'] == 'rectangle':
                category = label_data.generate_category(shape)
                coord = generate_rect_points(shape)
                category_list.append(category)
                coords_list.append(coord)
                del category,coord
            elif shape['shape_type'] == 'polygon':
                category = label_data.generate_category(shape)
                # if np.array(shape['points']).size != 8:
                #     print('file:'+json_path+' label:' +str(shape['label']) )
                coords_list.append(np.array(shape['points']).reshape((4,2)))
                category_list.append(category)
                del category
            else:
                print('we have other shape type:'+ str(shape['shape_type']))
        except:
            print('%s includes invalid annotation'%json_path)
    del  label_data
    return category_list, coords_list, file_name

def generate_rect_points(shape):
    # organize its vertex points to quadrangle
    points = np.array(shape['points']).reshape((2, 2))
    pt0 = np.min(points[:,0])
    pt1 = np.min(points[:,1])
    pt4 = np.max(points[:,0])
    pt5 = np.max(points[:,1])
    pt2 = pt4
    pt3 = pt1
    pt6 = pt0
    pt7 = pt5
    del  points
    #pts = np.zeros(4, 2)
    return np.array([[pt0, pt1],[pt2, pt3],[pt4, pt5],[pt6, pt7]]).reshape((4,2))




if __name__ == '__main__':
    #num_text = 0
    #num_arrow = 0

    num_text, num_nock, num_arrow,nock_files = preprocess()
    print('text:%d, nock:%d, arrow:%d'%(num_text,num_nock,num_arrow))
    for file in nock_files:
        print('figure including nock:' + str(file))
    #load_json_label(r'C:\Users\LSC-110\Desktop\Dima\labeled\cin_00047.json')
