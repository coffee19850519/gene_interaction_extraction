import numpy as np
from PIL import Image, ImageDraw
import os,label_file
import random
from tqdm import tqdm
from shape_tool import generate_rect_points
import cfg,json
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
            # load corresponding json label file
            category_list, coords_list, sub_box_list, current_file = load_json_label(
                os.path.join(
                    origin_txt_dir, o_img_fname[:-4] + '.json'))

            #xy_list_array = np.zeros((len(category_list), 5, 2))
            xy_list_dict = {}

            for category, coord, sub_boxes, i in zip(category_list, coords_list,sub_box_list,
                                          range(len(category_list))):
                xy_list = coord.astype(float)
                xy_list[:, 0] = xy_list[:, 0] * scale_ratio_w
                xy_list[:, 1] = xy_list[:, 1] * scale_ratio_h
                xy_list = reorder_vertexes(xy_list)
                # set the class label, needed one-hot encoding
                if category == 'text':
                    num_text = num_text + 1
                    #xy_list_array[i] = np.r_[xy_list, np.array([[0, 0]])]
                    xy_list_dict.update({
                        'category': category,
                        'coords':coord,
                        'sub_boxes':sub_boxes
                    })

                    # xy_list_array[i] = xy_list
                    _, shrink_xy_list, _ = shrink(xy_list, cfg.shrink_ratio)
                    shrink_1, _, long_edge = shrink(xy_list,
                                                    cfg.shrink_side_ratio)
                elif category == 'nock':
                    num_nock = num_nock + 1
                    #xy_list_array[i] = np.r_[xy_list, np.array([[0, 1]])]
                    xy_list_dict.update({
                        'category': category,
                        'coords': coord,
                        'sub_boxes': sub_boxes
                    })
                    _, shrink_xy_list, _ = shrink(xy_list, 0)
                    shrink_1, _, long_edge = shrink(xy_list,
                                                    0.5)
                    # nock_samples.append(current_file)
                elif category == 'arrow':
                    num_arrow = num_arrow + 1
                    #xy_list_array[i] = np.r_[xy_list, np.array([[1, 0]])]
                    xy_list_dict.update({
                        'category': category,
                        'coords': coord,
                        'sub_boxes': sub_boxes
                    })
                    _, shrink_xy_list, _ = shrink(xy_list, 0)
                    shrink_1, _, long_edge = shrink(xy_list,
                                                    0.5)
                elif category == 'relationship':

                    xy_list_dict.update({
                        'category': category,
                        'coords': coord,
                        'sub_boxes': sub_boxes
                    })
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

            #save the label file
            with  open(os.path.join(
                train_label_dir, o_img_fname[:-4] + '_label.json'),'w') as json_fp:
                json.dump(xy_list_dict,json_fp)

            if cfg.DEBUG:
                show_gt_im.save(
                    os.path.join(show_gt_image_dir, o_img_fname))

            train_val_set.append('{},{},{}\n'.format(o_img_fname,
                                                     d_wight,
                                                     d_height))

            del category_list, coords_list,sub_box_list, xy_list_dict

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
    sub_box_list = []
    file_name = label_data.filename
    for shape in label_data.shapes:
        try:
            category = label_data.generate_category(shape)
            category_list.append(category)
            if shape['shape_type'] == 'rectangle':
                coord = generate_rect_points(shape)
                coords_list.append(coord)
                del coord
            elif shape['shape_type'] == 'polygon':
                coord = np.array(shape['points']).reshape((4,2))
                coords_list.append(coord)
                category_list.append(category)
            else:
                print('we have other shape type:' + str(shape['shape_type']))

            if category == 'relationship':
                sub_shapes = label_data.get_sub_shape_in_relationship(shape)
                if sub_shapes is not None:
                    sub_boxes = []
                    for sub_shape in sub_shapes:
                        sub_boxes.append(sub_shape['points'])
                    sub_box_list.append(sub_boxes)
                    del sub_shapes,sub_boxes
                else:
                    #error arise when grouping relationship's sub-boxes
                    raise Exception('illegal relationship box found in ' + file_name)
            else:
                sub_box_list.append(None)

        except:
            print('%s includes invalid annotation'%json_path)
    del  label_data
    return category_list, coords_list, sub_box_list, file_name






if __name__ == '__main__':
    #num_text = 0
    #num_arrow = 0

    # num_text, num_nock, num_arrow,nock_files = preprocess()
    # print('text:%d, nock:%d, arrow:%d'%(num_text,num_nock,num_arrow))
    # for file in nock_files:
    #     print('figure including nock:' + str(file))
    load_json_label(r'C:\Users\coffe\Desktop\Henrys Annotations\pdf_2_A_2_1289.json')
