import argparse
import numpy as np
from PIL import Image, ImageDraw
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import cfg,os
from label import point_inside_of_quad
from network import East



def softmax(x):
    e_to_x = np.exp(x)
    return e_to_x / np.sum(e_to_x, axis=0)

def sigmoid(x):
 #   """`y = 1 / (1 + exp(-x))`"""
    return 1 / (1 + np.exp(-x))


def cut_text_line(geo, scale_ratio_w, scale_ratio_h, im_array, img_path, s):
    geo /= [scale_ratio_w, scale_ratio_h]
    p_min = np.amin(geo, axis=0)
    p_max = np.amax(geo, axis=0)
    min_xy = p_min.astype(int)
    max_xy = p_max.astype(int) + 2
    sub_im_arr = im_array[min_xy[1]:max_xy[1], min_xy[0]:max_xy[0], :].copy()
    for m in range(min_xy[1], max_xy[1]):
        for n in range(min_xy[0], max_xy[0]):
            if not point_inside_of_quad(n, m, geo, p_min, p_max):
                sub_im_arr[m - min_xy[1], n - min_xy[0], :] = 255
    sub_im = image.array_to_img(sub_im_arr, scale=False)
    sub_im.save(img_path + '_subim%d.jpg' % s)


def resize_image(im, max_train_size):

    max_img_size = 32 * max ( im.width // 32 , im.height // 32 )
    if max_img_size > 2 * max_train_size:
        max_img_size = 2 * max_train_size
    im_width = np.minimum ( im.width , max_img_size )


    if im_width == max_img_size < im.width:
        im_height = int ( (im_width / im.width) * im.height )
    else:
        im_height = im.height
    o_height = np.minimum ( im_height , max_img_size )
    if o_height == max_img_size < im_height:
        o_width = int ( (o_height / im_height) * im_width )
    else:
        o_width = im_width
    d_wight = o_width - (o_width % 32)
    d_height = o_height - (o_height % 32)
    return d_wight , d_height



def predict(east_detect,
            img_path,
            text_pixel_threshold = cfg.text_pixel_threshold,
            text_side_threshold = cfg.text_side_vertex_pixel_threshold,
            text_trunc_threshold = cfg.text_trunc_threshold,
            action_pixel_threshold = cfg.action_pixel_threshold,
            action_side_vertex_pixel_threshold = cfg.action_side_vertex_pixel_threshold,
            arrow_trunc_threshold = cfg.arrow_trunc_threshold,
            nock_trunc_threshold = cfg.nock_trunc_threshold,
            quiet = False):

    img = image.load_img(img_path)
    d_wight, d_height = resize_image(img, cfg.max_predict_img_size)
    img = img.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
    img = image.img_to_array(img)
    img = preprocess_input(img, mode='tf')
    x = np.expand_dims(img, axis=0)
    y = east_detect.predict(x)

    y = np.squeeze(y, axis=0)
    y[:, :, :1] = sigmoid(y[:, :, :1])
    y[:, :, 1:4] = softmax(y[:, :, 1:4])

    #y[:, :, :5] = sigmoid(y[:, :, :5])
    txt_items = []
    with Image.open(img_path) as im:
        im_array = image.img_to_array(im.convert('RGB'))
        d_wight, d_heig y[:, :, 4:6] = sigmoid(y[:, :, 4:6])ht = resize_image(im, cfg.max_predict_img_size)
        scale_ratio_w = d_wight / im.width
        scale_ratio_h = d_height / im.height
        im = im.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
        quad_im = im.copy()
        draw = ImageDraw.Draw(im)
        quad_draw = ImageDraw.Draw(quad_im)
        for idx in range(3, 0, -1):
            if idx == 1:
                cond_act = np.greater_equal(y[:, :, 0], text_pixel_threshold)
                cond_cls1 = y[:, :, 1] > y[:, :, 2]
                cond_cls2 = y[:, :, 1] > y[:, :, 3]
            elif idx == 2:
                cond_act = np.greater_equal(y[:, :, 0], action_pixel_threshold)
                cond_cls1 = y[:, :, 2] > y[:, :, 1]
                cond_cls2 = y[:, :, 2] > y[:, :, 3]
            elif idx == 3:
                cond_act = np.greater_equal(y[:, :, 0], action_pixel_threshold)
                cond_cls1 = y[:, :, 3] > y[:, :, 1]
                cond_cls2 = y[:, :, 3] > y[:, :, 2]

            activation_pixels = np.where(np.logical_and(cond_act,cond_cls1,cond_cls2))

            quad_scores, quad_after_nms = nms(y, activation_pixels,
                                              idx,
                                              text_side_threshold,
                                              text_trunc_threshold,
                                              action_side_vertex_pixel_threshold,
                                              nock_trunc_threshold,
                                              arrow_trunc_threshold)

            for i, j in zip(activation_pixels[0], activation_pixels[1]):
                px = (j + 0.5) * cfg.pixel_size
                py = (i + 0.5) * cfg.pixel_size
                line_width, line_color = 1, 'red'
                if idx == 1 and y[i, j, 4] >= text_side_threshold and  y[i, j, 5] < text_trunc_threshold:
                    line_width, line_color = 2, 'orange'
                elif idx == 1 and y[i, j, 4] >= text_side_threshold and  y[i, j, 5] >= 1 - text_trunc_threshold:
                    line_width, line_color = 2, 'blue'
                elif idx == 2 and y[i, j, 4] >= \
                    action_side_vertex_pixel_threshold and y[i, j, 5] >= 1 - \
                        nock_trunc_threshold:
                    line_width, line_color = 2, 'yellow'
                elif idx == 3 and y[i, j, 4] >= action_side_vertex_pixel_threshold and \
                    y[i, j, 5] >= arrow_trunc_threshold:
                    line_width, line_color = 2, 'purple'
                draw.line([(px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
                           (px + 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
                           (px + 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
                           (px - 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
                           (px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size)],
                          width=line_width, fill=line_color)

            for score, geo, s in zip(quad_scores, quad_after_nms,
                                     range(len(quad_scores))):
                # form a box for current object
                if np.amin(score) > 0:
                    if idx == 1:
                        convert_bounding_box(geo)
                        rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]

                        # text
                        quad_draw.line([tuple(geo[0]),
                                        tuple(geo[1]),
                                        tuple(geo[2]),
                                        tuple(geo[3]),
                                        tuple(geo[0])], width=3, fill='red')

                        # form bounding box

                        rescaled_geo_list = np.reshape(
                            rescaled_geo.astype(np.int32), (8,)).tolist()

                        # normalize rescaled_geo_list
                        # for list_idx in range(len(rescaled_geo_list)):
                        #     if rescaled_geo_list[list_idx] < 0:
                        #         rescaled_geo_list[list_idx] = 0
                        #     elif list_idx % 2 != 0 and rescaled_geo_list[
                        #         list_idx] > im.height:
                        #         rescaled_geo_list[list_idx] = im.width
                        #     elif list_idx % 2 == 0 and rescaled_geo_list[
                        #         list_idx] > im.width:
                        #         rescaled_geo_list[list_idx] = im.height

                        txt_item = ','.join(map(str, rescaled_geo_list))
                        txt_item = 'text' + '\t' + txt_item
                        txt_items.append(txt_item + '\n')
                        del txt_item
                    elif idx == 2:
                        # nock
                        # geo = convert_bounding_box(geo)
                        rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
                        quad_draw.line([tuple(geo[0]),
                                        tuple(geo[1]),
                                        tuple(geo[2]),
                                        tuple(geo[3]),
                                        tuple(geo[0])], width=3, fill='blue')
                        rescaled_geo_list = np.reshape(
                            rescaled_geo.astype(np.int32), (8,)).tolist()
                        txt_item = ','.join(map(str, rescaled_geo_list))
                        txt_item = 'nock' + '\t' + txt_item
                        txt_items.append(txt_item + '\n')
                        del txt_item
                    elif idx == 3:
                        # arrow
                        # geo = convert_bounding_box(geo)
                        rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]

                        quad_draw.line([tuple(geo[0]),
                                        tuple(geo[1]),
                                        tuple(geo[2]),
                                        tuple(geo[3]),
                                        tuple(geo[0])], width=3, fill='green')

                        # form bounding box

                        rescaled_geo_list = np.reshape(
                            rescaled_geo.astype(np.int32), (8,)).tolist()

                        # normalize rescaled_geo_list
                        # for list_idx in range(len(rescaled_geo_list)):
                        #     if rescaled_geo_list[list_idx] < 0:
                        #         rescaled_geo_list[list_idx] = 0
                        #     elif list_idx % 2 != 0 and rescaled_geo_list[
                        #         list_idx] > im.width:
                        #         rescaled_geo_list[list_idx] = im.width
                        #     elif list_idx % 2 == 0 and rescaled_geo_list[
                        #         list_idx] > im.height:
                        #         rescaled_geo_list[list_idx] = im.height

                        txt_item = ','.join(map(str, rescaled_geo_list))
                        txt_item = 'arrow' + '\t' + txt_item
                        txt_items.append(txt_item + '\n')
                        del txt_item
                elif not quiet:
                    print('quad invalid with vertex num less then 4.')
            del activation_pixels
        # im.save(img_path + '_act.jpg')
        # quad_im.save(img_path + '_predict.jpg')
    del im,quad_im,draw,quad_draw,img
    return txt_items, y[:, :, :1]

def predict_txt(east_detect, img_path, txt_path, pixel_threshold, quiet=False):
    img = image.load_img(img_path)
    d_wight, d_height = resize_image(img, cfg.max_predict_img_size)
    scale_ratio_w = d_wight / img.width
    scale_ratio_h = d_height / img.height
    img = img.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
    img = image.img_to_array(img)
    img = preprocess_input(img, mode='tf')
    x = np.expand_dims(img, axis=0)
    y = east_detect.predict(x)
    y = np.squeeze(y, axis=0)

    # activate the output layer
    y[:, :, :4] = softmax(y[:, :, :4])
    y[:, :, 4:6] = sigmoid(y[:, :, 4:6])

    cond = np.greater_equal(y[:, :, 0], pixel_threshold)
    activation_pixels = np.where(cond)
    quad_scores, quad_after_nms = nms(y, activation_pixels)

    txt_items = []
    for score, geo in zip(quad_scores, quad_after_nms):
        if np.amin(score) > 0:
            rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
            rescaled_geo_list = np.reshape(rescaled_geo, (8,)).tolist()
            txt_item = ','.join(map(str, rescaled_geo_list))
            txt_items.append(txt_item + '\n')
        elif not quiet:
            print('quad invalid with vertex num less then 4.')
    if cfg.predict_write2txt and len(txt_items) > 0:
        with open(txt_path, 'w') as f_txt:
            f_txt.writelines(txt_items)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p',
                        default='demo/012.png',
                        help='image path')
    parser.add_argument('--threshold', '-t',
                        default=cfg.pixel_threshold,
                        help='pixel activation threshold')
    return parser.parse_args()


def should_merge(region, i, j):
    neighbor = {(i, j - 1)}
    return not region.isdisjoint(neighbor)


def region_neighbor(region_set):
    region_pixels = np.array(list(region_set))
    j_min = np.amin(region_pixels, axis=0)[1] - 1
    j_max = np.amax(region_pixels, axis=0)[1] + 1
    i_m = np.amin(region_pixels, axis=0)[0] + 1
    region_pixels[:, 0] += 1
    neighbor = {(region_pixels[n, 0], region_pixels[n, 1]) for n in
                range(len(region_pixels))}
    neighbor.add((i_m, j_min))
    neighbor.add((i_m, j_max))
    return neighbor


def region_group(region_list):
    S = [i for i in range(len(region_list))]
    D = []
    while len(S) > 0:
        m = S.pop(0)
        if len(S) == 0:
            # S has only one element, put it to D
            D.append([m])
        else:
            D.append(rec_region_merge(region_list, m, S))
    return D


def rec_region_merge(region_list, m, S):
    rows = [m]
    tmp = []
    for n in S:
        if not region_neighbor(region_list[m]).isdisjoint(region_list[n]) or \
                not region_neighbor(region_list[n]).isdisjoint(region_list[m]):
            # 第m与n相交
            tmp.append(n)
    for d in tmp:
        S.remove(d)
    for e in tmp:
        rows.extend(rec_region_merge(region_list, e, S))
    return rows


def convert_bounding_box(geo):
    min_x = min(geo[0,0],geo[1,0],geo[2,0],geo[3,0])
    max_x = max(geo[0,0],geo[1,0],geo[2,0],geo[3,0])
    min_y = min(geo[0,1],geo[1,1],geo[2,1],geo[3,1])
    max_y = max(geo[0,1],geo[1,1],geo[2,1],geo[3,1])
    geo[0, 0] = min_x
    geo[0, 1] = min_y
    geo[1, 0] = min_x
    geo[1, 1] = max_y
    geo[2, 0] = max_x
    geo[2, 1] = max_y
    geo[3, 0] = max_x
    geo[3, 1] = min_y


def nms(predict,
        activation_pixels,cls_idx,
        text_side_vertex_pixel_threshold = cfg.text_side_vertex_pixel_threshold,
        text_trunc_threshold = cfg.text_trunc_threshold,
        action_side_vertex_pixel_threshold = cfg.action_side_vertex_pixel_threshold,
        nock_trunc_threshold = cfg.nock_trunc_threshold,
        arrow_trunc_threshold = cfg.arrow_trunc_threshold):

    region_list = []

    for i, j in zip(activation_pixels[0], activation_pixels[1]):
        merge = False
        for k in range(len(region_list)):
            if should_merge(region_list[k], i, j):
                region_list[k].add((i, j))
                merge = True
                # Fixme 重叠文本区域处理，存在和多个区域邻接的pixels，先都merge试试
                # break
        if not merge:
            region_list.append({(i, j)})


    D = region_group(region_list)
    quad_list = np.zeros((len(D), 4, 2))
    score_list = np.zeros((len(D), 4))
    for group, g_th in zip(D, range(len(D))):
        total_score = np.zeros((4, 2))
        for row in group:
            for ij in region_list[row]:
                score = predict[ij[0], ij[1], 4]
                # if score >= side_threshold:
                if cls_idx == 1 and score >= text_side_vertex_pixel_threshold:
                    ith_score = predict[ij[0], ij[1], 5:6]
                    if not (text_trunc_threshold <= ith_score < 1 - text_trunc_threshold):
                        ith = int(np.around(ith_score))
                        total_score[ith * 2:(ith + 1) * 2] += score
                        px = (ij[1] + 0.5) * cfg.pixel_size
                        py = (ij[0] + 0.5) * cfg.pixel_size
                        p_v = [px, py] + np.reshape(predict[ij[0], ij[1], 6:],
                                                    (2, 2))
                        quad_list[g_th, ith * 2:(ith + 1) * 2] += score * p_v
                    #
                elif (cls_idx == 2 and predict[ij[0], ij[1], 5:6] <
                      1 - nock_trunc_threshold and score >=
                      action_side_vertex_pixel_threshold) or (
                    cls_idx == 3 and predict[ij[0], ij[1], 5:6] >=
                        arrow_trunc_threshold and score >=  action_side_vertex_pixel_threshold):
                    # for cls=2 nock, label for trunc is 0
                    # for cls=3 arrow, label for trunc is 1
                    # regression for 4 vertexes
                    total_score[0: 2] += score
                    total_score[2: 4] += score

                    px = (ij[1] + 0.5) * cfg.pixel_size
                    py = (ij[0] + 0.5) * cfg.pixel_size
                    p_v = [px, py] + np.reshape(predict[ij[0], ij[1], 6:],
                                                (2, 2))
                    quad_list[g_th, 0] += score * p_v[0]

                    quad_list[g_th, 1] += [score * p_v[0, 0], score *
                                           p_v[1, 1]]

                    quad_list[g_th, 2] += score * p_v[1]

                    quad_list[g_th, 3] += [score * p_v[1, 0], score *
                                           p_v[0, 1]]

        score_list[g_th] = total_score[:, 0]
        quad_list[g_th] /= (total_score + cfg.epsilon)
    return score_list, quad_list


if __name__ == '__main__':
    # args = parse_args()
    # img_path = args.path
    # threshold = float(args.threshold)
    # print(img_path, threshold)
    # img_path = r'C:\Users\LSC-110\Desktop\test'
    img_path = r'C:\Users\LSC-110\Desktop\data\positive'
    east = East()
    east_detect = east.east_network()
    east_detect.load_weights(cfg.saved_model_weights_file_path)
    # load gene dictionary
    # 2019/3/10   cx
    # with open(cfg.dictionary_file,'r') as dict_fp:
    #     word_dictionary = dict_fp.readlines()

    # claim some variances for computing final assessment
    pos_correct_detect_total = 0
    num_pd_detect_total = 0
    num_gt_detect_total = 0
    pos_correct_gene_total = 0
    num_pd_gene_total = 0
    num_gt_gene_total = 0

    # from OCR import OCR
    # from correct_gene_names import postprocessing_OCR
    # from evaluation import calculate_detection_metrics,calculate_gene_match_metrics
    # from label_file import LabelFile

    for image_file in os.listdir(img_path):
        if os.path.splitext(image_file)[1] == '':
            continue
        predict_gene_box = predict(east_detect, os.path.join(img_path,
                                       image_file), quiet=False)
        with open(os.path.join(img_path, image_file[:-4] + '.txt'),
                  'w') as result_fp:
            result_fp.writelines(str(predict_gene_box))

    #     # do OCR, also help filter some
    #     OCR_result = OCR(img_path, image_file, predict_gene_box)
    #     with open(os.path.join(img_path, image_file[:-4] + "_OCR.txt"), 'r') as test_fp:
    #         OCR_results = test_fp.readlines()
    #     with open(r'D:\Study\Master\1st deep learning project\Dima data(2)\Dima data\dictionary.txt',
    #               'r') as dictionary_fp:
    #         word_dictionary = dictionary_fp.readlines()
    #
    #     # post-processing for gene names
    # #     gene_names = postprocessing_OCR(OCR_results, word_dictionary,
    # #                                      predict_gene_box)
    #     coord=[]
    #     for idx in range(len(OCR_results)):
    #         ocr_result, str_coord = str(OCR_results[idx]).strip().split('\t')
    #         OCR_results[idx] = ocr_result.upper()
    #         coord.append(str_coord+'\n')
    #     corrections = postprocessing_OCR(OCR_results, word_dictionary,
    #                                      predict_gene_box)
    #     file = open(os.path.join(img_path, image_file[:-4] + "_correct.txt"), 'w')
    #     for idx in range(len(corrections)):
    #         a = corrections[idx]
    #         b = coord[idx]
    #         s = ''.join(a).strip('\n') + '\t' + str(b)
    #         file.write(s)
    #         file.write('\n')
    #     file.close()
    #     # with open(os.path.join(img_path,image_file[:-4] + '_correction.txt'), 'w') as res_fp:
    #     #     res_fp.writelines(corrections)
    #    del corrections,predict_gene_box, OCR_results,word_dictionary
