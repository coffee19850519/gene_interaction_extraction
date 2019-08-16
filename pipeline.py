import os
import cfg
from loadexcldata import load_ground_truth_into_excl
from OCR import OCR, display
from correct_gene_names import map_result_to_dictionary
from predict import predict
from network import East
from to_labelme import to_labelme
import cv2
import numpy as np
from detect_arrow_processing import find_all_arrows_for_straight_line, \
  pair_gene, find_all_inhibits_for_straight_line, plot_connections
from label_file import LabelFile
from evaluation import calculate_all_metrics_by_json
from loadexcldata import load_dictionary_from_excl

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

if __name__ == '__main__':

    if not os.path.isdir(cfg.predict_folder):
        os.mkdir(cfg.predict_folder)

    east = East()
    east_detect = east.east_network()
    east_detect.load_weights(cfg.saved_model_weights_file_path)

    # load ground truth into gene dictionary
    if cfg.ground_truth_folder:
        load_ground_truth_into_excl(col_num=1)  # 1 for openpyxl, 0 for xlrd
    user_words = load_dictionary_from_excl(col_num=0)
    count = 0

    for image_file in os.listdir(cfg.image_folder):

        image_name, image_ext = os.path.splitext(image_file)
        if image_ext == ".json":
            continue

        sub_image_folder = os.path.join(cfg.predict_folder, image_name)

        if not os.path.exists(sub_image_folder):
            os.mkdir(sub_image_folder)

        # display(str(count) + ": \t" + str(image_file) + "\n", file=log_file)

        count = count + 1

        # step 1: predict images
        if os.path.splitext(image_file)[1] == '' or os.path.splitext(image_file)[1] == '.ini':
            continue

        image_path = os.path.join(cfg.image_folder, image_file)
        predict_box, _ = predict(east_detect, image_path, quiet=True)

        with open(os.path.join(cfg.predict_folder, image_name + '_predict.txt'), 'w') as result_fp:
            result_fp.writelines(predict_box)

        # step 2：DO OCR images
        current_img = cv2.imread(image_path)

        with open(os.path.join(cfg.predict_folder, image_name + '_predict.txt'), 'r') as result_fp:
            predict_box = result_fp.readlines()

        OCR_results, all_results, corrected_results, fuzz_ratios, coordinates = OCR(image_file, sub_image_folder,
                                                                                    predict_box, user_words)

        with open(os.path.join(cfg.predict_folder, image_name + "_correct.txt"), 'w') as res_fp:
            for idx in range(len(OCR_results)):
                # res_fp.write(str(idx) + '@' + str(corrections[idx]) + '@' + str(coordinates[idx])+'\n')
                res_fp.write(str(idx) + '@' + str(OCR_results[idx]) + '@' + str(coordinates[idx])+'\n')

                # draw the box and idx on image
                cv2.drawContours(current_img, [np.array(coordinates[idx], np.int32)],
                                 -1, (255, 0, 0), thickness=2)
                cv2.putText(current_img, str(idx), (coordinates[idx][3][0]-5, coordinates[idx][3][1]-5),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0))

        # step 3: transfer to Json label file
        to_labelme(image_file,
                   os.path.join(cfg.predict_folder, image_name + '_predict.txt'),
                   os.path.join(cfg.predict_folder, image_name + '_correct.txt'),
                   cfg.predict_folder)

        # # step 4: pair gene generate json file
        # label_file = os.path.join(cfg.predict_folder, image_name + '.json')
        #
        # try:
        #     label = LabelFile(label_file)
        # except:
        #     display("Exception: " + str(image_name), file=cfg.log_file)
        #     continue
        #
        # arrow_activator_list, activator_neighbor_list, arrow_receptor_list, receptor_neighbor_list, \
        #     text_shapes, arrows_with_overlap, arrows_without_overlap = find_all_arrows_for_straight_line(
        #         image_path, label)
        #
        # arrow_relationships = pair_gene(arrow_activator_list, activator_neighbor_list,
        #                                 arrow_receptor_list, receptor_neighbor_list,
        #                                 text_shapes, arrows_with_overlap, image_file)
        #
        # arrow_shapes = arrows_without_overlap + arrows_with_overlap
        #
        # # alternate the arrow label
        # for arrow_idx in range(len(arrow_shapes)):
        #
        #     cv2.drawContours(current_img, [np.array(arrow_shapes[arrow_idx]['points'], np.int32)], -1,
        #                      (0, 0, 255), thickness=2)
        #     cv2.putText(current_img, str(arrow_idx), (arrow_shapes[arrow_idx]['points'][3][0] - 5,
        #                 arrow_shapes[arrow_idx]['points'][3][1] - 5),
        #                 cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))
        #
        #     # write relation results to txt file to monitor
        #     with open(os.path.join(cfg.predict_folder, image_name + '_activate.txt'), 'a') as relation_fp:
        #         relation_fp.writelines(str(arrow_idx) + '\t'
        #                                + 'activate:' + arrow_relationships[arrow_idx] + '\n')
        #
        #     # write relation results to json file to save
        #     for label_shape in label.shapes:
        #         if label_shape['label'].split(':')[0] == 'activate' and \
        #                 arrow_shapes[arrow_idx]['points'] == label_shape['points']:
        #             label_shape['label'] = 'activate:' + arrow_relationships[arrow_idx]
        #             break
        #
        # # inhibit
        # inhibit_activator_list, inhibit_activator_neighbor_list, \
        #     inhibit_receptor_list, inhibit_receptor_neighbor_list, text_shapes, \
        #     inhibits_with_overlap, inhibits_without_overlap = find_all_inhibits_for_straight_line(image_path, label)
        #
        # inhibit_relationships = pair_gene(inhibit_activator_list, inhibit_activator_neighbor_list,
        #                                   inhibit_receptor_list, inhibit_receptor_neighbor_list,
        #                                   text_shapes,inhibits_with_overlap, image_file)
        #
        # inhibit_shapes = inhibits_without_overlap + inhibits_with_overlap
        #
        # # alternate the inhibit label
        # for inhibit_idx in range(len(inhibit_shapes)):
        #     cv2.drawContours(current_img, [np.array(inhibit_shapes[inhibit_idx]['points'],
        #                                             np.int32)], -1, (0, 255, 0), thickness=2)
        #     cv2.putText(current_img, str(inhibit_idx),
        #                 (inhibit_shapes[inhibit_idx]['points'][3][0] - 5,
        #                  inhibit_shapes[inhibit_idx]['points'][3][1] - 5),
        #                 cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
        #
        #     # write relation results to txt file to monitor
        #     with open(os.path.join(cfg.predict_folder, image_name + '_inhibit.txt'), 'a') as relation_fp:
        #         relation_fp.writelines(str(inhibit_idx) + '\t' + 'inhibit:'
        #                                + inhibit_relationships[inhibit_idx] + '\n')
        #
        #     for label_shape in label.shapes:
        #         if label_shape['label'].split(':')[0] == 'inhibit' \
        #                 and inhibit_shapes[inhibit_idx]['points'] == label_shape['points']:
        #             label_shape['label'] = 'inhibit:' + inhibit_relationships[
        #                                     inhibit_idx]
        #             break
        #
        # label.save(os.path.join(cfg.predict_folder, image_name + '.json'),
        #            label.shapes, image_file, None, None, None, None, {})

        cv2.imwrite(os.path.join(cfg.predict_folder, image_name + '_result_number'
                                 + image_ext), current_img)
        del current_img  # , label

    del user_words

    if cfg.ground_truth_folder and os.path.exists(cfg.ground_truth_folder):  # evaluate
        for value in cfg.detection_thresholds:
            calculate_all_metrics_by_json(value)

# end of file
