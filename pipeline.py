import os
import cfg
import cv2
import numpy as np
from loadexcldata import load_ground_truth_into_excl
from OCR import OCR
from predict import predict
from network import East
from to_labelme import to_labelme
from evaluation import calculate_all_metrics_by_json
from loadexcldata import load_dictionary_from_excl
from predict_relationship import generate_relationship_shapes, predict_relationships, get_relationship_pairs


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

    all_sub_image_boxes = {}
    all_sub_image_entity_boxes = {}
    all_relationship_shapes = {}

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
        sub_image_boxes, sub_image_entity_boxes, relationship_shapes = \
            generate_relationship_shapes(image_file, predict_box, OCR_results, 10)

        all_sub_image_boxes[image_file] = sub_image_boxes
        all_sub_image_entity_boxes[image_file] = sub_image_entity_boxes
        all_relationship_shapes[image_file] = relationship_shapes

        cv2.imwrite(os.path.join(cfg.predict_folder, image_name + '_result_number'
                                 + image_ext), current_img)
        del current_img  # , label

    del user_words

    filenames, predicted_classes = predict_relationships()
    get_relationship_pairs(all_sub_image_boxes, all_sub_image_entity_boxes, filenames, predicted_classes,
                           all_relationship_shapes)

    if cfg.ground_truth_folder and os.path.exists(cfg.ground_truth_folder):  # evaluate
        for value in cfg.detection_thresholds:
            calculate_all_metrics_by_json(value)

# end of file
