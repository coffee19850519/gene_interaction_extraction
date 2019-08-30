import os
import cfg
import cv2,shutil
import numpy as np
from OCR import OCR
from predict import predict
from network import East
from to_labelme import to_labelme
from evaluation import calculate_all_metrics_by_json
from loadexcldata import load_dictionary_from_excl
from predict_relationship import generate_relationship_shapes, predict_relationships, get_relationship_pairs,load_relation_predict_model
# from text_mining.biomedpdf_reader import  biomedpdf_reader
# from text_mining.gene_stat_collector import get_pair_counts
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


if __name__ == '__main__':

    if not os.path.isdir(cfg.predict_folder):
        os.mkdir(cfg.predict_folder)

    east = East()
    east_detect = east.east_network()
    east_detect.load_weights(cfg.saved_model_weights_file_path)

    # load ground truth into gene dictionary
    # if cfg.ground_truth_folder:
    #     load_ground_truth_into_excl(col_num=1)  # 1 for openpyxl, 0 for xlrd


    relation_model = load_relation_predict_model(cfg.sub_img_width_for_relation_predict,
                                                 cfg.sub_img_height_for_relation_predict,
                                                 cfg.num_channels)

    user_words = load_dictionary_from_excl(col_num=0)



    #pdf_reader = biomedpdf_reader(user_words)


    for image_file in os.listdir(cfg.image_folder):

        image_name, image_ext = os.path.splitext(image_file)
        if image_ext == ".json" or image_ext == ".pdf" or image_ext == ""  or image_ext == ".ini":
            continue

        # display(str(count) + ": \t" + str(image_file) + "\n", file=log_file)
        # calculate
        #all_pair_counts = pdf_reader.get_gene_pair_cooccurrence_counts('test/pdf/NCSLC_mechanisms/apigeninExample.pdf')

        # step 1: predict images
        image_path = os.path.join(cfg.image_folder, image_file)
        threshold_score_dict = {}
        threshold_relation_dict = {}
        threshold_boxes_dict = {}
        best_relation_boxes = []
        best_description = []
        #for testing set it to a fixed value
        for threshold in np.arange(0.95, 0.99, 0.05):
            predict_box, _ = predict(east_detect, image_path,
                                     text_pixel_threshold= threshold,
                                     action_pixel_threshold= threshold,
                                     quiet=True)

            with open(os.path.join(cfg.predict_folder, image_name + '_' + str(threshold) +'_predict.txt'), 'w') as result_fp:
                result_fp.writelines(predict_box)

            # step 2：DO OCR images
            current_img = cv2.imread(image_path)

            #creat a folder to save cropped text subimage
            ocr_sub_image_folder = os.path.join(cfg.predict_folder, 'OCR_' + image_name + '_' + str(threshold))

            if not os.path.exists(ocr_sub_image_folder):
                os.mkdir(ocr_sub_image_folder)

            OCR_results, all_results, corrected_results, fuzz_ratios, coordinates = OCR(image_file, ocr_sub_image_folder,
                                                                                        predict_box, user_words)
            shutil.rmtree(ocr_sub_image_folder)

            with open(os.path.join(cfg.predict_folder, image_name + '_' + str(threshold) + "_correct.txt"),
                      'w') as res_fp:
                for idx in range(len(OCR_results)):
                    # res_fp.write(str(idx) + '@' + str(corrections[idx]) + '@' + str(coordinates[idx])+'\n')
                    res_fp.write(str(idx) + '@' + str(OCR_results[idx]) + '@' + str(coordinates[idx]) + '\n')


            # # step 3: pair gene generate json file

            #generate all sub_images for current test sample
            sub_image_boxes, sub_image_entity_boxes, relationship_shapes, sub_image_folder = \
                generate_relationship_shapes(image_name, image_ext, threshold, predict_box, OCR_results, 10)

            filenames, predicted_classes = predict_relationships(relation_model, sub_image_folder,
                                                                 cfg.sub_img_width_for_relation_predict,
                                                                 cfg.sub_img_height_for_relation_predict,
                                                                 batch_size = 1)

            predicted_relationship_pairs, pair_descriptions, predicted_relationship_boxes = get_relationship_pairs(sub_image_boxes,
                                                                                        sub_image_entity_boxes,
                                                                                        filenames,
                                                                                        predicted_classes,
                                                                                        relationship_shapes)
            #update current threshold results to dicts
            threshold_boxes_dict[threshold] = predicted_relationship_boxes
            threshold_relation_dict[threshold] = pair_descriptions
            print('the predicted relationship pairs:', predicted_relationship_pairs)

            #step 4: get cooccurence of current group of candidates
            # predicted_pair_counts = get_pair_counts(all_pair_counts, predicted_relationship_pairs)
            # print(predicted_pair_counts)
            #step 5: score current group of candidates

            confidence_score = 1 # just a placeholder
            #update the score and threshold into dict
            threshold_score_dict[threshold] = confidence_score

        #outside the threshold for loop, pick the best score group
        # best_score = 10 # just a placeholder
        # best_threshold = threshold_score_dict[best_score]
        best_threshold = 0.95
        # best_relation_boxes = threshold_boxes_dict[best_threshold]
        # best_description = threshold_relation_dict[best_threshold]
        best_relation_boxes = threshold_boxes_dict[best_threshold]   #for test only
        best_description = threshold_relation_dict[best_threshold]   #for test only
        del threshold_score_dict, threshold_relation_dict, threshold_boxes_dict

        # convert best prediction results into json file
        # and plot the predicted information on original image
        results_string = to_labelme(image_name,
                   image_ext,
                   best_relation_boxes,
                   best_description,
                   os.path.join(cfg.predict_folder, image_name + '_' + str(best_threshold) + '_correct.txt'),
                   cfg.predict_folder,current_img)


        #save the plotted the best predictions' number on original image
        cv2.imwrite(os.path.join(cfg.predict_folder, image_name + '_result_number'
                                     + image_ext), current_img)
        with open(os.path.join(cfg.predict_folder, image_name + '_result_number'
                                     + '.txt'), 'w') as predict_results_fp:
            predict_results_fp.writelines(results_string)
        del current_img, best_relation_boxes, best_description    # , label

    del user_words, relation_model


    if cfg.ground_truth_folder and os.path.exists(cfg.ground_truth_folder):  # evaluate
        for value in cfg.detection_thresholds:
            calculate_all_metrics_by_json(value)

# end of file
