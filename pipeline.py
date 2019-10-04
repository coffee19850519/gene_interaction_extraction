import os
import cfg
import cv2,shutil
import numpy as np
from OCR import OCR
from predict import predict
from network import East
from to_labelme import to_labelme,predict_to_labelme
from evaluation import calculate_all_metrics_by_json,calculate_all_metrics_best_candidate_by_json
from loadexcldata import load_dictionary_from_excl
from data_generation.detect_arrow_processing import find_all_arrows, pair_gene, find_all_inhibits
from predict_relationship import load_relation_predict_model
from get_pdf_from_image import pdf_from_image_name
from text_mining.biomedpdf_reader import  biomedpdf_reader
from text_mining.gene_stat_collector import get_pair_counts, counted_score
from label_file import LabelFile
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import json


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



    pdf_reader = biomedpdf_reader(user_words)

    image_best_threshold_dict = {}
    for image_file in os.listdir(cfg.image_folder):

        image_name, image_ext = os.path.splitext(image_file)
        if image_ext == ".json" or image_ext == ".pdf" or image_ext == ""  or image_ext == ".ini":
            continue

        pdf_path = pdf_from_image_name(image_name)
        # display(str(count) + ": \t" + str(image_file) + "\n", file=log_file)
        # calculate
        if len(pdf_path) == 1:
            pdf_path = pdf_path[0]
            all_pair_counts = pdf_reader.get_gene_pair_cooccurrence_counts(pdf_path)
            pdf_name = pdf_path.split('\\')[-1]
            if len(pdf_name) > 20:
                pdf_name = pdf_name[:20] + '...'
            mediancount = np.median([item[1] for item in all_pair_counts.items()])
            print('all counts for {} are \n {}'.format(pdf_name, all_pair_counts))
        elif len(pdf_path) > 1:
            print("warning found multiple pdfs for image: {}".format(image_name))
        else:
            print("warning found no pdfs for image: {}".format(image_name))

        # step 1: predict images
        image_path = os.path.join(cfg.image_folder, image_file)
        threshold_score_dict = {}
        threshold_relation_dict = {}
        threshold_boxes_dict = {}
        best_relation_boxes = []
        best_description = []
        best_score = 0
        best_threshold = 0
        #for testing set it to a fixed value
        for threshold in np.arange(cfg.threshold_start_point, cfg.threshold_end_point, cfg.threshold_step):
            if cfg.predict_bounding_box:
                predict_box, _ = predict(east_detect, image_path,
                                         text_pixel_threshold= threshold,
                                         action_pixel_threshold= threshold,
                                         quiet=True)
                with open(os.path.join(cfg.predict_folder, image_name + '_' + str(threshold) + '_predict.txt'),
                          'w') as result_fp:
                     result_fp.writelines(predict_box)
            else:
                predict_box = []
                try:
                    print('loading predict box')
                    with open(os.path.join(cfg.predict_folder, image_name + '_' + str(threshold) +'_predict.txt'), 'r') as result_fp:
                        for file_line in result_fp.readlines():
                            predict_box.append(file_line)
                except:
                    print("failed for threshold {} on image {}".format(threshold, image_file))
                    continue

            current_img = cv2.imread(image_path)
            if cfg.predict_ocr:
                # step 2：DO OCR images


                # creat a folder to save cropped text subimage
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
            else:
                OCR_results = []
                coordinates = []
                print('loading OCR')
                try:
                    with open(os.path.join(cfg.predict_folder, image_name + '_' + str(threshold) + "_correct.txt"),
                              'r') as res_fp:
                        for file_line in res_fp.readlines():
                            filesplitat = file_line.split('@')
                            OCR_results.append(filesplitat[1])
                            coordinates.append(json.loads(filesplitat[2]))
                except:
                    print('failed on threshold {} for image {}'.format(threshold, image_file))
                    continue


            # # step 3: pair gene generate json file
            if cfg.predict_relation:
                #generate all sub_images for current test sample
                # sub_image_boxes, sub_image_entity_boxes, relationship_shapes, sub_image_folder = \
                #     generate_relationship_shapes(image_name, image_ext, threshold, predict_box, OCR_results, 10)
                #
                # filenames, predicted_classes = predict_relationships(relation_model, sub_image_folder,
                #                                                      cfg.sub_img_width_for_relation_predict,
                #                                                      cfg.sub_img_height_for_relation_predict,
                #                                                      batch_size = 1)
                #
                # predicted_relationship_pairs, pair_descriptions, predicted_relationship_boxes = get_relationship_pairs(sub_image_boxes,
                #                                                                             sub_image_entity_boxes,
                #                                                                             filenames,
                #                                                                             predicted_classes,
                #                                                                             relationship_shapes)
                #
                # #update current threshold results to dicts
                # print('the predicted relationship pairs:', predicted_relationship_pairs)
                # with open(os.path.join(cfg.predict_folder, image_name + '_' + str(threshold) + "_relationship.json"), 'w') as predict_file:
                #     json.dump([predicted_relationship_pairs, pair_descriptions, predicted_relationship_boxes],predict_file)

                #label_file = os.path.join(predict_folder, image_file + '.json')
                #label = LabelFile(label_file)

                # save predict results to json file
                predict_to_labelme(image_file, os.path.join(cfg.predict_folder, image_name + '_' + str(threshold) +'_predict.txt'),
                                   os.path.join(cfg.predict_folder, image_name + '_' + str(threshold) + '_correct.txt'),
                                   cfg.predict_folder)

                label = LabelFile(os.path.join(cfg.predict_folder, image_name + '_' + str(threshold) +'_predict.json'))
                activator_list, activator_neighbor_list, receptor_list, receptor_neighbor_list, text_shapes, arrow_boxes = find_all_arrows(
                    image_path, label)

                arrow_relationship_pairs, arrow_descriptions, arrow_relationship_boxes = pair_gene(activator_list, activator_neighbor_list, receptor_list,
                                                receptor_neighbor_list, text_shapes, arrow_boxes, cfg.predict_folder,cfg.image_folder,image_path)
                for arrow_description in arrow_descriptions:
                    arrow_description = 'activate:' + arrow_description

                # inhibit
                activator_list, activator_neighbor_list, receptor_list, receptor_neighbor_list, text_shapes, inhibit_boxes = find_all_inhibits(
                    image_path, label)

                inhibit_relationship_pairs, inhibit_descriptions, inhibit_relationship_boxes=pair_gene(activator_list, activator_neighbor_list, receptor_list, receptor_neighbor_list, text_shapes, inhibit_boxes,cfg.predict_folder,cfg.image_folder,image_path)
                for inhibit_description in inhibit_descriptions:
                    inhibit_description = 'inhibit:' + inhibit_description
                predicted_relationship_pairs = arrow_relationship_pairs + inhibit_relationship_pairs

                pair_descriptions = arrow_descriptions + inhibit_descriptions

                predicted_relationship_boxes = arrow_relationship_boxes + inhibit_relationship_boxes
                #draw all results on original image
                #del current_img
            else:
                with open(os.path.join(cfg.predict_folder, image_name + '_' + str(threshold) + "_relationship.json"), 'r') as predict_file:
                    [predicted_relationship_pairs, pair_descriptions, predicted_relationship_boxes] = json.load(predict_file)

            threshold_boxes_dict[threshold] = predicted_relationship_boxes
            threshold_relation_dict[threshold] = pair_descriptions
            #step 4:# convert best prediction results into json file
                    # and plot the predicted information on original image
            results_string = to_labelme(image_name,
                                        image_ext,
                                        threshold,
                                        predicted_relationship_boxes,
                                        pair_descriptions,
                                        cfg.predict_folder,
                                        current_img)

            # plotted predictions' number on original image
            cv2.imwrite(os.path.join(cfg.predict_folder, image_name + '_' + str(threshold) +'_result_number'
                                     + image_ext), current_img)
            with open(os.path.join(cfg.predict_folder, image_name + '_' + str(threshold) +'_result_number'
                                                       + '.txt'), 'w') as predict_results_fp:
                predict_results_fp.writelines(results_string)

            #step 5: get cooccurence of current group of candidates
            predicted_pair_counts = get_pair_counts(all_pair_counts, predicted_relationship_pairs)
            print('counts with threshold {} are {}'.format(threshold, predicted_pair_counts))

            # step 6: score current group of candidates
            confidence_score = sum([counted_score(predicted_pair[1], mediancount) for predicted_pair in
                                    predicted_pair_counts.items()])  # just a placeholder
            best_score, best_threshold = max((best_score, best_threshold), (confidence_score, threshold))
            print('confidence with threshold {} is {}\n'.format(threshold, confidence_score))
            # update the score and threshold into dict
            threshold_score_dict[threshold] = confidence_score


        #outside the threshold for loop, pick the best score group

        # best_relation_boxes = threshold_boxes_dict[best_threshold]
        # best_description = threshold_relation_dict[best_threshold]
        image_best_threshold_dict[image_name] = best_threshold
        best_relation_boxes = threshold_boxes_dict[best_threshold]   #for test only
        best_description = threshold_relation_dict[best_threshold]   #for test only
        del threshold_score_dict, threshold_relation_dict, threshold_boxes_dict



        #delete the intermediate text files
        #for threshold in np.arange(cfg.threshold_start_point, cfg.threshold_end_point, cfg.threshold_step):
        #    if os.path.exists(os.path.join(cfg.predict_folder, image_name + '_' + str(threshold) + "_correct.txt")):
        #        os.remove(os.path.join(cfg.predict_folder, image_name + '_' + str(threshold) + "_correct.txt"))
        #    if os.path.exists(os.path.join(cfg.predict_folder, image_name + '_' + str(threshold) +'_predict.txt')):
        #        os.remove(os.path.join(cfg.predict_folder, image_name + '_' + str(threshold) + "_predict.txt"))

        del current_img, best_relation_boxes, best_description    # , label

    del user_words, relation_model

    #evaluate
    if cfg.ground_truth_folder and os.path.exists(cfg.ground_truth_folder):
        # evaluate all threshold results
        for threshold in np.arange(cfg.threshold_start_point, cfg.threshold_end_point, cfg.threshold_step):
            for value in cfg.detection_IoU_thresholds:
                calculate_all_metrics_by_json(value, threshold)

        #evaluate best prediction
        for value in cfg.detection_IoU_thresholds:
            calculate_all_metrics_best_candidate_by_json(value, image_best_threshold_dict)


    del image_best_threshold_dict
# end of file
