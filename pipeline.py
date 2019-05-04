import os
import time

import cfg

from OCR import OCR
from correct_gene_names import map_result_to_dictionary
from predict import predict
from network import East
from to_labelme import to_labelme
import cv2
import numpy as np
from detect_arrow_processing import find_all_arrows, pair_gene, find_all_inhibits
from label_file import LabelFile
from evaluation import caculate_all_metrics

from loadexcldata import load_genename_from_excl


if __name__ == '__main__':

    #reference to configuration file variable values(file path)
    img_folder = r'C:\Users\LSC-110\Desktop\Images'
    label_folder = r'C:\Users\LSC-110\Desktop\ground_truth'
    predict_folder = r'C:\Users\LSC-110\Desktop\results'

    dictionary_file = r'C:\Users\LSC-110\Desktop\gene_dictionary.xlsx'


    east = East()
    east_detect = east.east_network()
    east_detect.load_weights(cfg.saved_model_weights_file_path)
    #load gene dictionary
    # with open(cfg.dictionary_file,'r') as dict_fp:
    #     word_dictionary = dict_fp.readlines()
    user_words = load_genename_from_excl(dictionary_file)


    for image_file in os.listdir(img_folder):
        # step1： predict images
        if os.path.splitext(image_file)[1] == '':
            continue
        current_img = cv2.imread(os.path.join(img_folder, image_file))
        image_name, image_ext = os.path.splitext(image_file)
        predict_gene_box = predict(east_detect, os.path.join(img_folder,
                                                             image_file),
                                                            quiet=False)
        with open(os.path.join(predict_folder, image_name + '_predict.txt'),
                      'w') as result_fp:
                result_fp.writelines(predict_gene_box)

        # step2：DO OCR images-------------------------------------------
        #create sub_image folder
        sub_image_folder = os.path.join(predict_folder,image_name)
        if not os.path.exists(sub_image_folder):
            os.mkdir(sub_image_folder)
        OCR_results,coords = OCR(img_folder, image_file,sub_image_folder,
                                 predict_gene_box)
        corrections = map_result_to_dictionary(OCR_results, user_words, 95)
        with open(os.path.join(predict_folder, image_name +  "_correct.txt"),
                   'w') as res_fp:
            for idx in range(len(corrections)):
                res_fp.write(str(idx)+ '\t' + str(corrections[idx]) + '\t' +
                             str(coords[idx])+'\n')
                #draw the box and idx on image
                cv2.drawContours(current_img,[np.array(coords[idx],np.int32)]
                                 ,-1,(255,0,0), thickness=2)
                cv2.putText(current_img,str(idx), (coords[idx][3][0]-5,
                            coords[idx][3][1]-5),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,
                            ((0,0,255)))
        del corrections
        # if os.path.exists(sub_image_folder):
        #     os.remove(sub_image_folder)
        cv2.imwrite(os.path.join(predict_folder,image_file), current_img)
        # step3: transfer to Json labelfile----------------------------------
        to_labelme(image_file,
                   os.path.join(predict_folder, image_name + '_predict.txt'),
                   os.path.join(predict_folder, image_name + '_correct.txt'),
                   predict_folder)


        #step4 pair gene generate json file
        # label_file = os.path.join(predict_folder, image_file + '.json')
        # #label = LabelFile(label_file)
        #
        #
        # activator_list, activator_neighbor_list, receptor_list, receptor_neighbor_list, text_shapes, arrow_shapes = find_all_arrows(
        #     os.path.join(img_folder, image_file), label_file)
        #
        # arrow_relationships = pair_gene(arrow_activator_list,
        #                                arrow_receptor_list,text_shapes,img_file)
        # # 存arrow 的dict
        # for idx in range(len(arrow_shapes)):
        #     tempDict = {}
        #     tempDict['label'] =arrow_relationships[idx]
        #     tempDict['line_color'] = None
        #     tempDict['fill_color'] = None
        #     tempDict['points'] = arrow_shapes[idx]['points']
        #     tempDict['shape_type'] = 'polygon'
        #     tempDict['alias'] = 'name'
        #     shapes.append(tempDict)
        # # inhibit
        # activator_list, activator_neighbor_list, receptor_list, receptor_neighbor_list, text_shapes, arrow_shapes = find_all_arrows(
        #     img_file, label_file)
        #
        # inhibit_relationships=pair_gene(inhibit_activator_list,inhibit_receptor_list,text_shapes,img_file)
        # for idx in range(len(inhibit_shapes)):
        #     tempDict = {}
        #     tempDict['label'] =inhibit_relationships[idx]
        #     tempDict['line_color'] = None
        #     tempDict['fill_color'] = None
        #     tempDict['points'] = arrow_shapes[idx]['points']
        #     tempDict['shape_type'] = 'polygon'
        #     tempDict['alias'] = 'name'
        #     shapes.append(tempDict)
        # # 存text的dict
        # for i in text_shapes:
        #     shapes.append(i)
        # label.save(os.path.join(cfg.final_json_path, image[:-4]) + '.json', shapes, image, None, None, None, None, {})
        # end2 = time.clock()
        #
        # #draw all results on original image
        # del current_img
    del east_detect, user_words

    #last step: calculate all assessment
    if label_folder is not None:
        caculate_all_metrics(img_folder, label_folder, predict_folder,
                             predict_folder, 0.1)