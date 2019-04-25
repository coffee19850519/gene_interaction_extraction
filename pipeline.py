import os
import time

import cfg

from OCR import OCR
from correct_gene_names import map_result_to_dictionary
from predict import predict
from network import East
from to_labelme import to_labelme

from detect_arrow_processing import find_all_arrows, pair_gene, find_all_inhibits
from label_file import LabelFile

from loadexcldata import load_genename_from_excl


if __name__ == '__main__':
    # img_path = r'D:\Study\Master\1st deep learning project\Dima data(2)\Dima data\imagetest'
    #reference to configuration file variable values(file path)
    img_path =cfg.image_path
    correctfolder =cfg.correct_result_path
    predictfolder =cfg.predict_result_path
    to_labelmefolder =cfg.result_in_json_path

    east = East()
    east_detect = east.east_network()
    east_detect.load_weights(cfg.saved_model_weights_file_path)
    #load gene dictionary
    # with open(cfg.dictionary_file,'r') as dict_fp:
    #     word_dictionary = dict_fp.readlines()

    # declaim some variances for computing final assessment
    pos_correct_detect_total = 0
    num_pd_detect_total = 0
    num_gt_detect_total = 0
    pos_correct_gene_total = 0
    num_pd_gene_total = 0
    num_gt_gene_total = 0
    # step1： predict images
    for image_file in os.listdir(img_path):
        if os.path.splitext(image_file)[1] == '':
            continue
        predict_gene_box = predict(east_detect, os.path.join(img_path,
                                                             image_file), quiet=False)
        with open(os.path.join(predictfolder, image_file + '.txt'),
                      'w') as result_fp:
                result_fp.writelines(predict_gene_box)

    # step2：DO OCR images-------------------------------------------
        filename = cfg.dictionary_file
        user_words = load_genename_from_excl(filename)
        OCR_results,coords = OCR(img_path, image_file, predict_gene_box)
        corrections = map_result_to_dictionary(OCR_results, user_words)
        with open(cfg.correct_result_path+'\\'+image_file[:-4]+"_correct.txt", 'w') as res_fp:
            for idx in range(len(corrections)):
                res_fp.write(str(corrections[idx]) + '\t' + str(coords[idx])+'\n')
        del corrections
    # step3: transfer to Json labelfile----------------------------------
    ocr_path_list = []
    predict_path_list = []
    for txt in os.listdir(correctfolder):
        ocr_path = os.path.join(correctfolder, txt)
        ocr_path_list.append(ocr_path)
    for txt in os.listdir(predictfolder):
        predict_path = os.path.join(predictfolder, txt)
        predict_path_list.append(predict_path)
    id = 0
    while id < len(ocr_path_list):
        to_labelme(predict_path_list[id], ocr_path_list[id], to_labelmefolder)
        id = id + 1

    #step4 pair gene   generate json file
    for image in os.listdir(img_path):
        shapes = []
        tempDict2 = {}
    # image = r'cin_00001.png'
        img_file = os.path.join(img_path, image)
        label_file = os.path.join(to_labelmefolder, image[:-4] + '.json')
        label = LabelFile(label_file)
        start = time.clock()
        arrow_activator_list,arrow_receptor_list,text_shapes,arrow_shapes=find_all_arrows(img_file, label_file)
        end = time.clock()
        start1 = time.clock()
        arrow_relationships=pair_gene(arrow_activator_list,arrow_receptor_list,text_shapes,img_file)
        end1 = time.clock()

        # 存arrow 的dict
        start2 = time.clock()
        for idx in range(len(arrow_shapes)):
            tempDict = {}
            tempDict['label'] =arrow_relationships[idx]
            tempDict['line_color'] = None
            tempDict['fill_color'] = None
            tempDict['points'] = arrow_shapes[idx]['points']
            tempDict['shape_type'] = 'polygon'
            tempDict['alias'] = 'name'
            shapes.append(tempDict)
        # inhibit
        inhibit_activator_list,inhibit_receptor_list,text_shapes,inhibit_shapes=find_all_inhibits(img_file, label_file)
        inhibit_relationships=pair_gene(inhibit_activator_list,inhibit_receptor_list,text_shapes,img_file)
        for idx in range(len(inhibit_shapes)):
            tempDict = {}
            tempDict['label'] =inhibit_relationships[idx]
            tempDict['line_color'] = None
            tempDict['fill_color'] = None
            tempDict['points'] = arrow_shapes[idx]['points']
            tempDict['shape_type'] = 'polygon'
            tempDict['alias'] = 'name'
            shapes.append(tempDict)
        # 存text的dict
        for i in text_shapes:
            shapes.append(i)
        label.save(os.path.join(cfg.final_json_path, image[:-4]) + '.json', shapes, image, None, None, None, None, {})
        end2 = time.clock()
        print('find arrow activator Running time: %s Seconds'%(end-start))
        print('find relationship Running time: %s Seconds' % (end1 - start1))
        print('save in json Running time: %s Seconds' % (end2 - start2))
        # tempDict2['version'] = '3.6.1'
        # tempDict2['flags'] = {}
        # tempDict2['shapes'] = shapes
        # tempDict2['lineColor'] = None
        # tempDict2['fillColor'] = None
        # tempDict2['imagePath'] = label.imagePath
        # tempDict2['imageData'] = None
        # tempDict2['imageHeight'] = None
        # tempDict2['imageWidth'] = None

        # f = open(os.path.join(pair_gen_path, label.imagePath[:-4]) + '.json', 'a', encoding="utf-8")
        # # f = open(os.path.join(pair_gen_path, label.imagePath.split('\\')[2])[:-4] + '.json', 'a', encoding="utf-8")
        # tempJson = json.dumps(tempDict2, indent=1, ensure_ascii=False)
        # f.write(tempJson + '\n')
        # f.close()
        # print(tempJson)

    # img_path_list=[]
    # json_path_list=[]
    # for img in os.listdir(img_path):
    #     path=os.path.join(img_path,img)
    #     img_path_list.append(path)
    # for js in os.listdir(to_labelmefolder):
    #     jspath=os.path.join(to_labelmefolder,js)
    #     json_path_list.append(jspath)
    # id = 0
    # while id < len(img_path_list):
    #     find_all_arrows(img_path_list[id],json_path_list[id])
    #     id = id + 1




#     #load ground truth gene labels
#     label = LabelFile(os.path.join(img_path, image_file[:-4] + '.txt'))
#     gene_box_gt = label.get_all_boxes_for_category('gene')
#     gene_name_gt = label.get_all_genes()
#
#     # evaluate the result of detection
#     precision_detect, recall_detect, pos_correct_detect, num_pd_detect, \
#     num_gt_detect=calculate_detection_metrics(gene_box_gt,predict_gene_box)
#
#     #evaluate the gene name recognition
#     precision_gene, recall_gene, pos_correct_gene, num_pd_gene, num_gt_gene = \
#         calculate_gene_match_metrics(gene_name_gt, gene_names)
#
#
#      #save result to file
#     with open(os.path.join(img_path, image_file[:-4] + '_predict.txt'),
#             'w') as result_fp:
#         for gene, box in zip(gene_names, predict_gene_box):
#             result_fp.write(label.filename + '\t' + gene + '\t' + box)
#
#
#      #save assessment to file
#     with open(os.path.join(img_path, 'evaluate.txt'),
#             'a') as result_fp:
#         result_fp.write(
#             str.format('%s \t %f \t %f \t %f \t %f \n')
#             %(label.filename, precision_detect, recall_detect,
#             precision_gene, recall_gene))
#
#     #accumulate
#     pos_correct_detect_total += pos_correct_detect
#     num_pd_detect_total += num_pd_detect
#     num_gt_detect_total += num_gt_detect
#     pos_correct_gene_total += pos_correct_gene
#     num_pd_gene_total += num_pd_gene
#     num_gt_gene_total += num_gt_gene
#
#     del predict_gene_box, OCR_results,gene_names, label, gene_box_gt,\
#         gene_name_gt,precision_detect, recall_detect, pos_correct_detect, \
#         num_pd_detect, num_gt_detect, precision_gene, \
#         recall_gene, pos_correct_gene, num_pd_gene, num_gt_gene
#
# #calculate the overall metrics
# overall_precision_detect = pos_correct_detect_total / num_pd_detect_total
# overall_recall_detect = pos_correct_detect_total / num_gt_detect_total
# overall_precision_gene = pos_correct_gene_total / num_pd_gene_total
# overall_recall_gene = pos_correct_gene_total / num_gt_gene_total