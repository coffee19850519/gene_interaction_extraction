import numpy as np
from shapely.geometry import Polygon
import cfg,os
from label_file import LabelFile


def intersection(g, p):
    if len(g) !=4:
        if len(g) == 2:
          #convert it into 8-dimension
          p1=[g[0][0],g[0][1]]
          p2=[g[1][0],g[0][1]]
          p3=[g[1][0],g[1][1]]
          p4=[g[0][0],g[1][1]]
          g=[p1,p2,p3,p4]
          print('ground truth polygon dimension is 4, its type:' + str(type(g)))
        else:
          print('ground truth polygon dimension is incompatible')
          return 0
    if  len(p) != 4:
        print('predicted polygon demension is incompatible')
        return 0

    g = Polygon(np.array(g[:4]).reshape((4, 2)))
    p = Polygon(np.array(p[:4]).reshape((4, 2)))
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    # union = g.area + p.area - inter
    if g.area == 0:
        return 0
    else:
        return inter/g.area


def calculate_detection_metrics(gt_boxes, pd_boxes, threshold):
    pos_correct = 0
    pos_error = 0
    num_gt = len(gt_boxes)
    num_pd = len(pd_boxes)
    if len(pd_boxes)==0 and len(gt_boxes)==0:
        precision=1
        recall=1
    elif len(pd_boxes)==0 and len(gt_boxes)!=0:
        precision = 0
        recall = 0
    elif len(pd_boxes) != 0 and len(gt_boxes) == 0:
        precision=0
        recall=0
    else:
        for predict_box in pd_boxes:
            IoU = []
            for gt in gt_boxes:
              IoU.append(intersection(gt,predict_box))
            best_match = max(IoU)
            if best_match >= threshold:
              # target one label successfully
              pos_correct = pos_correct + 1
              # remove the match label from ground truth
              list(gt_boxes).pop(IoU.index(best_match))
            else:
              # fail to target any label
              pos_error = pos_error + 1
            del  IoU
            if len(gt_boxes) == 0:
              #all labels have been matched
              pos_error = pos_error + num_pd - 1 - list(pd_boxes).index(predict_box)
              break

    #calculate the metrics
        precision = float(pos_correct) / num_pd
        recall = float(pos_correct) / num_gt
    return precision,recall,pos_correct,num_pd,num_gt

def calculate_gene_match_metrics(gt_list, pd_list):
    pos_correct = 0
    pos_error = 0
    num_gt = len(gt_list)
    num_pd = len(pd_list)
    if len(pd_list)==0 and len(gt_list)==0:
        precision=1
        recall=1
    elif len(pd_list)==0 and len(gt_list)!=0:
        precision = 0
        recall = 0
    elif len(pd_list) != 0 and len(gt_list) == 0:
        precision=0
        recall=0
    else:
        for predict_gene in pd_list:
            match = False
            for gt in gt_list:
                if predict_gene in gt:
                  pos_correct = pos_correct + 1
                  match = True
                  break
            if not match:
              pos_error = pos_error + 1

        # calculate the metrics
        precision = float(pos_correct) / num_pd
        recall = float(pos_correct) / num_gt
    return precision, recall, pos_correct, num_pd, num_gt


def caculate_all_metrics(img_path, ground_truth_path,
                                   predict_result_path,
                                   evaluation_result_path,
                                   detection_threshold):

    # declaim some variances for computing final assessment
    pos_genebox_correct_detect_total = 0
    num_genebox_pd_detect_total = cfg.epsilon
    num_genebox_gt_detect_total = cfg.epsilon
    pos_correct_gene_total = 0
    num_pd_gene_total = cfg.epsilon
    num_gt_gene_total = cfg.epsilon
    pos_arrowbox_correct_detect_total = 0
    num_arrowbox_pd_detect_total = cfg.epsilon
    num_arrowbox_gt_detect_total = cfg.epsilon
    pos_inhibitbox_correct_detect_total = 0
    num_inhibitbox_pd_detect_total = cfg.epsilon
    num_inhibitbox_gt_detect_total = cfg.epsilon
    pos_relation_correct_detect_total = 0
    num_relation_pd_detect_total = cfg.epsilon
    num_relation_gt_detect_total = cfg.epsilon
    # load ground truthlabels
    with open(os.path.join(evaluation_result_path, 'evaluate.txt'),
              'a') as result_fp:
      result_fp.write(
          str.format(
            '%s \t %s \t %s \t %s \t %s \t %s \t %s \t %s \t %s \t %s \t %s \t %s \t %s \t %s \t %s \t %s \t'
            ' %s \t %s \t %s \t %s \t %s \t %s \t %s \t %s \t %s \t %s\n')
          % (
            'imagename', 'genebox_recall','genebox_precision',
            'arrowbox_recall','arrowbox_precision', 'inhibitbox_recall', 'inhibitbox_precision',
            'gene_recall','gene_precision','relation_recall','relation_precision',
            'genebox_correct_num', 'genebox_pd_num', 'genebox_gt_num', 'arrowbox_correct_num',
            'arrowbox_pd_num', 'arrowbox_gt_num','inhibitbox_correct_num',
            'inhibitbox_pd_num', 'inhibitbox_gt_num', 'gene_correct_num', 'gene_pd_num','gene_gt_num', 'relation_correct_num',
            'relation_pd_num', 'relation_gt_num'))


    for image_file in os.listdir(img_path):
      # label = LabelFile(os.path.join(path+r'\gt', image_file[:-4] + '.json'))
      image_name, image_ext = os.path.splitext(image_file)
      try:
          label = LabelFile(
            os.path.join(ground_truth_path, image_name + '.json'))
      except:
          print('we do not have ground truth for this input yet')

      # load ground truth genelabels
      gene_box_gt = label.get_all_boxes_for_category('text')
      gene_name_gt = label.get_all_genes()
      # load ground truth arrow labels
      arrow_box_gt = label.get_all_boxes_for_category('arrow')
      # load ground truth relationship
      relation_name_gt = label.get_all_relations()
      # load ground truth inhibit labels
      inhibit_box_gt = label.get_all_boxes_for_category('nock')
      # load predict labels
      #     labelp=LabelFile(os.path.join(path+r'\pt',image_file[:-4]+'.json'))
      labelp = LabelFile(os.path.join(predict_result_path, image_file[:-4] + '.json'))
      # load predict gene labels
      predict_gene_box = labelp.get_all_boxes_for_category('text')
      # load predict arrow labels
      predict_arrow_box = labelp.get_all_boxes_for_category('arrow')
      # load predict relationship
      relation_names = labelp.get_all_relations()

      # load predict inhibit labels
      predict_inhibit_box = labelp.get_all_boxes_for_category('nock')
      # load predict genenames
      gene_names = labelp.get_all_genes()
      # evaluate the result of detection
      # evaluate the result of gene box detection
      precision_genebox_detect, recall_genebox_detect, pos_genebox_correct_detect, num_genebox_pd_detect, \
      num_genebox_gt_detect = calculate_detection_metrics(gene_box_gt,
                                                          predict_gene_box, detection_threshold)
      # evaluate the gene name recognition
      precision_gene, recall_gene, pos_correct_gene, num_pd_gene, num_gt_gene = \
        calculate_gene_match_metrics(gene_name_gt, gene_names)
      # evaluate the arrow box recognition
      precision_arrowbox_detect, recall_arrowbox_detect, pos_arrowbox_correct_detect, num_arrowbox_pd_detect, \
      num_arrowbox_gt_detect = calculate_detection_metrics(arrow_box_gt,
                                                           predict_arrow_box, detection_threshold)
      # evaluate the inhibit box recognition
      precision_inhibitbox_detect, recall_inhibitbox_detect, pos_inhibitbox_correct_detect, num_inhibitbox_pd_detect, \
      num_inhibitbox_gt_detect = calculate_detection_metrics(inhibit_box_gt,
                                                             predict_inhibit_box,
                                                             detection_threshold)
      # evaluate the gene relation recognition
      precision_relation, recall_relation, pos_correct_relation, num_pd_relation, num_gt_relation = \
        calculate_gene_match_metrics(relation_name_gt, relation_names)
      # save assessment to file
      with open(os.path.join(evaluation_result_path, 'evaluate.txt'),
                'a') as result_fp:
        result_fp.write(
            str.format(
              '%s \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f\t %f \t %f \t %f \t %f \t %f \t'
              ' %f \t %f \t %f \t %f \t %f\t %f \t %f \t %f \t %f \t %f\n')

            % (label.filename,  recall_genebox_detect,
               precision_genebox_detect, recall_arrowbox_detect, precision_arrowbox_detect, \
               recall_inhibitbox_detect, precision_inhibitbox_detect, \
               recall_gene,precision_gene,recall_relation,precision_relation, \
               pos_genebox_correct_detect, num_genebox_pd_detect,num_genebox_gt_detect, \
               pos_arrowbox_correct_detect, num_arrowbox_pd_detect,  num_arrowbox_gt_detect,\
               pos_inhibitbox_correct_detect, num_inhibitbox_pd_detect, num_inhibitbox_gt_detect,
                pos_correct_gene, num_pd_gene, num_gt_gene,
               pos_correct_relation,
               num_pd_relation, num_gt_relation))
      # accumulate
      # genebox
      pos_genebox_correct_detect_total += pos_genebox_correct_detect
      num_genebox_pd_detect_total += num_genebox_pd_detect
      num_genebox_gt_detect_total += num_genebox_gt_detect
      # genenames
      pos_correct_gene_total += pos_correct_gene
      num_pd_gene_total += num_pd_gene
      num_gt_gene_total += num_gt_gene
      # arrow
      pos_arrowbox_correct_detect_total += pos_arrowbox_correct_detect
      num_arrowbox_pd_detect_total += num_arrowbox_pd_detect
      num_arrowbox_gt_detect_total += num_arrowbox_gt_detect
      # inhibit
      pos_inhibitbox_correct_detect_total += pos_inhibitbox_correct_detect
      num_inhibitbox_pd_detect_total += num_inhibitbox_pd_detect
      num_inhibitbox_gt_detect_total += num_inhibitbox_gt_detect
      # relation
      pos_relation_correct_detect_total += pos_correct_relation
      num_relation_pd_detect_total += num_pd_relation
      num_relation_gt_detect_total += num_gt_relation

    del predict_gene_box, gene_names, label, labelp, gene_box_gt, \
      gene_name_gt, precision_genebox_detect, recall_genebox_detect, pos_genebox_correct_detect, \
      num_genebox_pd_detect, num_genebox_gt_detect, precision_gene, \
      recall_gene, pos_correct_gene, num_pd_gene, num_gt_gene, precision_inhibitbox_detect, recall_inhibitbox_detect, \
      pos_inhibitbox_correct_detect, num_inhibitbox_pd_detect, \
      num_inhibitbox_gt_detect, precision_relation, recall_relation, pos_correct_relation, num_pd_relation, num_gt_relation

    # calculate the overall metrics
    overall_genebox_precision_detect = pos_genebox_correct_detect_total / num_genebox_pd_detect_total
    overall_genebox_recall_detect = pos_genebox_correct_detect_total / num_genebox_gt_detect_total
    overall_precision_gene = pos_correct_gene_total / num_pd_gene_total
    overall_recall_gene = pos_correct_gene_total / num_gt_gene_total
    overall_arrowbox_precision_detect = pos_arrowbox_correct_detect_total / num_arrowbox_pd_detect_total
    overall_arrowbox_recall_detect = pos_arrowbox_correct_detect_total / num_arrowbox_gt_detect_total
    overall_inhibitbox_precision_detect = pos_inhibitbox_correct_detect_total / num_inhibitbox_pd_detect_total
    overall_inhibitbox_recall_detect = pos_inhibitbox_correct_detect_total / num_inhibitbox_gt_detect_total
    overall_relation_precision_detect = pos_relation_correct_detect_total / num_relation_pd_detect_total
    overall_relation_recall_detect = pos_relation_correct_detect_total / num_relation_gt_detect_total
    with open(os.path.join(evaluation_result_path, 'evaluate.txt'),
              'a') as result_fp:
      result_fp.write(
          str.format(
            '%s \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f\t %f \t %f \t %f \t %f \t %f \t'
            ' %f \t %f \t %f \t %f \t %f\t %f \t %f \t %f \t %f \t %f \n')
          % ('total evaluation', overall_genebox_recall_detect,overall_genebox_precision_detect,
             overall_arrowbox_recall_detect, overall_arrowbox_precision_detect,
             overall_inhibitbox_recall_detect, overall_inhibitbox_precision_detect,
             overall_recall_gene, overall_precision_gene,
             overall_relation_recall_detect,overall_relation_precision_detect,pos_genebox_correct_detect_total,
             num_genebox_pd_detect_total, num_genebox_gt_detect_total,
             pos_arrowbox_correct_detect_total,
             num_arrowbox_pd_detect_total, num_arrowbox_gt_detect_total,
             pos_inhibitbox_correct_detect_total,
             num_inhibitbox_pd_detect_total, num_inhibitbox_gt_detect_total,
             pos_correct_gene_total, num_pd_gene_total, num_gt_gene_total,
             pos_relation_correct_detect_total,num_relation_pd_detect_total,
             num_relation_gt_detect_total))



