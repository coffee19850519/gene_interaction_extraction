import numpy as np
from shapely.geometry import Polygon


def intersection(g, p):
    if len(g) != 8:
        if len(g) == 4:
          #convert it into 8-dimension
          print('ground truth polygen dimension is 4, its type:' + str(type(g)))
        else:
          print('ground truth polygen dimension is incompatible')
          return 0
    if  len(p) != 8:
        print('predicted polygen demension is incompatible')
        return 0

    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter/union


def calculate_detection_metrics(gt_boxes, pd_boxes, threshold):
    pos_correct = 0
    pos_error = 0
    num_gt = len(gt_boxes)
    num_pd = len(pd_boxes)

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






