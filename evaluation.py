import numpy as np
from shapely.geometry import Polygon
import cfg
import os
from label_file import LabelFile
from matplotlib import pyplot as plt


def intersection(g, p):
    if len(g) != 4:
        if len(g) == 2:
            # convert it into 8-dimension
            p1 = [g[0][0], g[0][1]]
            p2 = [g[1][0], g[0][1]]
            p3 = [g[1][0], g[1][1]]
            p4 = [g[0][0], g[1][1]]
            g = [p1, p2, p3, p4]
            # print('ground truth polygon dimension is 4, its type:' + str(type(
            # g)))
        else:
            print('ground truth polygon dimension is incompatible')
            return 0
    if len(p) != 4:
        print('predicted polygon demension is incompatible')
        return 0

    g = Polygon(np.array(g[:4]).reshape((4, 2)))
    p = Polygon(np.array(p[:4]).reshape((4, 2)))
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if g.area == 0:
        return 0
    else:
        # return inter/union
        return inter / g.area


def calculate_correction_number_by_json(gt_shapes, pd_shapes, IoU_threshold):
    correct_text_box = 0
    correct_arrow_box = 0
    correct_inhibit_box = 0
    correct_gene_recognition = 0
    incorrect_gene_recognition = 0
    correct_arrow_recognition = 0
    correct_inhibit_recognition = 0
    correct_neg_gene_recognition = 0
    idx_list = np.arange(len(pd_shapes)).tolist()
    all = {}

    for gt_idx in range(len(gt_shapes)):
        # find a best matching predict box for current gt_shape
        matched_IoU = {}
        for pd_idx in range(0, len(pd_shapes)):
            current_IoU = intersection(gt_shapes[gt_idx]['points'],
                                       pd_shapes[pd_idx]['points'])

            # pick un-matched pd_shape with higher IoU as candidates
            if current_IoU >= IoU_threshold and \
                    pd_shapes[pd_idx]['alias'] != 'matched':
                matched_IoU.update({pd_idx: current_IoU})
                all.update({pd_idx: current_IoU})
                if pd_idx in idx_list:
                    idx_list.remove(pd_idx)

        # no prediction can match current gt_shape:
        if len(matched_IoU) == 0:
            del matched_IoU
            continue

        # sort matched_IoU
        candidates = sorted(matched_IoU.items(), key=lambda d: d[1], reverse=True)
        del matched_IoU

        for candidate in candidates:
            candidate_idx = candidate[0]
            pd_category, pd_context = pd_shapes[candidate_idx]['label'].split(':', 1)
            gt_category, gt_context = gt_shapes[gt_idx]['label'].split(':', 1)
            pd_category = pd_category.strip()
            pd_context = pd_context.strip()
            gt_category = gt_category.strip()
            gt_context = gt_context.strip()

            if pd_category == gt_category or (pd_category == 'gene' and
                                              (gt_category == 'compound' or gt_category == 'location'
                                               or gt_category == 'ref_function' or gt_category == 'Title'
                                               or gt_category == 'other')):

                # match category successfully
                gt_shapes[gt_idx]['alias'] = 'matched'
                pd_shapes[candidate_idx]['alias'] = 'matched'
                if gt_category == 'activate':
                    correct_arrow_box = correct_arrow_box + 1
                    if pd_context.upper() == gt_context.upper():
                        correct_arrow_recognition = correct_arrow_recognition + 1
                elif gt_category == 'inhibit':
                    correct_inhibit_box = correct_inhibit_box + 1
                    if pd_context.upper() == gt_context.upper():
                        correct_inhibit_recognition = \
                            correct_inhibit_recognition + 1
                elif gt_category == 'gene':
                    correct_text_box = correct_text_box + 1
                    if pd_context.upper() == gt_context.upper():
                        correct_gene_recognition = correct_gene_recognition + 1
                    else:
                        incorrect_gene_recognition = incorrect_gene_recognition + 1
                break

        # all predicted objects can not target at this gt_shape
        gt_shapes[gt_idx]['alias'] = 'matched'

    for idx in idx_list:
        if pd_shapes[idx]['label'].split(':', 1)[1] == "*\t*\t*\t*\t*" and pd_shapes[idx]['alias'] != 'matched':
            correct_neg_gene_recognition += 1

    return correct_text_box, correct_arrow_box, correct_inhibit_box, \
        correct_gene_recognition, correct_arrow_recognition, correct_inhibit_recognition, \
        correct_neg_gene_recognition


def calculate_gene_match_metrics(gt_list, pd_list):
    pos_correct = 0
    pos_error = 0
    num_gt = len(gt_list)
    num_pd = len(pd_list)
    if len(pd_list) == 0 and len(gt_list) == 0:
        precision = 1
        recall = 1
    elif len(pd_list) == 0 and len(gt_list) != 0:
        precision = 0
        recall = 0
    elif len(pd_list) != 0 and len(gt_list) == 0:
        precision = 1
        recall = 0
    else:
        for gt_gene in gt_list:
            for predict_gene in pd_list:
                if gt_gene == predict_gene or \
                        gt_gene in predict_gene.split('\t'):
                    pos_correct = pos_correct + 1

        # calculate the metrics
        precision = float(pos_correct) / num_pd
        recall = float(pos_correct) / num_gt
    return precision, recall, pos_correct, num_pd, num_gt


def filter_interaction_without_gene(relation_list, gene_name_gt):
    relation_list_with_gene = []
    for relation in relation_list:
        gene1, gene2 = relation.split(':')[1].split('|')
        if gene1 in gene_name_gt and \
                gene2 in gene_name_gt:
            relation_list_with_gene.append(relation)
    del relation_list
    return relation_list_with_gene


def calculate_all_metrics_by_json(detection_threshold):

    # declaim some variances for computing final assessment
    pos_genebox_correct_detect_total = 0
    num_genebox_pd_detect_total = cfg.epsilon
    num_genebox_gt_detect_total = cfg.epsilon
    pos_correct_gene_total = 0
    neg_correct_gene_total = 0
    num_pd_gene_total = cfg.epsilon
    num_gt_gene_total = cfg.epsilon
    pos_arrowbox_correct_detect_total = 0
    num_arrowbox_pd_detect_total = cfg.epsilon
    num_arrowbox_gt_detect_total = cfg.epsilon
    pos_inhibitbox_correct_detect_total = 0
    num_inhibitbox_pd_detect_total = cfg.epsilon
    num_inhibitbox_gt_detect_total = cfg.epsilon
    pos_activate_correct_total = 0
    pos_inhibit_correct_total = 0

    # load ground truth labels
    with open(os.path.join(cfg.predict_folder, 'evaluate_' + str(int(detection_threshold * 100)) + ".txt"),
              'w') as result_fp:
        result_fp.write(
            str.format(
                '%s \t %s \t %s \t %s \t %s \t %s \t %s \t %s \t %s \t %s \t %s \t %s \t %s \t %s \n')
            % ('imagename', 'genebox_recall', 'genebox_precision',
               'arrowbox_recall', 'arrowbox_precision', 'inhibitbox_recall',
               'inhibitbox_precision', 'gene_recall', 'gene_precision', 'gene_ocr_accuracy',
               'arrow_recall', 'arrow_precision', 'inhibit_recall',
               'inhibit_precision'))

    for image_file in os.listdir(cfg.image_folder):
        # label = LabelFile(os.path.join(path + r'\gt', image_file[:-4] + '.json'))
        image_name, image_ext = os.path.splitext(image_file)
        if image_ext != ".json":
            continue
        print(image_file)
        try:
            label_gt = LabelFile(
                os.path.join(cfg.ground_truth_folder, image_name + '.json'))
        except():
            print('we do not have ground truth for this input yet')
            continue

        try:
            label_pd = LabelFile(os.path.join(cfg.predict_folder, image_name + '.json'))
        except():
            print('we cannot open current predicted json file')
            continue

        pos_genebox_correct_detect, pos_arrowbox_correct_detect, \
            pos_inhibitbox_correct_detect, pos_correct_gene, pos_correct_arrow, \
            pos_correct_inhibit, neg_correct_gene = calculate_correction_number_by_json(label_gt.shapes,
                                                                                        label_pd.shapes,
                                                                                        detection_threshold)

        # these variants might be divided, so they cannot be zero
        current_gt_text_num = label_gt.text_num + cfg.epsilon
        current_gt_arrow_num = label_gt.arrow_num + cfg.epsilon
        current_gt_inhibit_num = label_gt.inhibit_num + cfg.epsilon
        current_gt_gene_num = len(label_gt.get_all_genes()) + cfg.epsilon

        current_pd_text_num = label_pd.text_num + cfg.epsilon
        current_pd_arrow_num = label_pd.arrow_num + cfg.epsilon
        current_pd_inhibit_num = label_pd.inhibit_num + cfg.epsilon
        current_pd_gene_num = len(label_pd.get_all_genes()) + cfg.epsilon

        # calculate the metrics
        recall_genebox_detect = pos_genebox_correct_detect / current_gt_text_num
        precision_genebox_detect = pos_genebox_correct_detect / current_pd_text_num
        recall_arrowbox_detect = pos_arrowbox_correct_detect / current_gt_arrow_num
        precision_arrowbox_detect = pos_arrowbox_correct_detect / current_pd_arrow_num
        recall_inhibitbox_detect = pos_inhibitbox_correct_detect / current_gt_inhibit_num
        precision_inhibitbox_detect = pos_inhibitbox_correct_detect / current_pd_inhibit_num
        recall_gene = pos_correct_gene / current_gt_gene_num
        precision_gene = pos_correct_gene / current_pd_gene_num
        accuracy_gene = (pos_correct_gene + neg_correct_gene) / current_pd_gene_num
        recall_arrow = pos_correct_arrow / current_gt_arrow_num
        precision_arrow = pos_correct_arrow / current_pd_arrow_num
        recall_inhibit = pos_correct_inhibit / current_gt_inhibit_num
        precision_inhibit = pos_correct_inhibit / current_pd_inhibit_num

        # save assessment of current prediction to file
        with open(os.path.join(cfg.predict_folder, 'evaluate_' + str(int(detection_threshold * 100)) + ".txt"),
                  'a') as result_fp:
            result_fp.write(
                str.format(
                    '%s \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f\t %f \t %f \n')
                % (label_pd.filename, recall_genebox_detect, precision_genebox_detect,
                   recall_arrowbox_detect, precision_arrowbox_detect,
                   recall_inhibitbox_detect, precision_inhibitbox_detect,
                   recall_gene, precision_gene, accuracy_gene, recall_arrow, precision_arrow,
                   recall_inhibit, precision_inhibit))

        # accumulate
        # gene box
        pos_genebox_correct_detect_total += pos_genebox_correct_detect
        num_genebox_pd_detect_total += current_pd_text_num
        num_genebox_gt_detect_total += current_gt_text_num

        # gene names
        pos_correct_gene_total += pos_correct_gene
        neg_correct_gene_total += neg_correct_gene
        num_pd_gene_total += current_pd_gene_num
        num_gt_gene_total += current_gt_gene_num

        # arrow
        pos_arrowbox_correct_detect_total += pos_arrowbox_correct_detect
        num_arrowbox_pd_detect_total += current_pd_arrow_num
        num_arrowbox_gt_detect_total += current_gt_arrow_num

        # inhibit
        pos_inhibitbox_correct_detect_total += pos_inhibitbox_correct_detect
        num_inhibitbox_pd_detect_total += current_pd_inhibit_num
        num_inhibitbox_gt_detect_total += current_gt_inhibit_num

        # activate recognition
        pos_activate_correct_total += pos_correct_arrow

        # inhibit recognition
        pos_inhibit_correct_total += pos_correct_inhibit

        del label_gt, label_pd, current_gt_text_num, current_gt_arrow_num, \
            current_gt_inhibit_num, current_gt_gene_num, current_pd_text_num, \
            current_pd_arrow_num, current_pd_inhibit_num, current_pd_gene_num, \
            pos_genebox_correct_detect, pos_arrowbox_correct_detect, \
            pos_inhibitbox_correct_detect, pos_correct_gene, pos_correct_arrow, \
            pos_correct_inhibit

    # calculate the overall metrics
    overall_genebox_precision_detect = pos_genebox_correct_detect_total / num_genebox_pd_detect_total
    overall_genebox_recall_detect = pos_genebox_correct_detect_total / num_genebox_gt_detect_total
    overall_precision_gene = pos_correct_gene_total / num_pd_gene_total
    overall_recall_gene = pos_correct_gene_total / num_gt_gene_total
    overall_accuracy_gene = (pos_correct_gene_total + neg_correct_gene_total) / num_pd_gene_total
    overall_arrowbox_precision_detect = pos_arrowbox_correct_detect_total / num_arrowbox_pd_detect_total
    overall_arrowbox_recall_detect = pos_arrowbox_correct_detect_total / num_arrowbox_gt_detect_total
    overall_inhibitbox_precision_detect = pos_inhibitbox_correct_detect_total / num_inhibitbox_pd_detect_total
    overall_inhibitbox_recall_detect = pos_inhibitbox_correct_detect_total / num_inhibitbox_gt_detect_total
    overall_precision_arrow = pos_activate_correct_total / num_arrowbox_pd_detect_total
    overall_recall_arrow = pos_activate_correct_total / num_arrowbox_gt_detect_total
    overall_precision_inhibit = pos_inhibit_correct_total / num_inhibitbox_pd_detect_total
    overall_recall_inhibit = pos_inhibit_correct_total / num_inhibitbox_gt_detect_total

    with open(os.path.join(cfg.predict_folder, 'evaluate_' + str(int(detection_threshold * 100)) + ".txt"),
              'a') as result_fp:
        result_fp.write(
            str.format(
                '%s \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f\t %f \t %f \n')
            % ('total evaluation',
               overall_genebox_recall_detect,
               overall_genebox_precision_detect,
               overall_arrowbox_recall_detect,
               overall_arrowbox_precision_detect,
               overall_inhibitbox_recall_detect,
               overall_inhibitbox_precision_detect,
               overall_recall_gene,
               overall_precision_gene,
               overall_accuracy_gene,
               overall_recall_arrow,
               overall_precision_arrow,
               overall_recall_inhibit,
               overall_precision_inhibit))


def draw_curve(gene_list, arrow_list, inhibit_list):
    mean = np.mean([gene_list, arrow_list, inhibit_list], axis=0)
    genex = gene_list[:, 0]
    geney = gene_list[:, 1]
    arrowx = arrow_list[:, 0]
    arrowy = arrow_list[:, 1]
    inhibitx = inhibit_list[:, 0]
    inhibity = inhibit_list[:, 1]
    meanx = mean[:, 0]
    meany = mean[:, 1]
    plt.figure()
    plot1, = plt.plot(genex, geney, color="r", linewidth=1, marker="o")
    plot2, = plt.plot(arrowx, arrowy, color="g", linewidth=1, marker="+")
    plot3, = plt.plot(inhibitx, inhibity, color="b", linewidth=1, marker="x")
    plot4, = plt.plot(meanx, meany, color="y", linewidth=1, marker="*")
    plt.xlim([0, 1])
    # plt.xticks(np.linspace(0.2, 0.9, 15))
    plt.xticks(np.linspace(0, 1, 21))
    plt.xlabel("Recall", fontsize="x-large")

    # set Y axis
    plt.ylim([0, 1.05])
    # plt.yticks(np.linspace(0.2, 1.0, 17))
    plt.yticks(np.linspace(0, 1.0, 21))
    plt.ylabel("Precision", fontsize="x-large")

    # set figure information
    plt.title("Precision --- Recall", fontsize="x-large")
    plt.legend([plot1, plot2, plot3, plot4], ("Gene", "Arrow", "Inhibit", "Mean"), loc="lower right", numpoints=1)
    plt.grid(True)

    # draw the chart
    plt.show()


if __name__ == '__main__':
    # ground_truth_folder = r'C:\Users\LSC-110\Desktop\ground_truth'
    # img_folder = r'C:\Users\LSC-110\Desktop\Images'
    # predict_folder = r'C:\Users\LSC-110\Desktop\results'
    # ground_truth_folder = r'C:\Users\hefe\Desktop\history\ground_truth'

    if cfg.ground_truth_folder and os.path.exists(cfg.ground_truth_folder):  # evaluate
        for value in cfg.detection_IoU_thresholds:
            calculate_all_metrics_by_json(value)

# end of file
