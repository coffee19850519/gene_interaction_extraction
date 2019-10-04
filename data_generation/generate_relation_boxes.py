import cv2, os
from label_file import LabelFile
import numpy as np
from data_generation.detect_arrow_processing import find_all_inhibits,  find_all_arrows, pair_gene

def generate_relation_box_per_gene_pair(entity1_box, entity2_box, relation_box, img_width, img_height, offset):

    entity1_box = np.array(entity1_box, np.int32).reshape((-1, 2))
    entity2_box = np.array(entity2_box, np.int32).reshape((-1, 2))
    relation_box = np.array(relation_box, np.int32).reshape((-1, 2))

    # handle if entity boxes are rectangles and NOT polygons
    if entity1_box.shape[0] == 4:
        entity1_box = cv2.boxPoints(cv2.minAreaRect(entity1_box))
        entity1_box = np.int32(entity1_box)
    if entity2_box.shape[0] == 4:
        entity2_box = cv2.boxPoints(cv2.minAreaRect(entity2_box))
        entity2_box = np.int32(entity2_box)
    if relation_box.shape[0] ==4:
        relation_box = cv2.boxPoints(cv2.minAreaRect(relation_box))
        relation_box = np.int32(relation_box)


    # initialize starting dimensions of sub-image
    left_top_x = int(min(min(entity1_box[:, 0]), min(entity2_box[:, 0]), min(relation_box[:, 0])))
    left_top_y = int(min(min(entity1_box[:, 1]), min(entity2_box[:, 1]), min(relation_box[:, 1])))
    right_bottom_x = int(max(max(entity1_box[:, 0]), max(entity2_box[:, 0]), max(relation_box[:, 0])))
    right_bottom_y = int(max(max(entity1_box[:, 1]), max(entity2_box[:, 1]), max(relation_box[:, 1])))

    # add some padding to sub-image dimensions
    left_top_x = left_top_x - offset
    left_top_y = left_top_y - offset

    # if dimensions with offset are out of range (negative), then set to zero
    if (left_top_x < 0):
        left_top_x = 0
    if (left_top_y < 0):
        left_top_y = 0

    # if dimensions with offset are out of range (positive), then set to max edge of original image
    right_bottom_x = right_bottom_x + offset
    right_bottom_y = right_bottom_y + offset
    if right_bottom_x > img_width:
        right_bottom_x = img_width
    if right_bottom_y > img_height:
        right_bottom_y = img_height

    # check for bad created dimensions
    if (left_top_y == right_bottom_y) or (left_top_x == right_bottom_x):
            #or not re_check_generative_box():
        raise Exception('illegal relation ')
    else:
        return left_top_y, left_top_x, right_bottom_y, right_bottom_x


def generate_relation_boxes(image_path, label_path, new_label_path):
    current_label_file = LabelFile(label_path)
    activator_list, activator_neighbor_list, receptor_list, receptor_neighbor_list, text_shapes, arrow_boxes = \
        find_all_arrows(image_path, current_label_file)

    arrow_relationship_pairs, arrow_descriptions, arrow_relationship_boxes = pair_gene(activator_list,
                                                                                       activator_neighbor_list,
                                                                                       receptor_list,
                                                                                       receptor_neighbor_list,
                                                                                       text_shapes,
                                                                                       arrow_boxes,
                                                                                       image_path)

    for arrow_description, arrow_relationship_box in zip(arrow_descriptions, arrow_relationship_boxes):
        left_top_y, left_top_x, right_bottom_y, right_bottom_x = \
            generate_relation_box_per_gene_pair(arrow_relationship_box['entity1_bounding_box'],
                                                arrow_relationship_box['entity2_bounding_box'],
                                                arrow_relationship_box['relationship_bounding_box'],
                                                current_label_file.imageWidth,
                                                current_label_file.imageHeight,
                                                offset= 5)
        tempDict = {}
        tempDict['label'] = arrow_description
        tempDict['line_color'] = [0, 255, 0, 255]
        tempDict['fill_color'] = None
        tempDict['points'] = [[left_top_x, left_top_y],[right_bottom_x, right_bottom_y]]
        tempDict['shape_type'] = 'rectangle'
        tempDict['alias'] = 'name'
        current_label_file.shapes.append(tempDict)

    activator_list, activator_neighbor_list, receptor_list, receptor_neighbor_list, text_shapes, inhibit_boxes = \
    find_all_inhibits(image_path, current_label_file)
    inhibit_relationship_pairs, inhibit_descriptions, inhibit_relationship_boxes = pair_gene(activator_list,
                                                                                             activator_neighbor_list,
                                                                                             receptor_list,
                                                                                             receptor_neighbor_list,
                                                                                             text_shapes, inhibit_boxes,
                                                                                             image_path)
    for inhibit_description, inhibit_relationship_box in zip(inhibit_descriptions, inhibit_relationship_boxes):
        left_top_y, left_top_x, right_bottom_y, right_bottom_x = generate_relation_box_per_gene_pair(
                                                inhibit_relationship_box['entity1_bounding_box'],
                                                inhibit_relationship_box['entity2_bounding_box'],
                                                inhibit_relationship_box['relationship_bounding_box'],
                                                current_label_file.imageWidth,
                                                current_label_file.imageHeight,
                                                offset=5)
        tempDict = {}
        tempDict['label'] = inhibit_description
        tempDict['line_color'] = [0, 255, 0, 255]
        tempDict['fill_color'] = None
        tempDict['points'] = [[left_top_x, left_top_y], [right_bottom_x, right_bottom_y]]
        tempDict['shape_type'] = 'rectangle'
        tempDict['alias'] = 'name'
        current_label_file.shapes.append(tempDict)

    current_label_file.save(new_label_path,
                            current_label_file.shapes,
                            current_label_file.imagePath,
                            current_label_file.imageHeight,
                            current_label_file.imageWidth)


if __name__ == '__main__':
    img_fold = r'C:\Users\coffe\Desktop\test\label_generation'
    #sim_json_fold = r'C:\Users\LSC-110\Desktop\cxtest\test_sim'
    for image_file in os.listdir(img_fold):
        image_name, image_ext = os.path.splitext(image_file)
        if image_ext == '.json':
            continue
        else:
            generate_relation_boxes(os.path.join(img_fold, image_file),
                                    os.path.join(img_fold,image_name + '.json'),
                                    os.path.join(img_fold,image_name + '_new.json'))