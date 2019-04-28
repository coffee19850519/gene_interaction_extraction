import cv2,os,random,math
import numpy as np
import pandas as pd

from label_file import LabelFile


def detect_all_contours(img, img_file):

    binary_image = cv2.adaptiveThreshold(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),
                                         255,
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 11, 10)

    binary_image_INV = cv2.bitwise_not(binary_image)

    # cv2.imwrite(img_file[:-4] + '_binaryINV.png', binary_image_INV)

    # here is find_contour version
    contours, hierarchy = cv2.findContours(binary_image_INV, cv2.RETR_CCOMP,
                                          cv2.CHAIN_APPROX_NONE)

    #here is connected_analysis version
    # image, contours, stats,_ = cv2.connectedComponentsWithStats(
    #     binary_image_INV, connectivity= 8)

    #draw each component in results
    # for i in range(contours):
    #
    #     display_image = img.copy()

        #display_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2GRAY)

        #generate mask for ith component
        #mask = np.equal(contours, i).astype(np.int32)

        #draw it on a new image
        # display_image = cv2.bitwise_and(display_image, mask)
        # masked_image = cv2.multiply (display_image, mask)
        # display_image = img[mask]

        #draw its bounding box
        # cv2.rectangle(masked_image, (stats[i, cv2.CC_STAT_LEFT], stats[i,
        #     cv2.CC_STAT_TOP]), (stats[i, cv2.CC_STAT_LEFT] + stats[i,
        #     cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_TOP] + stats[i,
        #     cv2.CC_STAT_HEIGHT]),(255, 0, 0),thickness=2)

        #save the image
        # cv2.imwrite(img_file[:-4] + '_contours' + str(i) + '.png',
        #             labeled_img)
        #
        # #delete the mask, temp_image
        # del display_image,mask, masked_image

    for i in range(len(contours)):
        # if i != 105:
        #      continue

        markedImage = img.copy()
        # cv2.floodFill(markedImage,,contours[0],(0, 255, 0),
        #               cv2.FLOODFILL_FIXED_RANGE )
        # cv2.drawContours(markedImage, contours, i, (0, 255, 0), 2)
        # cv2.imwrite(img_file[:-4] + '_contours'+ str(i) +'.png', markedImage)
    del markedImage

    del binary_image
    return contours, hierarchy[0], binary_image_INV


def match_head_and_arrow_contour(img, head_box, contours, hierarchy):#,
  # detected_list):
    # arrow_head = Polygon(np.array(head_box, np.int32).reshape((-1,2)))
    #detected_list = list(detected_list)
    interaection_area = []
    for i in range(len(contours)) :
        # if i in detected_list:
        #     continue
        interaection_area.append(calculate_interaction_area(img, head_box,
                                                            contours[i]))
    if len(interaection_area) != 0:

        idx = interaection_area.index(max(interaection_area))
        del interaection_area
        return [idx, hierarchy[idx][2],hierarchy[idx][3]]

    else:
        del interaection_area
        #no matching arrow
        return []

def  calculate_interaction_area(img, head_box, contour):
    # draw the contour on a white image
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, color= 1, thickness= 1)
    #cv2.fillPoly(mask, contour, 1)
    # calculate the sum of pixels in head box
    head_box = np.array(head_box, np.int32).reshape((-1, 2))

    start_x = int(min(head_box[:, 0]))

    start_y = int(min(head_box[:, 1]))

    end_x = int(max(head_box[:, 0]))

    end_y = int(max(head_box[:, 1]))

    area = np.sum(mask[start_y : end_y, start_x: end_x])
    del mask
    return area

def erase_all_text(img, img_file,  text_shapes):
    for text_shape in text_shapes:
        text_box = text_shape['points']
        text_box = np.array(text_box, np.int32).reshape((-1, 2))

        start_x = int(min(text_box[:, 0]))

        start_y = int(min(text_box[:, 1]))

        end_x = int(max(text_box[:, 0]))

        end_y = int(max(text_box[:, 1]))

        cv2.rectangle(img, (start_x,start_y), (end_x,end_y), (255,255,255),
                                              thickness = -1)
    #cv2.imwrite(img_file[:-4]+ '_erase.png', img)

def get_angle_distance(point1, point2):
    dis = get_distance_between_two_points(point1, point2)
    if (point1[0] - point2[0] != 0):
        return abs(float(math.atan2(float(point1[1] - point2[
            1]), float(point1[0] - point2[0])))), dis
    else:
        return 0, dis


def slope(point1, point2):
    if point1[0] !=  point2[0]:
        return  math.atan2(float(point1[1] - point2[1]), float(point1[0] -
                                                           point2[0]))
    else:
        return None


def find_nearest_point(point, candidates):
    dis = []
    for candidate in candidates:
        distance = get_distance_between_two_points(point,candidate)
        dis.append(distance)
    point_idx = dis.index(min(dis))
    del dis
    return candidates[point_idx],point_idx


def get_distance_between_two_points(point1, point2):

    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
#2019/3/7
def dist_center(point1, point2):
  QP = [point1[0] - point2[0], point1[1] - point2[1]]
  return np.sqrt(QP[0] ** 2 + QP[1] ** 2)
#2019/3/7
def find_nearest_text(arrow_region, text_regions,sp):
  dists = []
  text_center=()
  heigth=sp[0]
  width=sp[1]
  diagonal=np.sqrt(sp[0]**2+sp[1]**2)
  for text_r in text_regions:
    text_center=((text_r[0][0]+text_r[2][0])/2,(text_r[0][1]+text_r[2][1])/2)
    dist = dist_center(arrow_region, text_center)
    dists.append(dist)
  if len(dists)!=0:
      nearest_index = np.argmin(dists)
      if dists[nearest_index]/diagonal<0.1:
          nearest_index=nearest_index
      else:
          nearest_index=None
      del dists
  else:
      nearest_index=None
  return nearest_index

def find_vertex_for_detected_arrow_by_distance(img, candidates, head_box):
  aggregate_img = img.copy()
  out_head = []
  in_head = []
  cv2.polylines(aggregate_img, [head_box], True, (0, 255, 0), thickness=2)
  for candidate in candidates:
    candidate = candidate[0]
    if cv2.pointPolygonTest(head_box, tuple(candidate),
                            measureDist=False) != -1:
      in_head.append(candidate)
    else:
      out_head.append(candidate)

  # determine receptor
  if len(in_head) != 0:
      receptor_point = np.mean(in_head, axis=0, dtype=np.int32)
  else:
      receptor_point = np.mean(head_box, axis=0, dtype=np.int32)
  if len(out_head) < 1:
      # activate_point= np.array([0,0],dtype = np.int32);
      #under this circumstance, the line is a dash line
      activator_point = None
      activator_neighbor = None
      #activate_slope = None
      receptor_point = None
      receptor_neighbor = None
      #receptor_slope = None
  elif len(out_head) == 1:
      #connected line is straight line
      activator_point = out_head[0]
      # activate_slope = slope(activate_point, receptor_point)
      # receptor_slope = activate_slope
      activator_neighbor = None
      receptor_neighbor = None

  elif len(out_head) == 2:
      # find first connected key-point for calculating slope
      first_point, first_index = find_nearest_point(receptor_point,out_head)
      #receptor_slope = slope(receptor_point, first_point)
      out_head.pop(first_index)
      activator_point = out_head[0]
      #activate_slope = slope(first_point, activate_point)
      activator_neighbor = first_point
      receptor_neighbor = first_point
  else:
      # find first connected key-point for calculating slope
      first_point, first_index  = find_nearest_point(receptor_point,                                                   out_head)
      #receptor_slope = slope(receptor_point, first_point)
      out_head.pop(first_index)
      receptor_neighbor = first_point
      # from receptor point to find the nearest point until end point
      activator_point = first_point.copy()

      while (len(out_head) > 1):
        activator_point, activate_index = find_nearest_point(activator_point,
                                                            out_head)
        out_head.pop(activate_index)
      # the last point is the farest point to receptor point
      first_point = activator_point
      activator_point = out_head[0]
      #activate_slope = slope(first_point, activate_point)
      activator_neighbor = first_point
  del in_head, out_head
  return activator_point,activator_neighbor, receptor_point, activator_point


def mean_neighbor_pixels(img, center_point, neighbor_size):
    mean_pixel = np.mean(img[center_point[1] - neighbor_size: center_point[1] +
                 neighbor_size, center_point[0] - neighbor_size:
                 center_point[0] + neighbor_size])
    return mean_pixel


def scoring_all_candidates(img, binary_img, startor, candidates):
    #put startor information into assessed path
    #detected_path = [startor]
    selected_list = [{
      'x': startor[0],
      'y': startor[1],
      'distance': 0,
      'delta_distance':0,
      'grad_distance' : 0,
      'angle': 0,
      'delta_angle':0,
      'grad_angle':0,
      'mean_pixel': mean_neighbor_pixels(binary_img, startor, 3),
      'delta_pixel':0,
      'grad_pixel':0,
      'score':0}]

    candidate_num = len(candidates)
    candidate_list = candidates.copy()

    receptor_neighbor, closest_idx = find_nearest_point(startor, candidates)
    #startor_slope = slope(closest_point, startor)


    while(len(selected_list) < candidate_num + 1):
        current_point = (selected_list[len(selected_list) - 1]['x'],
                         selected_list[len(selected_list) - 1]['y'])
        selected_point, selected_idx = find_nearest_point(current_point,
                                                          candidate_list)

        angle, distance = get_angle_distance(selected_point, current_point)
        neighbor_pixel = mean_neighbor_pixels(binary_img, selected_point, 3)

        selected_list.append({
          'x' : selected_point[0],
          'y' : selected_point[1],
          'distance' : distance,
          'delta_distance' : 0,
          'grad_distance': 0,
          'angle' : angle,
          'delta_angle' : 0,
          'grad_angle': 0,
          'mean_pixel' : neighbor_pixel,
          'delta_pixel': 0,
          'grad_pixel': 0,
          'score' : 0})
        candidate_list.pop(selected_idx)
        del current_point,selected_point, selected_idx,angle, distance, \
          neighbor_pixel

    candidate_table = pd.DataFrame(selected_list)
    del selected_list,candidate_list

    #calculate the delta information and gradient information for scoring
    for idx in range(1, len(candidate_table)):
        candidate_table.loc[idx,['delta_distance']] = candidate_table.loc[idx][
          'distance'] - candidate_table.loc[idx - 1]['distance']
        candidate_table.loc[idx,['delta_angle']] = candidate_table.loc[idx][
          'angle'] - candidate_table.loc[idx - 1]['angle']
        candidate_table.loc[idx,['delta_pixel']] = candidate_table.loc[idx][
           'mean_pixel'] - candidate_table.loc[idx - 1]['mean_pixel']
        # calculate the gradient of distance/angle/neighbor pixels to current
        #  point
        candidate_table.loc[idx,['grad_distance']] = np.gradient(
            candidate_table[:idx+1]['delta_distance'].values)[idx]
        candidate_table.loc[idx,['grad_angle']] = np.abs(np.gradient(
            candidate_table[:idx+1]['delta_angle'].values)[idx])
        candidate_table.loc[idx,['grad_pixel']] = np.gradient(
            candidate_table[:idx+1]['delta_pixel'].values)[idx]

        mean_grad_distance = np.mean(np.abs(candidate_table[:idx][
                                        'grad_distance'].values))
        std_grad_distance = np.std(np.abs(candidate_table[:idx][
                                      'grad_distance'].values))
        if  candidate_table.loc[idx]['grad_distance'] > 0 and\
            candidate_table.loc[idx]['grad_distance'] > mean_grad_distance + 2 * \
            std_grad_distance:
            candidate_table.loc[idx - 1, ['score']] = \
                candidate_table.loc[idx - 1]['score'] + 1

        #caclulate local score for each candidate
        # mean_delta_angle = np.mean(candidate_table[:idx]['delta_angle'].values)
        # std_delta_angle = np.std(candidate_table[:idx]['delta_angle'].values)
        # if candidate_table.loc[idx]['delta_angle'] < mean_delta_angle - 2 \
        #     *std_delta_angle:
        #     candidate_table.loc[idx,['score']] = candidate_table.loc[idx][
        #         'score'] + 1

        mean_grad_angle = np.mean(candidate_table[:idx]['grad_angle'].values)
        std_grad_angle = np.std(candidate_table[:idx]['grad_angle'].values)
        if candidate_table.loc[idx]['grad_angle'] > mean_grad_angle + 2 * \
            std_grad_angle:
            #
            candidate_table.loc[idx-1, ['score']] = candidate_table.loc[idx-1][
                                                  'score'] + 1

        # mean_delta_pixel = np.mean(candidate_table[:idx]['delta_pixel'].values)
        # std_delta_pixel = np.std(candidate_table[:idx]['delta_pixel'].values)
        # if candidate_table.loc[idx]['delta_pixel'] < mean_delta_pixel - 2 * \
        #     std_delta_pixel:
        #     candidate_table.loc[idx, ['score']] = candidate_table.loc[idx][
        #                                           'score'] + 0

        mean_grad_pixel = np.mean(np.abs(candidate_table[:idx]['grad_pixel'].values))
        std_grad_pixel = np.std(np.abs(candidate_table[:idx]['grad_pixel'].values))
        if  candidate_table.loc[idx]['grad_pixel'] < 0 and \
            np.abs(candidate_table.loc[idx]['grad_pixel']) > mean_grad_pixel + 2 * \
            std_grad_pixel:
            candidate_table.loc[idx-1, ['score']] = candidate_table.loc[idx-1][
                                                  'score'] + 1
        #considering the case at intersection, that means delta_distance is
        # small but delta_pixel is positive large
        if  candidate_table.loc[idx]['grad_pixel'] < 0 and \
            np.abs(candidate_table.loc[idx]['grad_pixel']) > mean_grad_pixel \
            + 2 * std_grad_pixel and (candidate_table.loc[idx]['grad_angle'] <
            mean_grad_angle or candidate_table.loc[idx]['grad_distance'] <  mean_grad_distance):
            candidate_table.loc[idx-1, ['score']] = candidate_table.loc[idx-1][
                                                  'score'] - 2

    #calculate global score to modify each candidate's score
    # mean_delta_angle = np.mean(candidate_table['delta_angle'].values)
    # std_delta_angle = np.std(candidate_table['delta_angle'].values)

    # mean_grad_angle = np.mean(candidate_table['grad_angle'].values)
    # std_grad_angle = np.std(candidate_table['grad_angle'].values)
    #
    #
    # mean_delta_pixel = np.mean(candidate_table['delta_pixel'].values)
    # std_delta_pixel = np.std(candidate_table['delta_pixel'].values)
    #
    #
    # mean_grad_pixel = np.mean(np.abs(candidate_table['grad_pixel'].values))
    # std_grad_pixel = np.std(np.abs(candidate_table['grad_pixel'].values))
    #
    # idx = 0
    # for idx in range(len(candidate_table)):
    #     # if candidate_table.loc[idx]['delta_angle'] < mean_delta_angle - 2 \
    #     #     * std_delta_angle:
    #     #   candidate_table.loc[idx, ['score']] = candidate_table.loc[idx][
    #     #                                           'score'] + 1
    #
    #     if candidate_table.loc[idx]['grad_angle'] > mean_grad_angle + 2 * \
    #         std_grad_angle:
    #       candidate_table.loc[idx, ['score']] = candidate_table.loc[idx][
    #                                               'score'] + 1
    #
    #     # if candidate_table.loc[idx]['delta_pixel'] < mean_delta_pixel - 2 * \
    #     #     std_delta_pixel:
    #     #   candidate_table.loc[idx, ['score']] = candidate_table.loc[idx][
    #     #                                           'score'] + 0
    #
    #     if  candidate_table.loc[idx]['grad_pixel'] < 0 and \
    #         np.abs(candidate_table.loc[idx]['grad_pixel']) > mean_grad_pixel + 2 * \
    #         std_grad_pixel:
    #         candidate_table.loc[idx - 1, ['score']] = \
    #             candidate_table.loc[idx - 1]['score'] + 1

    #pick up the candidates with larger score than mean
    candidate_table.loc[0, ['score']] = 0
    mean_score = np.mean(candidate_table['score'].values)
    #std_score = np.std(candidate_table['score'].values)
    vertex_candidates = candidate_table.query('score >= '+str(
        mean_score))

    # # for outliner judgement
    # _, _, mean_dis, mean_angle, mean_neighbor = np.mean(detected_path, axis=0)
    # _, _, std_dis, std_angle, std_neighbor = np.std(detected_path, axis=0)

    # award for candidates are not with minimal mean_pixel
    # vertex_candidates = vertex_candidates.query('mean_pixel ==' + str(np.max(
    #     vertex_candidates['mean_pixel'].values)))


    vertex_candidates.sort_values(by=['mean_pixel', 'distance'], inplace=True)

    #vertex_candidate_coords = vertex_candidates.head(1)
    idx = 0
    for idx in range(len(vertex_candidates)):
        vertex_candidate_coords = vertex_candidates[['x', 'y']]
        # draw them on image
        vertex_x = int(vertex_candidate_coords['x'].values[idx])
        vertex_y = int(vertex_candidate_coords['y'].values[idx])
        cv2.circle(img, (vertex_x, vertex_y), 3, (0, 0, 255), thickness=-1)

    # cv2.imwrite('contour' + str(cnt_index) + '_key_candidates.png', img)
    del candidate_table,vertex_candidates,img
    chosen_point = vertex_candidate_coords.head(1)[['x', 'y']].values[0]
    activator_neighbor, closest_idx = find_nearest_point(chosen_point,
                                                        candidates)
    #chosen_slope = slope(chosen_point, closest_point)
    return chosen_point,activator_neighbor,receptor_neighbor

def find_vertex_for_detected_arrow(img,binary_img, img_file, candidates, \
                                                 head_box):

    #draw the path from receptor to activator point by point
    aggregate_img = img.copy()

    #declare two set to avoid duplicates
    out_head = []
    in_head = []

    cv2.polylines(aggregate_img,[head_box], True,(0,255,0),thickness= 2)
    for candidate in candidates:
      candidate = candidate[0]
      if cv2.pointPolygonTest(head_box, tuple(candidate),
              measureDist= False) != -1:
        in_head.append(candidate)
      else:
        out_head.append(candidate)

    # determine receptor
    if len(in_head) != 0:
        receptor_point = np.mean(in_head, axis=0, dtype=np.int32)
    else:
        receptor_point = np.mean(head_box, axis=0, dtype=np.int32)

    #trace the path to locate
    #detected_path = []
    detected_path = np.array([receptor_point[0],receptor_point[1], 0, 0,
                              mean_neighbor_pixels(img,receptor_point,3)],
                             np.float).reshape((-1, 5))
    #cv2.circle(aggregate_img, tuple(receptor_point), 2,(255, 0, 0),
    # thickness=-1)

    activator_point, activator_neighbor,receptor_neighbor  = \
      scoring_all_candidates(
        aggregate_img, binary_img, receptor_point, out_head)

    del aggregate_img,detected_path, in_head,out_head

    return activator_point, activator_neighbor, receptor_point, receptor_neighbor

def find_all_arrows( img_file, label_file):
    origin_img = cv2.imread(img_file)
    label = LabelFile(label_file)
    arrow_shapes = label.get_all_shapes_for_category('arrow')
    # cv2.polylines(origin_img,arrow_shapes[0]['points'],True,(255,0,0),
    #               thickness= 2)
    # cv2.imwrite(img_file[:-4]+ 'rect' + '.png', origin_img)
    text_shapes = label.get_all_shapes_for_category('text')
    img = origin_img.copy()
    # locate_img = origin_img.copy()
    erase_all_text(img, img_file, text_shapes)
    candidate_contours, hierarchy, binary_img = detect_all_contours(img, img_file)
    detected_list = []
    activator_list=[]
    receptor_list=[]
    activator_neighbor_list = []
    receptor_neighbor_list = []

    for arrow_shape in arrow_shapes:
        head_box = arrow_shape['points']
        #draw the box of arrow
        # cv2.rectangle(origin_img, tuple(head_box[0]), tuple(head_box[1]), (r,g,b), \
        #               thickness=2)
        cnt_idx = match_head_and_arrow_contour(img, head_box, candidate_contours,hierarchy)
                                               #,detected_list)
        #make a list for aggregating overlapping arrows
        detected_list.append(cnt_idx)
        del head_box

    for arrow_idx in range(len(arrow_shapes)) :
        # if arrow_idx != 10:
        #     continue

        head_box=arrow_shapes[arrow_idx]['points']

        # determine out point
        #normalize 2 point to 4 point format, and then make a closed polygon
        if len(head_box) == 2:
            head_box = [head_box[0], [head_box[0][0], head_box[1][1]], head_box[1],
                        [head_box[1][0], head_box[0][1]], head_box[0]]
        elif len(head_box) == 4:
            head_box = [head_box[0], head_box[1], head_box[2], head_box[3],
                      head_box[0]]
        head_box = np.array(head_box, np.int32).reshape((-1, 2))

        cnt_idx = detected_list[arrow_idx]
        if cnt_idx != []:
          vertex_candidates = cv2.approxPolyDP(candidate_contours[cnt_idx[0]],
                                               epsilon=1, closed=True)
          # cv2.drawContours(locate_img, vertex_candidates, -1, (r, g, b),
          #                  thickness=5)
          if cnt_idx[1] != -1:
              vertex_candidates = np.concatenate((vertex_candidates,
                       cv2.approxPolyDP(candidate_contours[cnt_idx[1]], epsilon=1,
                                                      closed=True)))
          if cnt_idx[2] != -1:
              vertex_candidates = np.concatenate((vertex_candidates,
                        cv2.approxPolyDP(candidate_contours[cnt_idx[2]], epsilon=1,
                                                      closed=True)))

          # determine which strategy should use
          same_num = 0
          for detected_cnt in detected_list:
              if (cnt_idx[0] in detected_cnt) and \
                 (cnt_idx[1] in detected_cnt) and \
                 (cnt_idx[2] in detected_cnt):
                  same_num = same_num + 1

          if same_num > 1:
              #exist overlapping arrows
              activator_point, activator_neighbor, receptor_point, \
              receptor_neighbor = \
                find_vertex_for_detected_arrow(origin_img, binary_img,
                                                                   img_file,
                                                                   vertex_candidates,
                                                                   head_box)
          else:
              # straight forward way
              activator_point, activator_neighbor, receptor_point, receptor_neighbor = \
                find_vertex_for_detected_arrow_by_distance(binary_img,
                                                           vertex_candidates,
                                                           head_box)
          del vertex_candidates

          activator_list.append(activator_point)
          receptor_list.append(receptor_point)
          activator_neighbor_list.append(activator_neighbor)
          receptor_neighbor_list.append(receptor_neighbor)


    # cv2.imwrite(img_file[:-4]+'_detected.png', origin_img)
    del img,origin_img,detected_list
    return activator_list, activator_neighbor_list, receptor_list, \
           receptor_neighbor_list, text_shapes, arrow_shapes

def find_all_inhibits(img_file, label_file):
    origin_img = cv2.imread(img_file)
    label = LabelFile(label_file)
    #arrow_shapes = label.get_all_shapes_for_category('arrow')
    # cv2.polylines(origin_img,arrow_shapes[0]['points'],True,(255,0,0),
    #               thickness= 2)
    # cv2.imwrite(img_file[:-4]+ 'rect' + '.png', origin_img)
    inhibit_shapes = label.get_all_shapes_for_category('nock')
    text_shapes = label.get_all_shapes_for_category('text')
    img = origin_img.copy()
    # locate_img = origin_img.copy()
    erase_all_text(img, img_file, text_shapes)
    candidate_contours, hierarchy, binary_img = detect_all_contours(img, img_file)
    detected_list = []
    activator_list = []
    receptor_list = []
    activator_neighbor_list = []
    receptor_neighbor_list = []

    for inhibit in inhibit_shapes:
        head_box = inhibit['points']
        # draw the box of arrow
        # cv2.rectangle(origin_img, tuple(head_box[0]), tuple(head_box[1]), (r,g,b), \
        #               thickness=2)
        cnt_idx = match_head_and_arrow_contour(img, head_box, candidate_contours, hierarchy)
        # ,detected_list)
        # make a list for aggregating overlapping arrows
        detected_list.append(cnt_idx)
        del head_box

    for inhibit_idx in range(len(inhibit_shapes)):
        # if arrow_idx != 10:
        #     continue

        head_box = inhibit_shapes[inhibit_idx]['points']

        # determine out point
        # normalize 2 point to 4 point format, and then make a closed polygon
        if len(head_box) == 2:
            head_box = [head_box[0], [head_box[0][0], head_box[1][1]], head_box[1],
                        [head_box[1][0], head_box[0][1]], head_box[0]]
        elif len(head_box) == 4:
            head_box = [head_box[0], head_box[1], head_box[2], head_box[3],
                        head_box[0]]
        head_box = np.array(head_box, np.int32).reshape((-1, 2))

        cnt_idx = detected_list[inhibit_idx]
        if cnt_idx != []:
            vertex_candidates = cv2.approxPolyDP(candidate_contours[cnt_idx[0]],
                                                 epsilon=1, closed=True)
            # cv2.drawContours(locate_img, vertex_candidates, -1, (r, g, b),
            #                  thickness=5)
            if cnt_idx[1] != -1:
                vertex_candidates = np.concatenate((vertex_candidates,
                                                    cv2.approxPolyDP(candidate_contours[cnt_idx[1]], epsilon=1,
                                                                     closed=True)))
            if cnt_idx[2] != -1:
                vertex_candidates = np.concatenate((vertex_candidates,
                                                    cv2.approxPolyDP(candidate_contours[cnt_idx[2]], epsilon=1,
                                                                     closed=True)))

            # determine which strategy should use
            same_num = 0
            for detected_cnt in detected_list:
                if (cnt_idx[0] in detected_cnt) and \
                        (cnt_idx[1] in detected_cnt) and \
                        (cnt_idx[2] in detected_cnt):
                    same_num = same_num + 1

            if same_num > 1:
                # exist overlapping arrows
                activator_point, activator_neighbor, receptor_point, receptor_neighbor = \
                  find_vertex_for_detected_arrow(origin_img, binary_img,
                                                                     img_file,
                                                                     vertex_candidates,
                                                                     head_box)
            else:
                # straight forward way
                activator_point, activator_neighbor, receptor_point, receptor_neighbor = \
                    find_vertex_for_detected_arrow_by_distance(binary_img,
                                                               vertex_candidates,
                                                               head_box)
            del vertex_candidates

            activator_list.append(activator_point)
            receptor_list.append(receptor_point)
            activator_neighbor_list.append(activator_neighbor)
            receptor_neighbor_list.append(receptor_neighbor)


    # cv2.imwrite(img_file[:-4]+'_detected.png', origin_img)
    del img, origin_img, detected_list
    return activator_list, activator_neighbor_list, receptor_list, \
           receptor_neighbor_list, text_shapes, inhibit_shapes



    #return activator, receptor
#update by Weiwei Wang 2019/4/4
# def find_all_arrows( img_file, label_file):
#     origin_img = cv2.imread(img_file)
#     label = LabelFile(label_file)
#     arrow_shapes = label.get_all_shapes_for_category('arrow')
#     text_shapes = label.get_all_shapes_for_category('text')
#     img = origin_img.copy()
#     erase_all_text(img, img_file, text_shapes)
#     candidate_contours = detect_all_contours(img, img_file)
#     activator_list = []
#     receptor_list= []
#     for arrow_shape in arrow_shapes:
#       head_box = arrow_shape['points']
#       # generate a random color
#       r = random.randint(0, 255)
#       g = random.randint(0, 255)
#       b = random.randint(0, 255)
#       #draw the box of arrow
#       cv2.rectangle(origin_img, tuple(head_box[0]), tuple(head_box[1]), (r,g,b), \
#                     thickness=2)
#       cnt_idx = match_head_and_arrow_contour(img, head_box, candidate_contours)
#                                              #,detected_list)
#       if cnt_idx != -1:
#         #match successfully, then draw the arrow
#
#         contex_candidates = cv2.convexHull(candidate_contours[cnt_idx],
#                                      returnPoints = True)
#
#         #cv2.drawContours(origin_img, hull, -1, (255, 0, 0), thickness=5)
#         activator, receptor = find_contex_for_detected_arrow(contex_candidates,
#                                                      head_box)
#         # detected_arrow=np.vstack([activator,receptor])
#         # detected_arrow=detected_arrow.tolist()
#         activator=activator.tolist()
#         receptor=receptor.tolist()
#         cv2.circle(origin_img, tuple(activator), 5, (r,g,b), thickness= -1)
#         cv2.circle(origin_img, tuple(receptor), 5, (r,g,b), thickness= -1)
#         #detected_list.append(cnt_idx)
#         activator_list.append(activator)
#         receptor_list.append(receptor)
#     cv2.imwrite(img_file[:-4]+'_detected.png', origin_img)
#     del img,origin_img
#
#     return activator_list,receptor_list,text_shapes,arrow_shapes






# update by Weiwei Wang 2019/4/4
def pair_gene(activator_list,receptor_list,text_shapes,img_file):
    origin_img = cv2.imread(img_file)
    sp = origin_img.shape
    text_regions = []
    text_genename=[]
    relationships=[]
    for text_shape in text_shapes:
        text_regions.append(text_shape['points'])
        text_genename.append(text_shape['label'])
    for idx in range(0,len(activator_list)):
        activator=activator_list[idx]
        receptor=receptor_list[idx]
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        # if (activator==np.array([0,0],dtype = np.int32)).all() is False:
        if activator is not None:
            nearest_activator_index = find_nearest_text(activator, text_regions,sp)
            nearest_receptor_index=find_nearest_text(receptor,text_regions,sp)

            if nearest_activator_index is not None and nearest_receptor_index is not None and nearest_receptor_index!=nearest_activator_index:
                activator_geo=text_regions[nearest_activator_index]
                activator_gene = text_genename[nearest_activator_index]
                receptor_geo=text_regions[nearest_receptor_index]
                receptor_gene = text_genename[nearest_receptor_index]
                relationship='activate:'+activator_gene.split(':',1)[1]+'|'+receptor_gene.split(':',1)[1]
                cv2.rectangle(origin_img, (activator_geo[0][0], activator_geo[0][1]),
                              (activator_geo[2][0], activator_geo[2][1]),
                              (r, g, b), thickness=2)
                cv2.rectangle(origin_img, (receptor_geo[0][0], receptor_geo[0][1]),
                              (receptor_geo[2][0], receptor_geo[2][1]),
                              (r, g, b), thickness=2)
                cv2.circle(origin_img, tuple(activator), 5, (r, g, b), thickness=-1)
                cv2.circle(origin_img, tuple(receptor), 5, (r, g, b), thickness=-1)
            else:
                relationship = 'activate:'
        else:
            relationship='activate:'
        relationships.append(relationship)
    print(relationships)
    del origin_img
    return relationships

#
# def find_all_arrows( img_file, label_file):
#     #import pair gene relationship json files path
#     pair_gen_path=cfg.final_json_path
#     origin_img = cv2.imread(img_file)
#     #get picture’s size
#     sp=origin_img.shape
#     label = LabelFile(label_file)
#     arrow_shapes = label.get_all_shapes_for_category('arrow')
#     text_shapes = label.get_all_shapes_for_category('text')
#     #2019/3/7 cuixin:all the arrows or texts location in arrow shapes and text shapes(list type)
#     print(arrow_shapes)
#     print(text_shapes)
#     img = origin_img.copy()
#     erase_all_text(img, img_file, text_shapes)
#
#     candidate_contours = detect_all_contours(img, img_file)
#     #detected_list = []
#     # 2019/3/8 CX
#     text_regions = []
#     text_genename=[]
#     for text_shape in text_shapes:
#         text_regions.append(text_shape['points'])
#         text_genename.append(text_shape['label'])
#     # f = open(img_file[:-4] + '.json', 'a', encoding="utf-8")
#     gene_connection = []
#     activator_gene_name=[]
#     activator_geo_loc=[]
#
#     arrow_receptor=[]
#     receptor_gene_name=[]
#     receptor_geo_loc=[]
#
#     shapes = []
#     tempDict2={}
#     for arrow_shape in arrow_shapes:
#       tempDict = {}
#       arrow_activate = []
#       head_box = arrow_shape['points']
#       # generate a random color
#       r = random.randint(0, 255)
#       g = random.randint(0, 255)
#       b = random.randint(0, 255)
#       #draw the box of arrow
#       cv2.rectangle(origin_img, tuple(head_box[0]), tuple(head_box[1]), (r,g,b), \
#                     thickness=2)
#       cnt_idx = match_head_and_arrow_contour(img, head_box, candidate_contours)
#                                              #,detected_list)
#       if cnt_idx != -1:
#         #match successfully, then draw the arrow
#
#         contex_candidates = cv2.convexHull(candidate_contours[cnt_idx],
#                                      returnPoints = True)
#
#         #cv2.drawContours(origin_img, hull, -1, (255, 0, 0), thickness=5)
#         activator, receptor = find_contex_for_detected_arrow(contex_candidates,
#                                                      head_box)
#         # cv2.circle(origin_img, tuple(activator), 5, (r, g, b), thickness=-1)
#         # cv2.circle(origin_img, tuple(receptor), 5, (r, g, b), thickness=-1)
#         #2019.3.8 cx
#         nearest_activator_index = find_nearest_text(activator, text_regions,sp)
#         nearest_receptor_index=find_nearest_text(receptor,text_regions,sp)
#         if nearest_activator_index is not None and nearest_receptor_index is not None and nearest_receptor_index!=nearest_activator_index:
#             activator_geo=text_regions[nearest_activator_index]
#             activator_gene = text_genename[nearest_activator_index]
#             receptor_geo=text_regions[nearest_receptor_index]
#             receptor_gene = text_genename[nearest_receptor_index]
#             relationship='action:'+activator_gene+'<activate>'+receptor_gene
#             cv2.rectangle(origin_img, (activator_geo[0][0], activator_geo[0][1]),
#                           (activator_geo[1][0], activator_geo[1][1]),
#                           (r, g, b), thickness=2)
#             cv2.rectangle(origin_img, (receptor_geo[0][0], receptor_geo[0][1]),
#                           (receptor_geo[1][0], receptor_geo[1][1]),
#                           (r, g, b), thickness=2)
#             cv2.circle(origin_img, tuple(activator), 5, (r, g, b), thickness=-1)
#             cv2.circle(origin_img, tuple(receptor), 5, (r, g, b), thickness=-1)
#         else:
#             relationship = 'action:'
#
#         # activator_gene_name.append(activator_gene)
#         # activator_geo_loc.append(activator_geo)
#         # arrow_activate.append(head_box)
#         # arrow_receptor.append(receptor.tolist())
#         # receptor_gene_name.append(receptor_gene)
#         # receptor_geo_loc.append(receptor_geo)
#         # print(gene_connection)
#         # 存arrow 的dict
#         tempDict['label'] =relationship
#         tempDict['line_color'] = None
#         tempDict['fill_color'] = None
#         tempDict['points'] = head_box
#         tempDict['shape_type'] = 'rectangle'
#         tempDict['alias'] = 'name'
#         shapes.append(tempDict)
#
#     # 存text的dict
#     for i in text_shapes:
#         shapes.append(i)
#     tempDict2['version'] = '3.6.1'
#     tempDict2['flags'] = {}
#     tempDict2['shapes'] = shapes
#     tempDict2['lineColor'] = None
#     tempDict2['fillColor'] = None
#     tempDict2['imagePath'] =label.imagePath
#     tempDict2['imageData'] = None
#     tempDict2['imageHeight'] = None
#     tempDict2['imageWidth'] = None
#     f = open(os.path.join(pair_gen_path,label.imagePath[:-4]) + '.json', 'a', encoding="utf-8")
#     # f = open(os.path.join(pair_gen_path, label.imagePath.split('\\')[2])[:-4] + '.json', 'a', encoding="utf-8")
#     tempJson = json.dumps(tempDict2, indent=1, ensure_ascii=False)
#     f.write(tempJson+'\n')
#     f.close()
#     print(tempJson)

    # geneconnection = {}
    # geneconnection['activator_gene_name'] = activator_gene_name
    # geneconnection['activator_geo'] = activator_geo_loc
    # geneconnection['arrow_activate'] = arrow_activate
    # geneconnection['arrow_receptor'] = arrow_receptor
    # geneconnection['receptor_gene_name'] = receptor_gene_name
    # geneconnection['receptor_geo'] = receptor_geo_loc
    # print(geneconnection)
    # f1=DataFrame(geneconnection)
    # f1.to_csv(path_or_buf=r'D:\Study\Master\1st deep learning project\Dima data(2)\Dima data\Images\Test.csv')
    # tempJson = json.dumps(geneconnection, indent=4, ensure_ascii=False)
    # f.write(tempJson + '\n')
    # f.close()
    # print(tempJson)

        #detected_list.append(cnt_idx)
    # cv2.imwrite(img_file[:-4]+'_detected.png', origin_img)
    # del img,origin_img




if __name__== "__main__":
    #img_fold = r'C:\Users\LSC-110\Desktop\Dima data\Images'
    #label_fold = r'C:\Users\LSC-110\Desktop\Dima data\labeled'
    #img_fold = r'D:\Study\Master\1st deep learning project\Dima data(2)\Dima data\Images'
    #label_fold = r'D:\Study\Master\1st deep learning project\Dima data(2)\Dima data\labeled'
    #code for individual
    # image = r'cin_00001.png'
    # img_file = os.path.join(img_fold, image)
    # label_file = os.path.join(label_fold, image[:-4] + '.json')
    # find_all_arrows(img_file, label_file)

    #for batch
    #for image in os.listdir(img_fold):
        # img_file =r'D:\Study\Master\1st deep learning project\Dima data(2)\Dima data\Images\cin_00066.png'
        # # label_file =r'D:\Study\Master\1st deep learning project\Dima  data(2)\Dima data\labeled\cin_00015.json'
        # label_file = r'D:\Study\Master\1st deep learning project\Dima data(2)\Dima data\imagetest\tolabelme_txt\cin_00066.json'
        #
        # #img = cv2.imread(img_file)
        # find_all_arrows( img_file, label_file)
    img_fold = r'C:\Users\LSC-110\Desktop\cxtest\Images'
    label_fold = r'C:\Users\LSC-110\Desktop\cxtest\labeled'
    # img_fold = r'D:\Study\Master\1st deep learning project\Dima data(2)\Dima data\Images'
    # label_fold = r'D:\Study\Master\1st deep learning project\Dima data(2)\Dima data\labeled'

    for image in os.listdir(img_fold):
        # image = r'cin_00001.png'
        img_file = os.path.join(img_fold, image)
        label_file = os.path.join(label_fold, image[:-4] + '.json')
        find_all_arrows(img_file, label_file)
















