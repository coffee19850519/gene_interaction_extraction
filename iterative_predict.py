from predict import predict
from network import East
import cfg,os,shutil,cv2
import numpy as np


def erase_image(labels, target_image_path):
    target_img = cv2.imread(target_image_path)
    mask = np.zeros(target_img.shape[:2], np.uint8)
    for label in labels:
        shape, coord = label.split('\t')
        coord = coord.split(',')
        coord = [float(x) for x in coord]
        left_x = int(np.floor(min([coord[0], coord[2], coord[4], coord[6]])))
        right_x = int(np.ceil(max([coord[0], coord[2], coord[4], coord[6]])))
        left_y = int(np.floor(min([coord[1], coord[3], coord[5], coord[7]])))
        right_y = int(np.ceil(max([coord[1], coord[3], coord[5], coord[7]])))


        # capture its coordinate points
        # coords = np.array(list(map(float, corod.split(','))),
        #                   np.int32).reshape((4, 2))
        #remove detected objects
        cv2.rectangle(target_img, (left_x, left_y), (right_x, right_y), (255, 255,
                                                                   255), -1)
        #set its mask for background inputation
        cv2.rectangle(mask,(left_x,left_y),(right_x,right_y),(255, 255, 255),
                      -1)


        #cv2.fillPoly(target_img, box ,(255, 255, 255))
        #cv2.polylines(target_img,[coords], -1, (255, 255, 255))
    inputation_img = cv2.inpaint(target_img, mask, 5, cv2.INPAINT_TELEA)
    target_image_path, image_ext = target_image_path.split('+')
    idx, image_ext = os.path.splitext(image_ext)
    cv2.imwrite(target_image_path + '+' + str(int(idx) + 1) + image_ext,
                inputation_img)
    del target_img, mask, inputation_img


def iterative_prediction(image_path, image_name, result_image_path,
                         result_path, iterative_time):
    #copy original image to result_image_path
    if not os.path.exists(result_image_path):
        os.mkdir(result_image_path)
    if  not os.path.exists(result_path):
        os.mkdir(result_path)

    image_name, image_ext = os.path.splitext(image_name)
    shutil.copyfile(os.path.join(image_path, image_name+image_ext),
                    os.path.join(result_image_path, image_name+'+0'+image_ext))
    results = []
    #establish network
    east = East()
    east_detect = east.east_network()
    east_detect.load_weights(cfg.saved_model_weights_file_path)

    #start iteration
    for i in range(1, iterative_time + 1, 1):
        #load i-th result as input to predict
        current_result = predict(east_detect,
                                 os.path.join(result_image_path,image_name + '+' +
                                  str(i-1)+image_ext) ,

                                  text_pixel_threshold= cfg.text_pixel_threshold - i *
                                                   0.05,
                                  text_side_threshold=
                                              cfg.text_side_vertex_pixel_threshold - i * 0.05,

                                  text_trunc_threshold= cfg.text_trunc_threshold +
                                                   i * 0.03,
                                  action_pixel_threshold=
                                          cfg.action_pixel_threshold,
                                  arrow_trunc_threshold=
                                          cfg.arrow_trunc_threshold,
                                  nock_trunc_threshold=
                                          cfg.nock_trunc_threshold,

                                  quiet=False)


        results.extend(current_result)
        #according to current results, erase last iteration image
        erase_image(current_result,os.path.join(result_image_path,image_name + '+' +
                                                str(i-1)+image_ext))
        del current_result
    with open(os.path.join(result_path, image_name+'.txt'),'w') as result_fp:
        result_fp.writelines(results)

    del results


if __name__ == '__main__':
    #args = parse_args()
    #img_path = args.path
    #threshold = float(args.threshold)
    #print(img_path, threshold)
    # img_path = r'C:\Users\LSC-110\Desktop\test'
    # result_image_path = r'C:\Users\LSC-110\Desktop\test\iterative_images'
    # result_label_path = r'C:\Users\LSC-110\Desktop\test\iterative_labels'
    img_path = r'D:\Study\Master\1st deep learning project\Dima data(2)\Dima data\imagetest'
    result_image_path = r'D:\Study\Master\1st deep learning project\Dima data(2)\Dima data\imagetest\iterative_images'
    result_label_path = r'D:\Study\Master\1st deep learning project\Dima data(2)\Dima data\imagetest\iterative_labels'
    iterative_times = 4
    for image_file in os.listdir(img_path):
        iterative_prediction(img_path, image_file, result_image_path,
                             result_label_path, iterative_times)

