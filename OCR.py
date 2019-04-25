import cv2
import cfg,os,subprocess
import numpy as np
#!/usr/bin/python2.6
# -*- coding: utf-8 -*-

def OCR(image_path, image_file, text_regions_coords):
    image_array = cv2.imread(os.path.join(image_path, image_file))
    results = []
    coord = []
    idx = 0
    subimage_path=cfg.OCR_subimage_path
    for text_coord in text_regions_coords:
        shape, str_coords = str(text_coord).strip().split('\t')
        coords = np.array(str(str_coords).strip().split(','),
                          np.int32).reshape((4,2))
        if shape != 'text':
            continue
        #x, y, w, h = cv2.boundingRect(np.array(coords)
        # rect = cv2.minAreaRect(np.array(coords, dtype='float32'))
        # box = cv2.cv.BoxPoints(rect)
        #box = np.int0(box)

        # x0 = round(x - w * 0.5 - cfg.OCR_OFFSET)
        # if x0 < 0:
        #     x0 = 0
        # y0 = round(y - h * 0.5 - cfg.OCR_OFFSET)
        # if y0 < 0:
        #     y0 = 0
        # xe = round(x + w * 0.5 + cfg.OCR_OFFSET)
        # if xe > image_array.shape[0]:
        #     xe = image_array.shape[0]
        # ye = round(y + h * 0.5 + cfg.OCR_OFFSET)
        # if ye > image_array.shape[1]:
        #     ye = image_array.shape[1]


        h = abs(coords[0][1] - coords[2][1])
        w = abs(coords[0][0] - coords[2][0])
        Xs = [i[0] for i in coords]
        Ys = [i[1] for i in coords]
        x1 = min(Xs)
        y1 = min(Ys)

        # scaling ROI for better recognition
        try:
            textImage = cv2.resize(image_array[y1- cfg.OCR_OFFSET:y1 + h + cfg.OCR_OFFSET,
                                   x1 - cfg.OCR_OFFSET:x1 + w + cfg.OCR_OFFSET],
                                   (int(cfg.OCR_SCALE * w),
                                    int(cfg.OCR_SCALE * h)),
                                   interpolation=cv2.INTER_CUBIC)
            # textImage = image_array[y1 - cfg.OCR_OFFSET:y1 + h + cfg.OCR_OFFSET,
            #               x1 - cfg.OCR_OFFSET:x1 + w + cfg.OCR_OFFSET]
            if cfg.DEBUG and textImage.size > 0:
                if not os.path.exists(os.path.join(subimage_path, image_file[:-4])):
                    os.mkdir(os.path.join(subimage_path, image_file[:-4]))
                cv2.imwrite(os.path.join(os.path.join(subimage_path, image_file[:-4]), str(idx) + '.png'),
                            textImage)
                text_image_file=os.path.join(os.path.join(subimage_path, image_file[:-4]))

            result = ocr_text_from_image(text_image_file,idx)
            idx += 1
            #if result is not None and result != ' \n':
            if result and result != [' \n']:
                coords=coords.tolist()
                results.append(result)
                coord.append(coords)
                #results.append('\n')

            else:
                text_regions_coords.remove(text_coord)
            del textImage
        except Exception as err:
            print('Exception: ', err)
    print(results)
    # file = open(text_image_file + "_OCR.txt",'w')  change to next row by Xin
    file = open(cfg.OCR_result_path+'\\'+image_file[:-4]+"_OCR.txt", 'w')

    #file.writelines(results)
    #for result in results:
    ocr_results=[]
    for idx in range(len(results)):
        a=results[idx]
        b=coord[idx]
        ocr_result=''.join(a).replace('\n','').replace('\t','')
        ocr_results.append(ocr_result)
        s = ''.join(a).replace('\n','').replace('\t','')+'\t'+str(b)
        # s = a.strip('\n') + '\t' + str(b)
        file.write(s)
        file.write('\n')
    file.close()

    return ocr_results,coord




# def ocr_text_from_image(text_img_file, output):
def ocr_text_from_image(text_img_file,idx):


    # regist pytesseract path
    # pytesseract.pytesseract.tesseract_cmd = cfg.TERSSERACT_PATH
    #
    # call tersseract to reconize text
    # result = pytesseract.image_to_string(region_array, lang='eng',
    #                                    config=cfg.TERSSERACT_CONFIG)

    #comm = "tesseract \"" + text_img_file + str(idx) + "_resized_image_" + str(
    #    cfg.OCR_SCALE) + ".png\"" + " \"" + text_img_file + "\" --oem 0 --psm 3 -l eng bazaar"
    comm = "tesseract \"" + text_img_file +"\\" +  str(idx) + ".png\"" + " \"" + text_img_file + "\" --oem 0 --psm 3 -l eng digits"
    # 0 is the Legacy
    status = subprocess.getoutput(comm)
    if os.path.exists(text_img_file + ".txt") and "Can't" not in status[1]:
        with open(text_img_file + ".txt",encoding='gb18030', errors='ignore') as temp_file:
        #with open(text_img_file + ".txt", encoding='utf-8', errors='ignore') as temp_file:
            # file2 = open(text_img_file+"1.txt",'w',encoding='utf-8')
            # file2.write(temp_file.read())
            # file2.close()
            # with open(text_img_file+"1.txt",'r') as file:
            #     result=file.readlines()[:-1]
            # # result = temp_file.readlines()[:-1]
            # file.close()
            # temp_file.close()
            result=temp_file.readlines()[:-1]
            temp_file.close()
    else:
        result = []

    return result

