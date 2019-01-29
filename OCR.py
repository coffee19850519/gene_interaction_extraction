import cv2
import cfg,os,subprocess
import numpy as np


def OCR(image_path, image_file, text_regions_coords):

    image_array = cv2.imread(os.path.join(image_path, image_file))
    results = []
    idx = 0
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
        textImage = cv2.resize(image_array[y1- cfg.OCR_OFFSET:y1 + h + cfg.OCR_OFFSET,
                               x1 - cfg.OCR_OFFSET:x1 + w + cfg.OCR_OFFSET],
                               (int(cfg.OCR_SCALE * w),
                                int(cfg.OCR_SCALE * h)),
                               interpolation=cv2.INTER_CUBIC)
        # textImage = image_array[y1 - cfg.OCR_OFFSET:y1 + h + cfg.OCR_OFFSET,
        #               x1 - cfg.OCR_OFFSET:x1 + w + cfg.OCR_OFFSET]

        if cfg.DEBUG and textImage.size > 0:
            if not os.path.exists(os.path.join(image_path, image_file[:-4])):
                os.mkdir(os.path.join(image_path, image_file[:-4]))
            cv2.imwrite(os.path.join(os.path.join(image_path, image_file[:-4]), str(idx) + '.png'),
                        textImage)

        result = ocr_text_from_image(textImage)
        idx += 1
        if result is not None and result != '':
            results.append(result)
        else:
            text_regions_coords.remove(text_coord)
        del textImage

    return results




def ocr_text_from_image(text_img_file, output):


    # regist pytesseract path
    #pytesseract.pytesseract.tesseract_cmd = cfg.TERSSERACT_PATH

    # call tersseract to reconize text
    # result = pytesseract.image_to_string(region_array, lang='eng',
    #                                    config=cfg.TERSSERACT_CONFIG)

    comm = "tesseract \"" + path + str(idx) + "_resized_image_" + str(
        Configer.SCALE) + ".png\"" + " \"" + textfile + "\" --oem 0 --psm 3 -l eng bazaar"  # 0 is the Legacy
    status = subprocess.getoutput(comm)
    if os.path.exists(textfile + ".txt") and "Can't" not in status[1]:
        with open(textfile + ".txt") as temp_file:
            result = temp_file.readlines()[:-1]
            temp_file.close()
    else:
        result = None

    return result