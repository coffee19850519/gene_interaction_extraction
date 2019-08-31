import os
from label_file import LabelFile
import cv2
import numpy as np


def to_labelme(image_name,
               image_txt,
               predicted_relation_boxes,
               predicted_relation_description,
               ocr_path,
               to_labelmefolder,
               current_img):
    colorarrow = [255, 0, 0, 128]
    colorinhibit = [0, 255, 0, 128]
    colorgene = [0, 0, 0, 128]
    shapes = []
    index = 0
    result_string = ''
    # annotation by CX:2019/4/4
    # f = open(os.path.join(to_labelmefolder,fename[:-8])+'.json', 'a', encoding="utf-8")
    #读取predict后的txt取出arror信息
    for box,description in zip(predicted_relation_boxes, predicted_relation_description):
        tempDict = {}
        tempDict['label'] = description
        if description.strip().split(':',1)[0] == 'activate':
            tempDict['line_color'] = colorarrow
        elif  description.strip().split(':',1)[0] == 'inhibit':
            tempDict['line_color'] = colorinhibit
        tempDict['fill_color'] = None
        tempDict['points'] = box['relationship_bounding_box']
        tempDict['shape_type'] = 'polygon'
        tempDict['alias'] = 'name'
        shapes.append(tempDict)
        #plot the index and box into current_img
        #cv2.drawContours(current_img,[box['relationship_bounding_box']], -1, tempDict['line_color'], thickness=2)
        cv2.putText(current_img, str(index), (box['relationship_bounding_box'][3][0] - 5,
                                              box['relationship_bounding_box'][3][1] - 5),
                                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, tempDict['line_color'])
        result_string += str(index) + '\t' + description + '\n'
        index += 1
        del tempDict

    # with open(predict_path, 'r') as pre_res:
    #     pre_results = pre_res.readlines()
    #     for pre_each in pre_results:
    #         tempDict = {}
    #         points = []
    #         shape, str_coords = str(pre_each).strip().split('\t',1)
    #         if shape== 'arrow':
    #             tempDict['label'] = 'activate:'
    #             tempDict['line_color'] = colorarrow
    #             tempDict['fill_color'] = None
    #             A=str_coords.replace('[','')
    #             A=A.replace(']','')
    #             A=A.split(',')
    #             x1=int(A[0])
    #             y1=int(A[1])
    #             x2=int(A[2])
    #             y2=int(A[3])
    #             x3 = int(A[4])
    #             y3 = int(A[5])
    #             x4 = int(A[6])
    #             y4 = int(A[7])
    #             A1 = (x1, y1)
    #             A2 = (x2, y2)
    #             A3 = (x3, y3)
    #             A4 = (x4, y4)
    #             points.append(A1)
    #             points.append(A2)
    #             points.append(A3)
    #             points.append(A4)
    #             tempDict['points'] = points
    #             del (points)
    #             tempDict['shape_type'] = 'polygon'
    #             tempDict['alias'] = 'name'
    #             shapes.append(tempDict)
    #         elif shape=='text':
    #             pass
    #         elif shape=='nock':
    #             tempDict['label'] = 'inhibit:'
    #             tempDict['line_color'] = colorinhibit
    #             tempDict['fill_color'] = None
    #             A = str_coords.replace('[', '')
    #             A = A.replace(']', '')
    #             A = A.split(',')
    #             x1 = int(A[0])
    #             y1 = int(A[1])
    #             x2 = int(A[2])
    #             y2 = int(A[3])
    #             x3 = int(A[4])
    #             y3 = int(A[5])
    #             x4 = int(A[6])
    #             y4 = int(A[7])
    #             A1 = (x1, y1)
    #             A2 = (x2, y2)
    #             A3 = (x3, y3)
    #             A4 = (x4, y4)
    #             points.append(A1)
    #             points.append(A2)
    #             points.append(A3)
    #             points.append(A4)
    #             tempDict['points'] = points
    #             del (points)
    #             tempDict['shape_type'] = 'polygon'
    #             tempDict['alias'] = 'name'
    #             shapes.append(tempDict)

            # elif shape=='bind':
                # return shape, str_coords
    #读取OCR->correction后的txt
    with open(ocr_path, 'r') as OCR_RES:
        OCR_results = OCR_RES.readlines()
        for OCR_each in OCR_results:
            tempDict = {}
            points = []
            idx, OCR_result, str_coords = str(OCR_each).strip().split('@')
            tempDict['label'] = 'gene:'+ OCR_result
            tempDict['line_color'] = colorgene
            tempDict['fill_color'] = None
            A=str_coords.replace('[','')
            A=A.replace(']','')
            A=A.split(',')
            x1=int(A[0])
            y1=int(A[1])
            x2=int(A[2])
            y2=int(A[3])
            x3=int(A[4])
            y3=int(A[5])
            x4=int(A[6])
            y4=int(A[7])
            A1 =(x1,y1)
            A2 =(x2,y2)
            A3 =(x3,y3)
            A4 =(x4,y4)
            points.append(A1)
            points.append(A2)
            points.append(A3)
            points.append(A4)
            tempDict['points'] = points
            del (points)
            tempDict['shape_type'] = 'polygon'
            tempDict['alias'] = 'name'
            shapes.append(tempDict)
            # plot the index and box into current_img
            # cv2.drawContours(current_img, np.array(tempDict['points']), -1, tempDict['line_color'], thickness=2)
            cv2.putText(current_img, str(index), (A3[0] - 5, A3[1] - 5),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, tempDict['line_color'])

            result_string += str(index) + '\t' + OCR_result + '\n'
            index += 1
            del tempDict
            # return shape, str_coords
    #         annotation by CX 2019/4/4
    # tempDict3['version'] = '3.6.1'
    # tempDict3['flags'] = {}
    # tempDict3['shapes'] = shapes
    # tempDict3['lineColor'] = None
    # tempDict3['fillColor'] = None
    # tempDict3['imagePath'] = fename[:-4]
    # # tempDict3['imagePath'] = path + ".png"
    # tempDict3['imageData'] = None
    # tempDict3['imageHeight'] = None
    # tempDict3['imageWidth'] = None
    # tempJson = json.dumps(tempDict3, indent=1, ensure_ascii=False)
    # f.write(tempJson + '\n')
    # f.close()
    # print(tempJson)
    #call save function 2019/4/4
    label= LabelFile()
    label.save(os.path.join(to_labelmefolder, image_name + '.json'),
               shapes,
               image_name + image_txt,
               None,None,None,None,{})
    return result_string
        ########################################
        # coords = np.array(str(str_coords).strip().split(','),np.int32).reshape((4, 2))

if __name__ == '__main__':
    correctfolder=r'D:\Study\Master\1st deep learning project\Dima data(2)\Dima data\imagetest\correct_txt'
    predictfolder=r'D:\Study\Master\1st deep learning project\Dima data(2)\Dima data\predict_txt'
    to_labelmefolder=r'D:\Study\Master\1st deep learning project\Dima data(2)\Dima data\imagetest\tolabelme_txt'
    ocr_path_list = []
    predict_path_list=[]
    for txt in os.listdir(correctfolder):
        ocr_path=os.path.join(correctfolder,txt)
        ocr_path_list.append(ocr_path)
    for txt in os.listdir(predictfolder):
        predict_path=os.path.join(predictfolder,txt)
        predict_path_list.append(predict_path)
    # ocr_path=r'D:\Study\Master\1st deep learning project\Dima data(2)\Dima data\imagetest\cin_00089_OCR.txt'
    # predict_path=r'D:\Study\Master\1st deep learning project\Dima data(2)\Dima data\cin_00089.txt'
    id=0
    while id<len(ocr_path_list):
        to_labelme(predict_path_list[id],ocr_path_list[id],to_labelmefolder)
        id=id+1
#
#         h = abs(coords[0][1] - coords[2][1])
#         w = abs(coords[0][0] - coords[2][0])
#         Xs = [i[0] for i in coords]
#         Ys = [i[1] for i in coords]
#         x1 = min(Xs)
#         y1 = min(Ys)
#
#         # scaling ROI for better recognition
#         textImage = cv2.resize(image_array[y1 - cfg.OCR_OFFSET:y1 + h + cfg.OCR_OFFSET,
#                                x1 - cfg.OCR_OFFSET:x1 + w + cfg.OCR_OFFSET],
#                                (int(cfg.OCR_SCALE * w),
#                                 int(cfg.OCR_SCALE * h)),
#                                interpolation=cv2.INTER_CUBIC)
#         # textImage = image_array[y1 - cfg.OCR_OFFSET:y1 + h + cfg.OCR_OFFSET,
#         #               x1 - cfg.OCR_OFFSET:x1 + w + cfg.OCR_OFFSET]
#
#         if cfg.DEBUG and textImage.size > 0:
#             if not os.path.exists(os.path.join(image_path, image_file[:-4])):
#                 os.mkdir(os.path.join(image_path, image_file[:-4]))
#             cv2.imwrite(os.path.join(os.path.join(image_path, image_file[:-4]), str(idx) + '.png'),
#                         textImage)
#             text_image_file = os.path.join(os.path.join(image_path, image_file[:-4]))
#
#         result = ocr_text_from_image(text_image_file, idx)
#         idx += 1
#         # if result is not None and result != ' \n':
#         if result and result != [' \n']:
#             coords = coords.tolist()
#             results.append(result)
#             coord.append(coords)
#             # results.append('\n')
#
#         else:
#             text_regions_coords.remove(text_coord)
#         del textImage
#     print(results)
#     file = open(text_image_file + "_OCR.txt", 'w')
#     # file.writelines(results)
#     # for result in results:
#     for idx in range(len(results)):
#         a = results[idx]
#         b = coord[idx]
#         s = ''.join(a).strip('\n') + '\t' + str(b)
#         file.write(s)
#         file.write('\n')
#     file.close()
#
#     return results
# rootdir = './AAA/'
# list = os.listdir(rootdir)
# for i in list:
#     # path = os.path.join(rootdir,list[i])
#     index = i.rfind('.')
#     path = i[:index]
#     root1 = et.parse('./AAA/' + path + '.xml')
#     root = root1.getroot()  # 拿到根标签<pathway>
#     f = open('./CCC/' + path + '.json', 'a', encoding="utf-8")
#     shapes = []
#     tempDict3 = {}
#     for entry in root.iter('entry'):
#         tempDict = {}
#         points = []
#         graphics = entry.find('graphics')
#         if graphics.attrib['type'] == 'rectangle':
#             tempDict['label'] = entry.attrib['type'] + ':' + entry.attrib['name']
#             # tempDict['label'] = graphics.attrib['name']
#             tempDict['line_color'] = None
#             tempDict['fill_color'] = None
#             # if graphics.attrib['type'] == 'rectangle':
#             x = int(graphics.attrib['x'])
#             y = int(graphics.attrib['y'])
#             width = int(graphics.attrib['width'])
#             height = int(graphics.attrib['height'])
#             x1 = int(x - width / 2)
#             y1 = int(y + height / 2)
#             x2 = int(x + width / 2)
#             y2 = int(y - height / 2)
#             # A = [[0] * 2 for i in range(2)]
#             A1 = (x1, y1)
#             A2 = (x2, y2)
#             points.append(A1)
#             points.append(A2)
#
#             # A[0][0] = x1
#             # A[0][1] = y1
#             # A[1][0] = x2
#             # A[1][1] = y2
#             tempDict['points'] = points
#             del (points)
#             tempDict['shape_type'] = graphics.attrib['type']
#             try:
#                 tempDict['alias'] = graphics.attrib['name']
#             except:
#                 tempDict['alias'] = 'no name attribute'
#             else:
#                 pass
#             shapes.append(tempDict)
#         else:
#             pass
#
#         # print(tempDict)
#
#     # print(tempDict.items())
#     # print(tempJson)
#
#     # tempJson = json.dumps(shapes,ensure_ascii=False)
#     # print(shapes)
#
#     tempDict3['version'] = '3.6.1'
#     tempDict3['flags'] = {}
#     tempDict3['shapes'] = shapes
#     tempDict3['lineColor'] = None
#     tempDict3['fillColor'] = None
#     tempDict3['imagePath'] = path + ".png"
#     tempDict3['imageData'] = None
#     tempDict3['imageHeight'] = None
#     tempDict3['imageWidth'] = None
#     tempJson = json.dumps(tempDict3, indent=1, ensure_ascii=False)
#     f.write(tempJson + '\n')
#     f.close()
#     print(tempJson)
#
# """
# tempJson=json.dumps(shapes,ensure_ascii=False)
# print(tempJson)
# f.write(tempJson+'\n')
# f.close()
# """
# """
#     for graphics in root1.iter('graphics'):
#         tempDict['label']=graphics.attrib['name']
#         tempDict['line_color']=None
#         tempDict['fill_color']=None
#         if graphics.attrib['type']=='rectangle':
#             x=int(graphics.attrib['x'])
#             y=int(graphics.attrib['y'])
#             width=int(graphics.attrib['width'])
#             height=int(graphics.attrib['height'])
#             x1=int(x-width/2)
#             y1=int(y+height/2)
#             x2=int(x+width/2)
#             y2=int(y-height/2)
#             A=[[0]*2 for i in range(2)]
#             A[0][0]=x1
#             A[0][1]=y1
#             A[1][0]=x2
#             A[1][1]=y2
#             tempDict['points'] = A
#         else:
#             tempDict['points']=None
#         tempDict['shape_type']=graphics.attrib['type']
#         #print(tempDict)
#         #print(tempDict.items())
#         tempJson=json.dumps(tempDict,ensure_ascii=False)
#         print(tempJson)
#         """
