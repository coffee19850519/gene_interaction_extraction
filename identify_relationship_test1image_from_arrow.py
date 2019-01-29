import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import os,shutil
from configuration import Configer
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import io
import commands
def binary_image(srcImage):
    #gray = cv2.fastNlMeansDenoisingColored(srcImage,
    #                                       Configer.DENOISE_STRENGTH,
    #                                       Configer.DENOISE_COLOR_STRENGTH,
    #                                       Configer.DENOISE_TEMPLATE_SIZE,
    #                                       Configer.SEARCH_SIZE)
    gray=cv2.cvtColor(srcImage, cv2.COLOR_RGB2GRAY) # convert 3D channel to 1D    
    #binaryImage = cv2.adaptiveThreshold(cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    #                      cv2.THRESH_BINARY, Configer.BLOCK_SIZE, Configer.C)
    #binaryImage=cv2.bitwise_not(binaryImage)
    
    _,binaryImage = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    #_,binaryImage = cv2.threshold(gray,127,255,cv2.THRESH_TRUNC)
    #binaryImage=cv2.bitwise_not(binaryImage)#text should be black
    return binaryImage
    


def ocr_text_from_image(regions,srcImage,path):
    text = []
    result_fp = io.open(path+"/text_results.txt", 'w+', encoding='utf-8')
    #regist pytesseract path
    pytesseract.pytesseract.tesseract_cmd = Configer.TERSSERACT_PATH#'tesseract'
    #Configer.TERSSERACT_CONFIG = '-l eng+equ+osd  --oem 2 --psm 7 '
    Configer.TERSSERACT_CONFIG = '-l eng+equ+osd  --oem 1 --psm 3 bazaar' #3 works for 3line  #tesseract 97_text_image.png test_97 -l eng bazaar
    #generate a copy to binaryImage for saving results
    resImage = srcImage.copy()
    
    idx = 1
    
    for region in regions:
        #if idx ==11:
        #    result_fp.close()
        #    return text
        # crop regions from raw image considering offset
        h = abs(region[3] - region[1])
        w = abs(region[2] - region[0])
        Xs = [region[0],region[2]]
        Ys = [region[1],region[3]]
        x1 = min(Xs)
        y1 = min(Ys)
        Configer.OFFSET= -1 #0
        #binaryImage = binary_image(resImage)
        regionImage = resImage[y1-Configer.OFFSET:y1 + h+Configer.OFFSET,
                      x1-Configer.OFFSET:x1 + w+Configer.OFFSET]
        
        regionImage = binary_image(regionImage)
        if Configer.DEBUG and regionImage.size > 0:
            cv2.imwrite(os.path.join(path, str(idx)+"_text_image.png"), regionImage)
        
        
        
        # scaling ROI for better recogniti
        Configer.SCALE = 10
        textImage = cv2.resize(regionImage, (int(Configer.SCALE * w), int(Configer.SCALE * h)),
                                             interpolation=cv2.INTER_CUBIC)
        if Configer.DEBUG and regionImage.size > 0:
            cv2.imwrite(os.path.join(path, str(idx)+"_resized_image_"+str(Configer.SCALE)+".png"), textImage)
        
        #textImage = regionImage
        # call tersseract to reconize text
        result = pytesseract.image_to_string(textImage, lang='eng', config=Configer.TERSSERACT_CONFIG)
        print("recongnize "+str(idx)+" as "+result+"\n")
        if result is not None:# and result != '':
            text.append(result)
            result_fp.write(str(idx).encode('utf-8')+":".encode('utf-8') + result + '\n'.encode('utf-8'))
        else:
            text.append(u'null\n')
            result_fp.write(str(idx).encode('utf-8')+": null".encode('utf-8') + '\n'.encode('utf-8'))
        
        del regionImage,textImage
        idx = idx + 1
    
    result_fp.close()
    return text

def get_convex_box(region1,region2):
    regionx = [region1[0],region1[2],region2[0],region2[2]]
    regiony = [region1[1],region1[3],region2[1],region2[3]]
    x_left = np.min(regionx)
    x_right = np.max(regionx)
    y_up = np.min(regiony)
    y_down = np.max(regiony)
    return [x_left,y_up,x_right,y_down]

def find_lines_inregion(img,arrow_region,resimg=None,minexp=1e-8): #img = ori_img_binary.copy()
    _,contours,hier = cv2.findContours(
                      cv2.bitwise_not(img[arrow_region[1]:arrow_region[3],
                      arrow_region[0]:arrow_region[2]]),
                      cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    if contours:
       contours_length = [len(x) for x in contours]
       cnt = contours[ np.argmax(contours_length)]
       # then apply fitline() function
       [vx,vy,x,y] = cv2.fitLine(cnt,cv2.DIST_L2,0,0.01,0.01)
       if x:
            x_img = x + arrow_region[0]
            y_img = y+arrow_region[1]
            if resimg:
                if vx < minexp:
                          cv2.line(resimg,(x_img,resimg.shape[0]-1),(x_img,0),(255,255,0),2)
                else:
                     # Now find two extreme points on the line to draw line
                     lefty = int((-x_img*vy/vx) + y_img)
                     righty = int(((img.shape[1]-x_img)*vy/vx)+y_img)
                     cv2.line(resimg,(resimg.shape[1]-1,righty),(0,lefty),(255,255,0),2)
                
                cv2.rectangle(resimg, (arrow_region[0], arrow_region[1]),
                                   (arrow_region[2], arrow_region[3]),
                                   (255,0,255), thickness=1)
            return vx[0],vy[0],x_img[0],y_img[0]
       
       else:
            return 0,0,0,0
    else:
        return 0,0,0,0

def dist(vx,vy,x,y,point):
    QP = [point[0]-x,point[1]-y]
    v=[vx,vy]
    h = np.linalg.norm(np.cross(QP, v)/np.linalg.norm(v))
    return h

def dist_center(point1,point2):
    QP = [point1[0]-point2[0],point1[1]-point2[1]]
    return np.sqrt(QP[0]**2+QP[1]**2)


def find_nearest_text(arrow_region,text_regions):
    dists=[]
    for text_r in text_regions:
        dist = dist_center(arrow_region,text_r)
        dists.append(dist)
    
    nearest_index = np.argmin(dists)
    return nearest_index


#resimg = ori_img.copy()
#for arrow_region in arrow_regions:
#  find_lines_inregion(ori_img_binary.copy(),arrow_region,resimg)
#
#cv2.imwrite('test_find_lines.png', resimg)

def get_candidate_seeds(region,size_w,size_h,scale_out=1.5,scale_in=0.4):
    h = abs(region[3] - region[1])
    w = abs(region[2] - region[0])
    outer_left_x=max(region[0]-int(w*scale_out/2),0)
    outer_right_x=min(region[2]+int(w*scale_out/2),size_w-1)
    outer_up_y=max(region[1]-int(h*scale_out/2),0)
    outer_down_y=min(region[3]+int(h*scale_out/2),size_h-1)
    in_left_x=region[0]+int(w*scale_in/2)
    in_right_x=region[2]-int(w*scale_in/2)
    in_up_y=region[1]+int(h*scale_in/2)
    in_down_y=region[3]-int(h*scale_in/2)
    seeds=[]
    for i in range(outer_left_x,outer_right_x+1):
        for j in range(outer_up_y,outer_down_y+1):
            if i>in_left_x and i<in_right_x and j>in_up_y and j<in_down_y:
               continue
            else:
               seeds.append([i,j])
    
    return seeds,[outer_left_x,outer_up_y,outer_right_x,outer_down_y],[in_left_x,in_up_y,in_right_x,in_down_y]

def get_candidate_seeds2(region):
    seeds=[]
    for i in range(region[0],region[2]+1):
        for j in range(region[1],region[3]+1):
               seeds.append([i,j])
    
    return seeds

def detect_connect_regions(seed1,seed2,img): #img = ori_img_binary
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    for seedPoint in seed1:
        seed_pt= (seedPoint[0],seedPoint[1]) #38,161
        if img[(seedPoint[1],seedPoint[0])] == 0: #the seed point must be black
            im_floodfill=img.copy()
            mask[:]=0
            _=cv2.floodFill(im_floodfill, mask, seed_pt,100,(0,),(0,),flags=8|cv2.FLOODFILL_FIXED_RANGE)
            cv2.imwrite('floodfill_test.png', im_floodfill)
            
            for s2 in seed2:
              if im_floodfill[(s2[1],s2[0])] == 100: #the filled region has overlap with target box
                 print("find seed2!")
                 return 1
    
    return 0

def get_predicted_regions(input_image="cin_00015.png",input_predict="cin_00015.txt"):
    ori_img = cv2.imread(input_image)
    resImage = ori_img.copy()
    regions=[]
    arrow_regions=[]
    with open(input_predict) as result_fp:
      results = result_fp.readlines()
    
    for line in results:
         #label =line.strip().split('\t')[0]
         #if label != 'circle':
         geo = line.strip().split('\t')[1].split(',')
         geo = [float(x) for x in geo]
         left_x = int(np.floor(min([geo[0],geo[2],geo[4],geo[6]])))
         right_x = int(np.ceil(max([geo[0],geo[2],geo[4],geo[6]])))
         left_y =int(np.floor(min([geo[1],geo[3],geo[5],geo[7]])))  #int(geo[geo.index(str(left_x))+1])
         right_y = int(np.ceil(max([geo[1],geo[3],geo[5],geo[7]])))  #int(geo[geo.index(str(right_x))+1])
         #marked predict results
         box = [left_x, left_y, right_x, right_y]
         if line.strip().split('\t')[0] == "text":
            cv2.rectangle(resImage, (left_x, left_y),
                       (right_x, right_y),
                       (255, 0, 0), thickness=1)
            regions.append(box)
         if line.strip().split('\t')[0] == "arrow":
            cv2.rectangle(resImage, (left_x, left_y),
                    (right_x, right_y),
                    (255, 0, 255), thickness=1)
            arrow_regions.append(box)
    
    cv2.imwrite('marked_image.png', resImage)
    return regions,arrow_regions

def plot_connections(img,file,connect_regions):
        for c in connect_regions:
            region1= c[0][0]
            region2= c[1][0]
            cv2.rectangle(img, (region1[0], region1[1]),
                           (region1[2], region1[3]),
                           (255, 0, 0), thickness=1)
            
            cv2.rectangle(img, (region2[0], region2[1]),
                   (region2[2], region2[3]),
                   (255, 0, 0), thickness=1)
            
            rect1center = ((region1[0]+region1[2])/2, (region1[1]+region1[3])/2)
            rect2center = ((region2[0]+region2[2])/2, (region2[1]+region2[3])/2)
            cv2.line(img, rect1center, rect2center, color=(255,0,0), thickness=1)
            cv2.imwrite(file,img)

if __name__ == '__main__':
    input_image="cin_00090.jpg"
    input_predict = "cin_00090.txt"
    regions,arrow_regions=get_predicted_regions(input_image,input_predict)
    ori_img = cv2.imread(input_image)
    #######test single start##################
    #region1 = regions[10]#regions[4]#regions[21]#regions[25]#regions[10]#regions[20]#regions[10]
    #region2=  regions[31]#regions[5]#regions[31]#regions[36]#regions[21]#regions[38]#regions[31]
    ori_img_binary = binary_image(ori_img.copy())
    cv2.imwrite('binary_'+input_image, ori_img_binary)
    
    connect_regions=[]
    text_centers = [((x[0]+x[2])/2, (x[1]+x[3])/2) for x in regions]
    resImage2=ori_img.copy()
    for arrow_region in arrow_regions:
        acenter = ((arrow_region[0]+arrow_region[2])/2, (arrow_region[1]+arrow_region[3])/2)
        nearest_index = find_nearest_text(acenter,text_centers)
        region_target = regions[nearest_index]
        target_center = text_centers[nearest_index]
        from_assign = []
        from_centers =[]
        vx,vy,x,y=find_lines_inregion(ori_img_binary,arrow_region)
        if vx==0 and vy==0 and x==0 and y==0:
            continue
        for i in range(len(regions)):
            region_from = regions[i]
            from_center = text_centers[i]
            xl,yl,xr,yr = get_convex_box(region_target,region_from)
            if arrow_region[0] >= xl and arrow_region[2] <=xr and arrow_region[1]>=yl and arrow_region[3] <=yr:
                 print("1 one arrow in correct place")
                 if dist(vx,vy,x,y,target_center)<=10 and dist(vx,vy,x,y,from_center)<=10:
                        print("2 arrow line dist is ok!")
                        from_assign.append(regions[i])
                        from_centers.append(from_center)
                        
        
        if from_assign:
           nearest_from_index = find_nearest_text(acenter,from_centers)
           arrow_seeds = get_candidate_seeds2(arrow_region)#[(int(x),int(y))]
           from_seed,out_box2,in_box2 = get_candidate_seeds(from_assign[nearest_from_index],ori_img_binary.shape[1],ori_img_binary.shape[0],scale_out=2)
           
           cv2.rectangle(resImage2, (out_box2[0], out_box2[1]),
                           (out_box2[2], out_box2[3]),
                           (255, 0, 255), thickness=1)
           cv2.rectangle(resImage2, (in_box2[0], in_box2[1]),
                           (in_box2[2], in_box2[3]),
                           (255, 0, 255), thickness=1)
           if detect_connect_regions(arrow_seeds,from_seed,ori_img_binary.copy()):
              print("3 these two regions can be flood")
              connect_regions.append(([region_target],[from_assign[nearest_from_index]]))
    
    cv2.imwrite('marked_image.png', resImage2)
    plot_connections(ori_img.copy(),"connetion_plot_fromarrow.png",connect_regions)
    
#################################
#img_path = "./"
#text=ocr_text_from_image(regions,ori_img,img_path)

'''
# draw rectangle on binary image
idx=1
ori_img = cv2.imread("./cin_00015.png")
resImage2 = ori_img.copy()

for region in regions:
  #if idx > 8  :
  #   break
  
  #if text[idx-1] != u'null\n':
  if text[idx-1]:
     cv2.rectangle(resImage2, (region[0]-2, region[1]-2),
         (region[2]+2, region[3]+2),
         (255, 0, 0), thickness=1)
     # put the detected text to binary image
     cv2.putText(resImage2, text[idx-1].strip().encode('utf-8'), (region[0], region[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0))
  
  idx+=1;

cv2.imwrite("text_labeled_image.png",  resImage2)
###################################################
'''
