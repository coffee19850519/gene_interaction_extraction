import os


#train_task_id = '3T'+str(choice([1024, 1280, 1536, 1792, 2048]))
max_train_img_size = 736
train_task_id = '3T' +str(max_train_img_size)
DEBUG = True

#train_task_id = '3T1500'
initial_epoch = 0
epoch_num = 200
lr = 1e-4
decay = 5e-5
# clipvalue = 0.5  # default 0.5, 0 means no clip
patience = 40
load_weights = True
lambda_inside_score_loss = 3.0
lambda_cls_loss = 3.0
lambda_side_vertex_code_loss = 1.0
lambda_side_vertex_coord_loss = 1.0


total_img = 120
validation_split_ratio = 0.25

#max_train_img_size = int(train_task_id[-4:])
max_predict_img_size = max_train_img_size#int(train_task_id[-4:])  # 2400

assert max_train_img_size in [512, 736, 1024, 1280, 1536, 1792, 2048], \
    'max_train_img_size must in [1024, 1280, 1536, 1792, 2048]'
if max_train_img_size == 512:
    batch_size = 20
elif max_train_img_size == 736:
    batch_size = 10
elif max_train_img_size == 1024:
    batch_size = 5
elif max_train_img_size == 1280:
    batch_size = 4
elif max_train_img_size == 1536:
    batch_size = 3
elif max_train_img_size == 1792:
    batch_size = 2
else:
    batch_size = 1

steps_per_epoch = total_img * (1 - validation_split_ratio) // batch_size
validation_steps = total_img * validation_split_ratio // batch_size

data_dir = r'C:\Users\LSC-110\Desktop\cxtest\Images'
origin_image_dir_name = r'inhibit'
origin_txt_dir_name = r'labels'
train_image_dir_name = r'images_%s' % train_task_id
train_label_dir_name = r'labels_%s' % train_task_id
show_gt_image_dir_name = r'show_gt_images_%s' % train_task_id
show_act_image_dir_name = r'show_act_images_%s' % train_task_id
val_fname = 'val_%s.txt' % train_task_id
train_fname = 'train_%s.txt' % train_task_id
# in paper it's 0.3, maybe to large to this problem
shrink_ratio = 0.2
# pixels between 0.2 and 0.6 are side pixels
shrink_side_ratio = 0.8
epsilon = 1e-4

num_channels = 3
feature_layers_range = range(5, 1, -1)
# feature_layers_range = range(3, 0, -1)
feature_layers_num = len(feature_layers_range)
# pixel_size = 4
pixel_size = 2 ** feature_layers_range[-1]

locked_layers = True

if not os.path.exists('saved_model'):
    os.mkdir('saved_model')


saved_model_weights_file_path = r'saved_model\east_model_weights.h5'


text_pixel_threshold = 0.8
action_pixel_threshold = 0.9
text_side_vertex_pixel_threshold = 0.8
action_side_vertex_pixel_threshold = 0.85
text_trunc_threshold = 0.2
nock_trunc_threshold = 0.05
arrow_trunc_threshold = 0.9

predict_cut_text_line = False
predict_write2txt = True

dictionary_file = r'D:\Study\Master\1st deep learning project\Dima data(2)\Dima data\dictionary.xlsx'

#OCR configuration
TERSSERACT_PATH = r'D:\python\Scripts\pytesseract.exe'
TERSSERACT_CONFIG = '-l eng+equ+osd  --oem 1 --psm 3 bazaar'
crop_image_path = r''
OCR_SCALE = 10
OCR_OFFSET = 2
#pipline configuration
'''
image_path = r'D:\test'
OCR_result_path = r'D:\Study\Master\1st deep learning project\Dima data(2)\Dima data\imagetest\ocr_txt'
predict_result_path = r'D:\Study\Master\1st deep learning project\Dima data(2)\Dima data\predict_txt'
result_in_json_path = r'D:\Study\Master\1st deep learning project\Dima data(2)\Dima data\imagetest\tolabelme_txt'
'''
#image_path = r'C:\Users\LSC-110\Desktop\cxtest\Images'
OCR_result_path = r'D:\test\ocr_txt'
predict_result_path = r'D:\test\predict_txt'
result_in_json_path = r'D:\test\tolabelme_txt'
#OCR path
OCR_subimage_path=r'D:\subimage'
correct_result_path=r'D:\test\correct_txt'
#pair gene relation saved in Json file
final_json_path=r'D:\test\pair_gene_json_test'
