import os
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Nadam
import cfg
from network import East
from losses import quad_loss
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
#from random import choice

def gen(batch_size=cfg.batch_size, is_val=False):
    img_h, img_w = cfg.max_train_img_size, cfg.max_train_img_size
    x = np.zeros((batch_size, img_h, img_w, cfg.num_channels), dtype=np.float32)
    pixel_num_h = img_h // cfg.pixel_size
    pixel_num_w = img_w // cfg.pixel_size
    y = np.zeros((batch_size, pixel_num_h, pixel_num_w, 15), dtype=np.float32)
    if is_val:
        with open(os.path.join(cfg.train_data_dir, cfg.val_fname), 'r') as f_val:
            f_list = f_val.readlines()
    else:
        with open(os.path.join(cfg.train_data_dir, cfg.train_fname), 'r') as f_train:
            f_list = f_train.readlines()
    while True:
        for i in range(batch_size):
            # random gen an image name
            random_img = np.random.choice(f_list)
            img_filename = str(random_img).strip().split(',')[0]
            # load img and img anno
            img_path = os.path.join(cfg.train_data_dir,
                                    cfg.train_image_dir_name,
                                    img_filename)
            img = image.load_img(img_path)
            img = image.img_to_array(img)
            x[i] = preprocess_input(img, mode='tf')
            gt_file = os.path.join(cfg.train_data_dir,
                                   cfg.train_label_dir_name,
                                   img_filename[:-4] + '_gt.npy')
            y[i] = np.load(gt_file)
        yield x, y


def  train_model():
  east = East()
  east_network = east.east_network()
  east_network.summary()

  east_network.compile(loss=quad_loss, optimizer=Nadam(lr=cfg.lr,
                                                      # clipvalue=cfg.clipvalue,
                                                      schedule_decay=cfg.decay))

  # load pre-trained model
  if cfg.load_weights and os.path.exists(cfg.saved_model_weights_file_path):
    east_network.load_weights(cfg.saved_model_weights_file_path, by_name=True,
                              skip_mismatch=True)

  print('start training task:' + cfg.train_task_id + '....................')

  # train on current scale data
  east_network.fit_generator(generator=gen(),
                             steps_per_epoch=cfg.steps_per_epoch,
                             epochs=cfg.epoch_num,
                             validation_data=gen(is_val=True),
                             validation_steps=cfg.validation_steps,
                             verbose=2,
                             initial_epoch=cfg.initial_epoch,
                             callbacks=[
                               EarlyStopping(patience=cfg.patience, verbose=2),
                               ModelCheckpoint(
                                   filepath=cfg.saved_model_weights_file_path,
                                   save_best_only=True,
                                   save_weights_only=True,
                                   verbose=1)])
  del east_network,east


if __name__ == '__main__':

  # for size in [1024, 2048, 1280, 1792, 1536]:
  #
  #   cfg.max_train_img_size = size
  #   cfg.train_task_id = '3T' + str(cfg.max_train_img_size)
  #   cfg.max_predict_img_size = cfg.max_train_img_size  # int(train_task_id[-4:])  # 2400
  #
  #   if cfg.max_train_img_size == 1024:
  #     cfg.batch_size = 5
  #   elif cfg.max_train_img_size == 1280:
  #     cfg.batch_size = 4
  #   elif cfg.max_train_img_size == 1536:
  #     cfg.batch_size = 3
  #   elif cfg.max_train_img_size == 1792:
  #     cfg.batch_size = 2
  #   else:
  #     cfg.batch_size = 1
  #
  #   cfg.train_image_dir_name = r'images_%s' % cfg.train_task_id
  #   cfg.train_label_dir_name = r'labels_%s' % cfg.train_task_id
  #   cfg.show_gt_image_dir_name = r'show_gt_images_%s' % cfg.train_task_id
  #   cfg.show_act_image_dir_name = r'show_act_images_%s' % cfg.train_task_id
  #   cfg.val_fname = 'val_%s.txt' % cfg.train_task_id
  #   cfg.train_fname = 'train_%s.txt' % cfg.train_task_id
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    train_model()




