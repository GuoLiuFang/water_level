import os
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import blur
import cfg


class random_uniform_num():
    """
    均匀随机，确保每轮每个只出现一次
    """

    def __init__(self, total):
        self.total = total
        self.range = [i for i in range(total)]
        np.random.shuffle(self.range)
        self.index = 0

    def get(self, batchsize):
        r_n = []
        if (self.index + batchsize > self.total):
            r_n_1 = self.range[self.index:self.total]
            np.random.shuffle(self.range)
            self.index = (self.index + batchsize) - self.total
            r_n_2 = self.range[0:self.index]
            r_n.extend(r_n_1)
            r_n.extend(r_n_2)
        else:
            r_n = self.range[self.index: self.index + batchsize]
            self.index = self.index + batchsize
        return r_n


def gen(batch_size=cfg.batch_size, is_val=False):
    if batch_size == 0: # 这里不用等比例了，
        if is_val:
            with open(os.path.join(cfg.data_dir, cfg.val_fname), 'r') as f_val:
                f_list = f_val.readlines()
        else:
            with open(os.path.join(cfg.data_dir, cfg.train_fname), 'r') as f_train:
                f_list = f_train.readlines()
        r_n = random_uniform_num(len(f_list))
        _imagefile = np.array(f_list)
        while 1:
            shufimagefile = _imagefile[r_n.get(batch_size)]
            
            img_filename = str(shufimagefile[0]).strip().split(',')[0]
            img_w, img_h = int(str(shufimagefile[0]).strip().split(',')[1]), int(str(shufimagefile[0]).strip().split(',')[2])
            img_path = os.path.join(cfg.data_dir, cfg.train_image_dir_name, img_filename)
            img = image.load_img(img_path)
            x = np.zeros((batch_size, img_h, img_w, cfg.num_channels), dtype=np.float32)
            pixel_num_h = img_h // cfg.pixel_size  # pixel_size==4
            pixel_num_w = img_w // cfg.pixel_size  # pixel_size==4
            y = np.zeros((batch_size, pixel_num_h, pixel_num_w, 7), dtype=np.float32)

            img = image.img_to_array(img)
            x[0] = preprocess_input(img, mode='tf')
            gt_file = os.path.join(cfg.data_dir,
                                   cfg.train_label_dir_name,
                                   img_filename[:-4] + '_gt.npy')
            y[0] = np.load(gt_file)
            yield x, y
    else:
        img_h, img_w = cfg.max_train_img_size, cfg.max_train_img_size
        if is_val:
            with open(os.path.join(cfg.data_dir, cfg.val_fname), 'r') as f_val:
                f_list = f_val.readlines()
        else:
            with open(os.path.join(cfg.data_dir, cfg.train_fname), 'r') as f_train:
                f_list = f_train.readlines()

        x = np.zeros((batch_size, img_h, img_w, cfg.num_channels), dtype=np.float32)
        pixel_num_h = img_h // cfg.pixel_size  # pixel_size==4
        pixel_num_w = img_w // cfg.pixel_size  # pixel_size==4
        y = np.zeros((batch_size, pixel_num_h, pixel_num_w, 7), dtype=np.float32)

        r_n = random_uniform_num(len(f_list))
        _imagefile = np.array(f_list)
        while 1:
            shufimagefile = _imagefile[r_n.get(batch_size)]
            for i in range(len(shufimagefile)):
                img_filename = str(shufimagefile[i]).strip().split(',')[0]
                img_path = os.path.join(cfg.data_dir, cfg.train_image_dir_name, img_filename)
                img = image.load_img(img_path)
                img = image.img_to_array(img)
                x[i] = preprocess_input(img, mode='tf')
                gt_file = os.path.join(cfg.data_dir,
                                       cfg.train_label_dir_name,
                                       img_filename[:-4] + '_gt.npy')
                y[i] = np.load(gt_file)
            yield x, y


def gen_bootstrap(batch_size=cfg.batch_size, is_val=False, app_blur=False):
    img_h, img_w = cfg.max_train_img_size, cfg.max_train_img_size
    x = np.zeros((batch_size, img_h, img_w, cfg.num_channels), dtype=np.float32)
    pixel_num_h = img_h // cfg.pixel_size
    pixel_num_w = img_w // cfg.pixel_size
    y = np.zeros((batch_size, pixel_num_h, pixel_num_w, 7), dtype=np.float32)
    if is_val:
        with open(os.path.join(cfg.data_dir, cfg.val_fname), 'r') as f_val:
            f_list = f_val.readlines()
    else:
        with open(os.path.join(cfg.data_dir, cfg.train_fname), 'r') as f_train:
            f_list = f_train.readlines()
    while True:
        for i in range(batch_size):
            # random gen an image name
            random_img = np.random.choice(f_list)
            img_filename = str(random_img).strip().split(',')[0]
            # load img and img anno
            img_path = os.path.join(cfg.data_dir,
                                    cfg.train_image_dir_name,
                                    img_filename)
            img = image.load_img(img_path)
            if app_blur:
                img, blur_type = blur.apply_random_blur(img)
            img = image.img_to_array(img)
            x[i] = preprocess_input(img, mode='tf')
            gt_file = os.path.join(cfg.data_dir,
                                   cfg.train_label_dir_name,
                                   img_filename[:-4] + '_gt.npy')
            y[i] = np.load(gt_file)
        yield x, y
