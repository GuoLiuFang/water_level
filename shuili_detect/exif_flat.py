# -*- coding: utf-8 -*-
from PIL import Image, ExifTags, ImageOps
from tqdm import tqdm
from glob import glob
import cfg
import os
import warnings

def apply_exif_orientation(image):
    try:
        exif = image._getexif()
    except AttributeError:
        exif = None

    if exif is None:
        return False

    exif = {
        ExifTags.TAGS[k]: v
        for k, v in exif.items()
        if k in ExifTags.TAGS
    }

    orientation = exif.get('Orientation', None)

    if orientation == 1:
        # do nothing
        return False
    elif orientation == 2:
        # left-to-right mirror
        return ImageOps.mirror(image)
    elif orientation == 3:
        # rotate 180
        return image.transpose(Image.ROTATE_180)
    elif orientation == 4:
        # top-to-bottom mirror
        return ImageOps.flip(image)
    elif orientation == 5:
        # top-to-left mirror
        return ImageOps.mirror(image.transpose(Image.ROTATE_270))
    elif orientation == 6:
        # rotate 270
        return image.transpose(Image.ROTATE_270)
    elif orientation == 7:
        # top-to-right mirror
        return ImageOps.mirror(image.transpose(Image.ROTATE_90))
    elif orientation == 8:
        # rotate 90
        return image.transpose(Image.ROTATE_90)
    else:
        return False

if __name__ == '__main__':
    if cfg.PIL_TRANS:
        # 这里消除由于PiL未读取元信息造成的误差
        print('PIL IMAGE TRANS')
        print(cfg.origin_image_dir_name)
        for img_path in tqdm(glob(os.path.join(cfg.data_dir+ cfg.origin_image_dir_name, '*.jpg'))):
            try:
                img = Image.open(img_path)
                # print(img_path)
                img_flated = apply_exif_orientation(img)
                # print(img_flated)
                if img_flated:
                    # print(img_flated)
                    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
                    img_flated.save(img_path, "JPEG", quality=100)
                    print(img_path)
            except AttributeError as e:
                # print(img_path)
                print(e)
                # raise
            except Exception as e:
                print(img_path)
                print(e)
                raise
