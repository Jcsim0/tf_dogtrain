import glob
import os
import shutil
import random


def get_train_parent():
    return "E:/01Jcsim/01project/Python/dogtrain/mainproject/ResNet/"


data_root = get_train_parent()+"train_Img/"
files = os.listdir(data_root)
for file in files:
    train_fpaths = glob.glob(data_root + file+"/*.jpg")
    random.shuffle(train_fpaths)

    val_fpaths = train_fpaths[int(0.9 * len(train_fpaths)):]
    print('val_fpaths-val data number:', len(val_fpaths))

    # split train val
    for path in val_fpaths:
        val_path = path.replace('train_Img', 'valid_Img')
        fold = val_path.split("\\")[0]
        name = val_path.split("\\")[1]
        if not os.path.exists(fold):
            os.makedirs(fold)
        print(fold, "--------", name)
        shutil.move(path, fold+"/"+name)





