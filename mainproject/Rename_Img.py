import os
import tensorflow as tf

def FileRename(FilePath,Dogtype ):
    count = 0
    for types in Dogtype:
        file_counter = 0
        subfolder = os.listdir(FilePath+types)
        # print(subfolder)
        for subclass in subfolder:
            print(subclass)
            file_counter += 1
            os.renames(FilePath+types+"/"+subclass,
                       FilePath+types+"/"+str(count)+"_"+str(file_counter)+'__'+str(types)+".jpg")
        count += 1


if __name__ == '__main__':
    # Dogtype = ["哈士奇", "德国牧羊犬", "拉布拉多犬", "柯基犬", "沙皮犬", "秋田犬", "英国牧羊犬", "萨摩耶犬", "藏獒", "金毛"]
    # FileRename(FilePath="/ResNet/train_Img/", Dogtype=Dogtype)
    version = tf.__version__
    # ok = tf.test.is_gpu_available()
    print("version=", version)


