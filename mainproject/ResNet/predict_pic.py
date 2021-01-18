import numpy as np
from tensorflow.keras.preprocessing import image
from ResNet.resnet50_model import ResNet50
import matplotlib.pyplot as plt
from PIL import Image


#  loss: 0.0512 - accuracy: 0.9859 - val_loss: 0.1399 - val_accuracy: 0.9531
class Prediction(object):
    def __init__(self, ModelFile, PredictFile):
        self.modelfile = ModelFile
        self.predict_file = PredictFile

    @staticmethod
    def invert_dict(invert):
        return dict([(v, k) for k, v in invert.items()])

    def Predict(self):
        realdic = {'哈士奇': 0, '德国牧羊犬': 1, '拉布拉多犬': 2, '柯基犬': 3, '沙皮犬': 4, '秋田犬': 5,
                   '英国牧羊犬': 6, '萨摩耶犬': 7, '藏獒': 8, '金毛': 9}
        realdic = self.invert_dict(realdic)

        model = ResNet50(classes=10)
        model.load_weights(self.modelfile, by_name=True)

        img = image.load_img(self.predict_file, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        preds = model.predict(x, batch_size=16)

        index = np.argmax(preds)
        # print("predict result is: ", realdic[index])
        # print(preds[0][index])
        probability = "%.2f%%" % (preds[0][index] * 100)
        return realdic[index], probability

    def ShowPredImg(self):
        img = Image.open(self.predict_file)
        plt.imshow(img)
        plt.show()

# Pred = Prediction(PredictFile="E:/01Jcsim/01project/Python/dogtrain/mainproject/ResNet/test_Img/1913.jpg",
#                   ModelFile="E:/01Jcsim/01project/Python/dogtrain/mainproject/ResNet/resnet50_best.h5")
# # Pred.ShowPredImg()
# result, prob = Pred.Predict()
# print(result)
# print(prob)
