from djjava ango.shortcuts import render
from django.http import HttpResponse
from ResNet.predict_pic import Prediction
import random
# Create your views here.


def up(request):
    myFile = request.FILES.get("img")
    ran_name = str(random.randint(1000, 9999)) + '.jpg'
    destination = open("E:/01Jcsim/01project/Python/dogtrain/mainproject/static/predic_Img/" + ran_name, 'wb+')  # 打开特定的文件进行二进制的写操作
    # 分块写入文件
    for chunk in myFile.chunks():
        destination.write(chunk)
    destination.close()
    return ran_name


def pred(request):
    if request.method == "POST":
        img_name = up(request)
        # 调用神经网络
        Pred = Prediction(PredictFile=r"E:/01Jcsim/01project/Python/dogtrain/mainproject/static/predic_Img/"+img_name,
                          ModelFile="E:/01Jcsim/01project/Python/dogtrain/mainproject/ResNet/resnet50_best.h5")
        # 预测结果
        result, probability = Pred.Predict()
        # print(result)
        return render(request, 'result.html', {"data": result, "img": img_name, "probability": probability})
    else:
        return HttpResponse('please visit us with post')


def index(request):
    return render(request, 'uploadimg.html')




