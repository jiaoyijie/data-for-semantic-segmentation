from nets.pspnet import mobilenet_pspnet
from PIL import Image
import numpy as np
import random
import copy
import os

def letterbox_image(image, size):  # 将图像设置统一大小

    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh),Image.BICUBIC)
    new_image = Image.new('RGB', size, (0,0,0))
    new_image.paste(image,((w-nw)//2, (h-nh)//2))
    return new_image, nw, nh

random.seed(0)
class_colors = [[random.randint(0,255),random.randint(0,255),random.randint(0,255)] for _ in range(41)]
NCLASSES = 41
HEIGHT = 512
WIDTH = 512

model = mobilenet_pspnet(n_classes=NCLASSES,input_height=HEIGHT, input_width=WIDTH)
model.load_weights("logs/ep011-loss0.039-val_loss0.066.h5")
imgs = os.listdir("E:/Coding/Semantic-Segmentation-master/pspnet_Multi_Mobile/img")

for jpg in imgs:

    img = Image.open("E:/Coding/Semantic-Segmentation-master/pspnet_Multi_Mobile/img/"+jpg)
    img = letterbox_image(img,HEIGHT,WIDTH)
    old_img = copy.deepcopy(img)
    orininal_h = np.array(img).shape[0]
    orininal_w = np.array(img).shape[1]

    img = img.resize((WIDTH,HEIGHT))
    img = np.array(img)
    img = img/255
    img = img.reshape(-1,HEIGHT,WIDTH,3)
    pr = model.predict(img)[0]

    pr = pr.reshape((int(HEIGHT/4), int(WIDTH/4),NCLASSES)).argmax(axis=-1)

    seg_img = np.zeros((int(HEIGHT/4), int(WIDTH/4),3))
    colors = class_colors

    for c in range(NCLASSES):
        seg_img[:,:,0] += ( (pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
        seg_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
        seg_img[:,:,2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8')

    seg_img = Image.fromarray(np.uint8(seg_img)).resize((orininal_w,orininal_h))

    image = Image.blend(old_img,seg_img,0.3)
    image.save("E:/Coding/Semantic-Segmentation-master/pspnet_Multi_Mobile/img_out/"+jpg)


