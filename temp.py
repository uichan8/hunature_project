# -*- coding: utf-8 -*-
import cv2
import numpy as np
from time import time

model_path = 'DetectionModels/yolox_tiny/yolox_tiny'
device_name = 'MYRIAD'

net = cv2.dnn.readNet('DetectionModels/model/model.xml', 'DetectionModels/model/model.bin')
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

while True:
    start_time = time()
    blob = np.zeros([1,3,224,224])
    net.setInput(blob)
    out = net.forward()
    print(1000*(time() - start_time))

print("end_precess")