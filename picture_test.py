from openvino.inference_engine import IECore
from time import time

import argparse
import cv2
import numpy as np

from DetectionTools.preprocess import preprocess
from DetectionTools.demo_process import demo_process
from DetectionTools.nms import multi_nms

from TrackingTools.byte_tracker import BYTETracker

from Utils.visualize import visualize, plot_tracking


def detection(img):
    img2 = np.copy(img)
    print(img.shape)
    width = img.shape[0]
    height = img.shape[1]

    ie = IECore()
    model_path = "DetectionModels/yolov5s/yolov5s"
    model      = ie.read_network(model= model_path+'.xml', weights = model_path+'.bin')
    network    = ie.load_network(network=model, device_name='MYRIAD')
    input_key  = list(network.input_info)[0]
    output_key = list(network.outputs.keys())[0]
    
    tracker      = BYTETracker(0.1, 10, 0.1,frame_rate = 30)

    img,ratio = preprocess(img,(320,320),yolo_v5 = True)
    img = img[np.newaxis, ...]
                
    output = network.infer(inputs={input_key: img})
    output = output[output_key]

    predict = output.reshape(-1,8)

    boxes = predict[:, :4]
    scores = predict[:, 4:5] * predict[:, 5:8]

    #xywh2xyxy
    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
    boxes_xyxy /= ratio
    
    #NMS
    tracking_result = multi_nms(boxes_xyxy, scores, 0.5, 0.5)
    #print(tracking_result)

    #Update the tracklets
    online_targets = tracker.update(tracking_result[:, :-1], [height, width], [height, width])
    online_tlwhs = []
    online_ids = []
    online_scores = []
    online_centroids = []
    for t in online_targets:
        tlwh = t.tlwh
        tid = t.track_id
        centroid = t.tlwh_to_xyah(tlwh)[:2]
        vertical = tlwh[2] / tlwh[3] > 1.6
        if tlwh[2] * tlwh[3] > 10 and not vertical:
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            online_scores.append(t.score)
            online_centroids.append(centroid)

    return plot_tracking(img2, online_tlwhs, online_ids, online_centroids, 1, 1)


if __name__ == "__main__":
    img = cv2.imread("Pedestrian.jpg")
    frame = detection(img)
    cv2.imwrite("result.jpg", frame)
