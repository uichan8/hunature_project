import os
from openvino.inference_engine import IECore
from time import time, sleep

import argparse
import cv2
import numpy as np

from DetectionTools.preprocess import preprocess
from DetectionTools.demo_process import demo_process
from DetectionTools.nms import multi_nms

from TrackingTools.byte_tracker import BYTETracker

from Utils.visualize import visualize, plot_tracking

def make_parser():
    parser = argparse.ArgumentParser("Counting algorithm")
    #path args

    parser.add_argument(
        "-m", "--model_path",
        type=str,
        default="DetectionModels/yolox_tiny/yolox_tiny",
        help="Input your onnx model path without 확장자."
    )
    parser.add_argument(
        "-v","--video_path",
        type=str,
        default='0',
        help="Input your video path."
    )
    parser.add_argument(
        "-o", "--output_path",
        type=str,
        default='0',
        help="Path to your output directory."
    )
    #detection args
    parser.add_argument(
        "-s", "--score_thr",
        type=float,
        default=0.5,
        help="Score threshold to filter the result."
    )
    parser.add_argument(
        "-n", "--nms_thr",
        type=float,
        default=0.7,
        help="NMS threshold."
    )
    parser.add_argument(
        "-i", "--input_shape",
        type=tuple,
        default=(416,416),
        help="Specify an input shape for inference."
    )
    #tracker args
    parser.add_argument(
        "--track_thr",
        type=float,
        default=0.5, 
        help="tracking confidence threshold"
    )
    parser.add_argument(
        "--match_thr",
        type=float,
        default=0.5,
        help="matching threshold for tracking"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30, 
        help="video fps"
    )
    parser.add_argument(
        "--track_buffer", 
        type=int, 
        default=30, 
        help="the frames for keep lost tracks"
    )
    parser.add_argument(
        "--min_box_area",
        type=float, 
        default=10, 
        help='filter out tiny boxes'
    )
    return parser

def main(args):
    #PATH args
    model_path  = args.model_path
    video_path  = 0 if args.video_path == '0' else args.video_path
    saving_path = args.output_path

    #detection args
    classes     = ["person", "bicycle", "car"]
    input_shape = args.input_shape
    score_thr   = args.score_thr
    nms_thr     = args.nms_thr

    #tracking args
    fps = args.fps
    track_thresh = args.track_thr
    match_thresh = args.match_thr
    track_buffer = args.track_buffer
    min_box_area = args.min_box_area

    #camera setting
    cap    = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #NCS2 setting
    ie = IECore()
    model      = ie.read_network(model= model_path+'.xml', weights = model_path+'.bin')
    network    = ie.load_network(network=model, device_name='MYRIAD')
    input_key  = list(network.input_info)[0]
    output_key = list(network.outputs.keys())[0]

    #main code
    start = time()
    tracker      = BYTETracker(track_thresh, track_buffer, match_thresh,frame_rate = fps)
    frame_id     = 0
    count        = 0

    
    while True:
        #read image from camera
        s = time()

        ret_val, frame = cap.read()
        if ret_val == False:
            break

        print(f"capture      : {int((time()-s)*1000)}")
        s = time()
    
        #object detection part
        #process image
        img,ratio = preprocess(frame,input_shape)
        img = img[np.newaxis, ...]

        print(f"preprocess   : {int((time()-s)*1000)}")
        s = time()
    
        #inference
        output = network.infer(inputs={input_key: img})
        output = output[output_key]

        print(f"infer        : {int((time()-s)*1000)}")
        s = time()

        #demo processing
        predict = demo_process(output,input_shape)[0]

        boxes = predict[:, :4]
        scores = predict[:, 4:5] * predict[:, 5:8]

        print(f"demo process : {int((time()-s)*1000)}")
        s = time()
    
        #xywh2xyxy
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio

        print(f"change boxes : {int((time()-s)*1000)}")
        s = time()
    
        #NMS
        tracking_result = multi_nms(boxes_xyxy, scores, nms_thr, score_thr)

        print(f"nms          : {int((time()-s)*1000)}")
        s = time()

        #update fps
        end = time()
        fps = 1/(end - start)
        start = time()
    
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
            if tlwh[2] * tlwh[3] > min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
                online_centroids.append(centroid)
        
        #count
        online_ids = np.array(online_ids)
        if online_ids.shape[0] != 0 and count < online_ids.max():
            count = online_ids.max()

        print(f"TRACK        : {int((time()-s)*1000)}")
        s = time()


        #print_image
        frame = plot_tracking(frame, online_tlwhs, online_ids, online_centroids, count, fps)
        cv2.imshow("img", frame)
        if cv2.waitKey(1) == 27:
            break

        print(f"image print  : {int((time()-s)*1000)}")
        s = time()
        
        #information update
        frame_id += 1

        #clear terminal
        
        sleep(4)
        os.system("clear")

if __name__ == '__main__':
    args = make_parser().parse_args()
    main(args)