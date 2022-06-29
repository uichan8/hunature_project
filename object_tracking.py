from openvino.inference_engine import IECore
from time import time

import argparse
import cv2
import numpy as np

from DetectionTools.preprocess import preprocess
from DetectionTools.demo_process import demo_process
from DetectionTools.nms import multi_nms

from TrackingTools.byte_tracker import BYTETracker

from visualize import visualize, plot_tracking

def make_parser():
    parser = argparse.ArgumentParser("Counting algorithm")
    #path args
    parser.add_argument(
        "-m", "--model_path",
        type=str,
        default="DetectionModels/yolox_tiny",
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

def is_in_line(centroid, line, margin):
    print(centroid, line)
    if (centroid[1] > line[0][1] - margin) and (centroid[1] < line[0][1] + margin):
        return True
    return False

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

    #if raspberry-pi is not working, use this to resize output window size.
    #_, frame = cap.read()
    #cv2.imshow("img", frame)
    #cv2.waitKey(5000)

    #NCS2 setting
    ie = IECore()
    model      = ie.read_network(model= model_path+'.xml', weights = model_path+'.bin')
    network    = ie.load_network(network=model, device_name='MYRIAD')
    input_key  = list(network.input_info)[0]
    output_key = list(network.outputs.keys())[0]

    #main code
    tracker = BYTETracker(track_thresh, track_buffer, match_thresh,frame_rate = fps)
    frame_id = 0
    process_fps = 0
    appear_flags = {}
    count = 0
    while True:
        start = time()
        #read image from camera
        ret_val, frame = cap.read()
        if ret_val == False:
            break
    
        #object detection part
        #process image
        img,ratio = preprocess(frame,input_shape)
        img = img[np.newaxis, ...]
    
        #inference
        output = network.infer(inputs={input_key: img})
        output = output[output_key]
    
        #demo processing
        predict = demo_process(output,input_shape)[0]
    
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
        tracking_result = multi_nms(boxes_xyxy, scores, nms_thr, score_thr)

        # Define two lines for counting objects
        line = [((0, int(0.2 * height)), (int(width), int(0.2 * height))),
                    ((0, int(0.8 * height)), (int(width), int(0.8 * height)))]
    
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

            # Count the objects
            if is_in_line(centroid, line[0], margin=10):
                if tid in appear_flags:
                    if appear_flags[tid] == 2:
                        count += 1
                        del appear_flags[tid]
                appear_flags[tid] = 1
                    
            if is_in_line(centroid, line[1], margin=10):
                if tid in appear_flags:
                    if appear_flags[tid] == 1:
                        count += 1
                        del appear_flags[tid]
                appear_flags[tid] = 2
    
        #print_image
        frame = plot_tracking(frame, online_tlwhs, online_ids, online_centroids, count, frame_id = frame_id + 1,fps = process_fps)
        cv2.line(frame, line[0][0], line[0][1], (0, 255, 0), 2)
        cv2.line(frame, line[1][0], line[1][1], (0, 255, 0), 2)
        cv2.imshow("img", frame)

        if cv2.waitKey(1) == 27:
            break
    
        #information update
        frame_id += 1
        end = time()
        process_fps = 1/(end-start)

if __name__ == '__main__':
    args = make_parser().parse_args()
    main(args)