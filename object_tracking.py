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

#python3 object_tracking.py -i 640 -m 'DetectionModels/model/model' -s 0.1

def make_parser():
    parser = argparse.ArgumentParser("Counting algorithm")
    #setting args
    parser.add_argument(
        "-m", "--model_path",
        type=str,
        default="DetectionModels/yolox_tiny/yolox_tiny",
        help="모델의 확장자를 제외한 경로를 입력해주세요."
    )
    parser.add_argument(
        "-v","--video_path",
        type=str,
        default='0',
        help="비디오를 입력으로 넣고싶은 경우 경로를 입력해 주세요, 입력안하면 카메라 정보가 입력으로 들어갑니다."
    )
    parser.add_argument(
        "-o", "--output_path",
        type=str,
        default=None,
        help="저장을 원하면 경로를 입력해주세요"
    )

    #detection args
    parser.add_argument(
        "-s", "--score_thr",
        type=float,
        default=0.3,
        help="detection 에서 confidence_score의 입계값을 입력해 주세요"
    )
    parser.add_argument(
        "-n", "--nms_thr",
        type=float,
        default=0.5,
        help="NMS 임계값을 입력해 주세요"
    )
    parser.add_argument(
        "-i", "--input_shape",
        type=int,
        default=0,
        help="이미지 입력 사이즈를 입력해주세요"
    )
    parser.add_argument(
        "-c", "--class_num",
        type=int,
        default=8,
        help="클래스 수를 입력해 주세요"
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
        help="video fps를 입력해주세요"
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
    input_shape = (args.input_shape,args.input_shape)
    score_thr   = args.score_thr
    nms_thr     = args.nms_thr
    class_num   = args.class_num

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
    out = cv2.VideoWriter(saving_path, cv2.VideoWriter_fourcc(*'DIVX'), 30, (width, height))

    #NCS2 setting
    ie = IECore()
    model      = ie.read_network(model= model_path+'.xml', weights = model_path+'.bin')
    network    = ie.load_network(network=model, device_name='MYRIAD')
    input_key  = list(network.input_info)[0]
    output_key = list(network.outputs.keys())[0]

    #main code
    tracker      = BYTETracker(track_thresh, track_buffer, match_thresh,frame_rate = fps)
    frame_id     = 0
    count        = 0
    start = time()
    
    while True:
        #read image from camera
        ret_val, frame = cap.read()
        if ret_val == False:
            break
    
        #object detection part
        #process image
        img,ratio = preprocess(frame,input_shape,yolo_v5 = True)
        img = img[np.newaxis, ...]
    
        #inference
        output = network.infer(inputs={input_key: img})
        output = output[output_key]

        #demo processing
        predict = output.reshape(-1,class_num + 5)

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

        #print_image
        frame = plot_tracking(frame, online_tlwhs, online_ids, online_centroids, count, fps)
        cv2.imshow("img", frame)
        if cv2.waitKey(1) == 27:
            break

        #save_video
        if saving_path != None:
            out.write(frame)

        
        #information update
        frame_id += 1
    out.release()

if __name__ == '__main__':
    args = make_parser().parse_args()
    main(args)