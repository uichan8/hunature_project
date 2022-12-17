from openvino.inference_engine import IECore
from time import time

import argparse
import cv2
import numpy as np
import os

from DetectionTools.preprocess import preprocess
from DetectionTools.demo_process import demo_process
from DetectionTools.nms import multi_nms

from TrackingTools.byte_tracker import BYTETracker

from Utils.visualize import visualize, plot_tracking
from tqdm import tqdm

# USAGE
# from camera
# $ python3 object_tracking.py -i 320 -m 'DetectionModels/yolov5s_320/yolov5s_320'
#
# from video
# $ python3 object_tracking.py -i 640 -m 'DetectionModels/yolov5s_640/yolov5s_640' -v "station.mpg" -o "result/station.mpg" --video_sampling "1" --fps "30"

def main(args):
    #env args
    device      = args.device 
    model_path  = args.model_path

    #detection args
    input_shape = (int(args.input_shape),int(args.input_shape))
    score_thr   = args.score_thr
    nms_thr     = args.nms_thr
    class_num   = args.class_num

    #tracking args
    fps = args.fps
    track_thresh = args.track_thr
    match_thresh = args.match_thr
    track_buffer = args.track_buffer
    min_box_area = args.min_box_area
    
    #NCS2 setting
    ie = IECore()
    model      = ie.read_network(model= model_path+'.xml', weights = model_path+'.bin')
    network    = ie.load_network(network=model, device_name=device)
    input_key  = list(network.input_info)[0]
    output_key = list(network.outputs.keys())[0]
    
    #main code
    tracker      = BYTETracker(track_thresh, track_buffer, match_thresh,frame_rate = fps)
    frame_id     = 0
    count        = 0
    start = time()

    #mask
    mask = cv2.imread("mask.png")[:,:,0]
    count_set = set([])
    
    for img_path in tqdm(np.sort(os.listdir("vidf"))):
        for f in np.sort(os.listdir(os.path.join("vidf",img_path))):
            #object detection part
            #process image
            frame = cv2.imread(os.path.join("vidf",img_path,f))
            img,ratio = preprocess(frame,input_shape,yolo_v5 = True)
            img = img[np.newaxis, ...]
            width = frame.shape[1]
            height = frame.shape[0]

        
            #inference
            output = network.infer(inputs={input_key: img})
            output = output[output_key].astype(np.float64)

            #demo processing
            predict = output.reshape(-1,6)
            boxes = predict[:, :4]
            scores = predict[:, 4:5]
        
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

            #real_count
            for i in range(len(online_ids)):
                if mask[int(online_centroids[i][1]),int(online_centroids[i][0])]:
                    count_set.add(online_ids[i])


            r_count = len(count_set)

            #information update
            frame_id += 1

            frame = plot_tracking(frame, online_tlwhs, online_ids, online_centroids, r_count, fps)
            f_name = str(10000000 + frame_id)[1:]
            cv2.imwrite(os.path.join("result",f_name + ".jpg"),frame)

            #print_image
            # cv2.imshow("img", frame)
            # if cv2.waitKey(1) == 27:
            #     break

            


def make_parser():
    parser = argparse.ArgumentParser("Counting algorithm")
    #setting args
    parser.add_argument(
        "-d", "--device",
        type=str,
        default="MYRIAD",
        help="CPU or MYRIAD (ncs2)"
    )
    parser.add_argument(
        "-m", "--model_path",
        type=str,
        default="DetectionModels/hunature_val_pede/hunature_val_pede",
        help="모델의 확장자를 제외한 경로를 입력해주세요."
    )
    parser.add_argument(
        "-v","--video_path",
        type=str,
        default='0',
        help="비디오를 입력으로 넣고싶은 경우 경로를 입력해 주세요, 입력값이 없으면 카메라 정보가 입력으로 들어갑니다."
    )
    parser.add_argument(
        "-o", "--output_path",
        type=str,
        default=None,
        help="저장을 원하면 경로를 입력해주세요"
    )
    parser.add_argument(
        "--video_sampling",
        type=int,
        default= 1,
        help="비디오에서 프레임을 몇단위로 샘플링 할 것인지를 뜻합니다. 30프레임 -> 10프레임 으로 할경우 30 // 10 = 3을 입력해야 합니다."
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
        default=320,
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

if __name__ == '__main__':
    args = make_parser().parse_args()
    main(args)