# README.md

# 소개

다중 객체 추적(Multi Object Tracking, MOT)은 연속된 프레임 속에서 다중 객체에 대한 bounding box와 ID를 지속적으로 추적하는 것을 목표로 합니다. 대부분의 방법은 신뢰도(confidence score)가 임계값(threshold)보다 높게 검출된 객체의 bounding box를 연결하여 ID를 부여합니다.

해당 시스템은 Raspberry Pi 4 Model B (RPi4B) 와 Intel Neural Compute Stick 2 (NCS2) 상에서 구동을 확인했습니다.
추가적으로 NCS2를 활용하기 위한 OpenVINO toolkit 그리고 ByteTrack 알고리즘을 이용합니다.

## YOLO X

- **YOLOX는 기본적으로 1 Stage Detector로 Input - Backbone - Neck - Dense Prediction의 구조를 가진다.**
- **YOLOX는 Darknet53의 Backbone을 통해 Feature Map을 추출하며, SPP Layer를 통해 성능을 개선한다.**
- **FPN을 통해 Multi-Scale Feature Map을 얻고 이를 통해 작은 해상도의 Feature Map에서는 큰 Object를 추출하고 큰 해상도의 Feature Map에서는 작은 Object를 추출하게끔 한 Neck 구조를 차용하였다.**
- **최종적으로 Head 부분에서는 기존 YOLOv3와 달리 Decoupled Head를 사용하였다.**

-**Feature Pyramid Network**는 임의의 크기의 **single-scale** 이미지를 convolutional network에 입력하여 다양한 scale의 feature map을 출력하는 네트워크입니다. (성숙도가 낮은 아래 feature map의 문제를 개선)

-SPP는 Convolution Layer에서 생성된 feature map을 입력 받고, 각 feature map에 대해 pooling 연산을 하여 고정된 길이의 출력을 만들어 낼 수 있다.

1. Anchor-free 방식  
anchor-based detector는 많은 anchor-box들을 지정해두고 input image들이 CNN을 통과    
YOLOX는 anchor가 없이 bounding box를 예측하는데, anchor-based detector의 단점을 개선한 anchor-free detector [FCOS](https://arxiv.org/abs/1904.01355)
2. 발전된 label assignment 기법 사용: SimOTA
3. 강력한 data augmentation: Mosaic, MixUp

## ByteTrack
[ByteTrack Paper](https://arxiv.org/abs/2110.06864)

[ByteTrack Official GitHub Repo](https://github.com/ifzhang/ByteTrack)

## OpenVino
OpenVINO는 AI 추론을 엣지 디바이스 상에서 최적화하기 위한 오픈 소스 toolkit 입니다. OpenVINO를 참조하기 위해서는 다음과 같은 페이지를 참조하시기 바랍니다.

[OpenVINO_GitHub](https://github.com/openvinotoolkit/openvino)

[OpenVINO_Overview](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)

[OpenVINO_Releases_Notes](https://www.intel.com/content/www/us/en/developer/articles/release-notes/openvino-relnotes.html)

OpenVINO의 지원 OS
- [Linux](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_linux.html)
- [Windows](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_windows.html)
- [macOS](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_macos.html)
- [Raspbian](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_raspbian.html)


  
# 개발환경

Device
- Raspberry Pi 4 model B+ (4GB,8GB)
- Intel Neural Compute Stick 2
- usb2.0 camera

OS
- Raspbian 10 Buster, 32-bit

Util
- openvino == 2021.4.2 (with openCV)
- cmake == 3.16.3

Python
- python == 3.7.3

pip3 packages
- numpy == 1.21.6
- scipy == 1.7.3
- loguru == 0.6.0
- lap == 0.4.0
- cython_bbox == 0.1.3
- Pillow == 5.4.1
  
# 사용법
## 라즈베리파이 초기 설정
1. Raspbian Buster, 32-bit OS를 설치합니다. [Pimager](https://www.raspberrypi.com/software/)
2. 라즈베리파이에 OpenVINO toolkit을 설치합니다. [모든 버전](https://storage.openvinotoolkit.org/repositories/openvino/packages/)
    1. tem 경로로 이동 합니다.  
    ```shell
    cd /tmp
    ```
    2. 빌드된 openvino 파일 다운로드 합니다.
    ```shell
    wget https://storage.openvinotoolkit.org/repositories/openvino/packages/2021.4.2/l_openvino_toolkit_runtime_raspbian_p_2021.4.752.tgz
    ```
    3. openvino 설치 경로 생성하고 받은 파일을 설치해줍니다.  
    ```shell
    sudo mkdir -p /opt/intel  
    cd /opt/intel  
    sudo tar -xf /tmp/l_openvino_toolkit_runtime_raspbian_p_2021.4.752.tgz -C /opt/intel  
    sudo mv l_openvino_toolkit_runtime_raspbian_p_2021.4.752 openvino  
    ```
    4. 다운로드 파일 삭제합니다.  
    ```shell
    rm -f /tmp/l_openvino_toolkit_runtime_raspbian_p_2021.4.752.tgz
    ```
    5. cmake를 설치합니다. 
    ```shell
    sudo apt install cmake
    ```
    6. openvino 환경 실행해줍니다.
    ```shell
    source /opt/intel/openvino/bin/setupvars.sh
    ```
    7. (optional) 터미널을 킬 때마다 위의 명령어를 실행하도록 하는 명령어 입니다.
    ```shell
    echo "source /opt/intel/openvino/bin/setupvars.sh" >> ~/.bashrc
    ```
    8. NCS2 규칙을 추가해 줍니다. `"$(whoami)"`에 계정이름으로 바꿔서 입력해주어야 합니다. (defult = pi)
    ```shell
    sudo usermod -a -G users "$(whoami)"
    sh /opt/intel/openvino/install_dependencies/install_NCS_udev_rules.sh
     ```
3. 기타 파일을 설치해줍니다
```shell
sudo apt install libgfortran5 libatlas3-base
sudo apt-get install libatlas-base-dev
```
## 코드 다운 및 실행

1. 코드를 git clone 해줍니다.
```shell
git clone https://github.com/ByungOhKo/Counting_ByteTrack_YOLO.git
cd ByteTrack
pip3 install -r requirements.txt
```
2. 코드에 필요한 패키지를 설치해줍니다.
```shell
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip3 install cython_bbox
sudo apt install libgfortran5 libatlas3-base
sudo apt-get install libatlas-base-dev
```
3. 코드를 다음과 같이 실행해 줍니다.
```shell
cd ByteTrack
python3 openvino_inference.py -m model/yolox_tiny_openvino/yolox_tiny -i "416,416" -s 0.5 --track_thresh 0.5
```
  
# 사용자 detection 모델을 사용하는 방법
## 메인 기기 openvino-dev 설치(window, ubuntu)
```shell
pip install openvino==2021.4.1
pip install openvino-dev==2021.4.1
pip install openvino-dev[onnx]
```

## onnx 를 IR(.bin, .xml)로 변환
1. openvino-dev를 설치합니다.
2. 훈련된 모델을 .onnx로 변환합니다.
3. 다음 명령어를 입력해줍니다.
```shell
mo --input_model <INPUT_MODEL_PATH>.onnx
```
4. --model 에 IR 파일 경로를,  --input_shape 모델의 입력크기를 넣어줍니다.
  
# runtime 매개변수
## 파일경로 args
 --model:  
 IR(.bin, .xml) 모델의 path를 입력으로 넣어주세요.  
   
 --video_path:  
 비디오를 입력으로 넣고싶은 경우 경로를 입력합니다.
   
 --output_dir:  
 파일을 저장할 경로를 입력합니다.
 
 ## object detection args  
 --input_shape:  
 IR 모델을 훈련 시켰을 때 입력 크기를 입력합니다.  
 
 --score_thr:  
 model를 통과하고 나온 detection중 score_thr 보다 낮은 detection은 제거됩니다.  
   
 --nms_thr:  
 non-maximal-suppression 의 threshold 입니다. 높을 수록 score가 높은 detection의 area와 IOU 가 높은 다른 detection을 많이 제거합니다.  
   
 --with_p6:  
 FPN/PAN 에서 p6를 사용할지를 결정합니다. p6를 사용하게 되면 더 512^2의 anchor box를 사용하게 됩니다.  
   
 ## object tracking args
 --track_thresh:  
 detection score에서 high, low를 나누는 threshold입니다.  
   
 --track_buffer:  
 tracking에서 잃어버린 track들에 대해서 몇 frame까지 정보를 가지고 있을지 정합니다. 값이 클 수록 더 오랫동안 도중에 tracking이 끊긴 track들에 대해서 정보를 가지고 있습니다.  
   
 --match_thresh:  
 헝가리안 알고리즘(최적 비용 매칭)에서 IOU값으로 매칭 detection을 매칭시킬 때 사용하는 threshold값 입니다. 높을 수록 더 많이 매칭이 되지만 정확도가 떨어지게 됩니다.  
   
 --min-box-area:  
 매칭 된 detection 중 min-box-area보다 작은 detection에 대해서는 버려줍니다.  
   
 --mot20:  
 mot20 challenge에 대해서 test할지 정합니다.  
 

