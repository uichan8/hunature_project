import numpy as np
import cv2
import albumentations as A
import albumentations

def preprocess(img, input_size):
    #이미지 변환 도구
    Resize = A.Compose([A.LongestMaxSize(input_size[0])])
    Normalize = A.Compose(A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],max_pixel_value=1, always_apply=True))
    swap = (2, 0, 1)

    #이미지 기틀 생성
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114
    
    #이미지 가로세로중 큰쪽이 416이 되도록 비율 산정
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])

    #이미지 크기 조정
    resized_img = Resize(image=img)['image']
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img[:, :, ::-1]
    padded_img = Normalize(image=padded_img)['image'].transpose(swap)

    return padded_img, r
