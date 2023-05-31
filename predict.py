import uvicorn
from fastapi import FastAPI
from fastapi import FastAPI, UploadFile, File

import torch
# import torchvision
import torchvision.transforms as T
from torch.autograd import Variable
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import os
import io
# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# necessary imports to be able to load the datasets
# from torch.utils.data import DataLoader, Dataset
# import torchvision.datasets as dataset
# from torchvision import models
from PIL import Image
from collections import Counter

# init app
app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    drug_num = await predict_pill_labels(file, model)
    return {"success": True, "message":"File uploaded successfully", "drug": drug_num}

model = torch.load('model.pth', map_location=torch.device('cpu'))

dict_label = {
    1: "아로나민골드",
    2: "게보린",
    3: "우루사",
    4: "타이레놀",
    5: "머시론",
    6: "인후신",
    7: "아세트아미노펜",
    8: "이가탄",
    9: "지르텍",
    10: "인사돌플러스",
    11: "임팩타민",
    12: "멜리안",
    13: "액티리버모닝",
    14: "둘코락스에스",
    15: "판시딜",
    16: "센시아",
    17: "모드콜에스",
    18: "이지엔6이브",
    19: "벤포벨",
    20: "동성정로환",
    21: "카베진코와"
}

def iou(box1, box2):
    # 바운딩 박스 좌표를 언팩하여 추출합니다.
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    # 바운딩 박스들의 너비와 높이를 계산합니다.
    width1 = x2 - x1
    height1 = y2 - y1
    width2 = x4 - x3
    height2 = y4 - y3

    # 바운딩 박스들의 겹치는 영역을 계산합니다.
    intersection_width = max(0, min(x2, x4) - max(x1, x3))
    intersection_height = max(0, min(y2, y4) - max(y1, y3))
    intersection_area = intersection_width * intersection_height

    # 두 바운딩 박스의 합집합 영역을 계산합니다.
    union_area = width1 * height1 + width2 * height2 - intersection_area

    # IoU 값을 계산합니다.
    iou = intersection_area / union_area

    return iou

def get_selected_boxes(prediction, threshold=0.5):
    # 예측된 레이블 값과 확률을 받아옵니다.
    predicted_boxes = prediction[0]["boxes"].detach().cpu().numpy()
    predicted_labels = prediction[0]["labels"].detach().cpu().numpy()
    predicted_scores = torch.softmax(prediction[0]["scores"], dim=0).detach().cpu().numpy()

    # 각 객체당 하나의 바운딩 박스만 선택합니다.
    selected_boxes = []
    selected_labels = []
    selected_scores = []
    used_indices = []
    for i in range(len(predicted_boxes)):
        if i in used_indices:
            continue
        box = predicted_boxes[i]
        label = predicted_labels[i]
        score = predicted_scores[i]

        selected_boxes.append(box)
        selected_labels.append(label)
        selected_scores.append(score)

        # 겹치는 영역이 있는 바운딩 박스를 제거합니다.
        for j in range(i + 1, len(predicted_boxes)):
            if iou(box, predicted_boxes[j]) > threshold:
                used_indices.append(j)

    return selected_boxes, selected_labels, selected_scores

async def predict_pill_labels(upload_file, model):
    # 이미지 불러오기 및 전처리
    # file = Image.file.convert("RGB")
    # image = Image.open(image_path).convert("RGB")

# 재익이형
    image_bytes = await upload_file.read()
    dataBytesIO = io.BytesIO(image_bytes)

    img = Image.open(dataBytesIO).convert('RGB')

    image_tensor = T.ToTensor()(img).unsqueeze(0)

    # 모델에 이미지 전달하여 예측 수행
    predictions = model(image_tensor)

    # 각 객체당 하나의 바운딩 박스만 선택합니다.
    selected_boxes, selected_labels, selected_scores = get_selected_boxes(predictions)

    label_list = []
    for box, label, score in zip(selected_boxes, selected_labels, selected_scores):
        label_name = label
        label_list.append(label_name)

    label_counts = Counter(label_list)
    label_counts_list = [[int(label), int(count)] for label, count in label_counts.items()]
    sorted_label_counts_list = sorted(label_counts_list, key=lambda x: x[1])

    return sorted_label_counts_list

@app.get("/test")
async def root():
    return {"message" : "this is test"}


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port = 8000)
