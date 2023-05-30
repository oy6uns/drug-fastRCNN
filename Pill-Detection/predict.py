import uvicorn
from fastapi import FastAPI
from fastapi import FastAPI, UploadFile, File

import torch
# import torchvision
import torchvision.transforms as transforms
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

# init app
app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # drug_num = await convert_image_to_tensor(file)
    # drug = drug_label_dict[drug_num]
    return {"success": True, "message":"File uploaded successfully", "drug": [[1, 2], [5, 1]]}

# model = torch.load('resnet50_fintuned_epoch50_v1.pt', map_location=torch.device('cpu'))

drug_label_dict = {
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

# async def convert_image_to_tensor(upload_file):
#     image_bytes = await upload_file.read()
#     img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
#     transform = transforms.Compose([
#         transforms.RandomResizedCrop(size=256, scale=(0.95, 1.0)),
#         transforms.RandomRotation(degrees=15),
#         transforms.ColorJitter(),
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])

#     b= transform(img)
#     tensor_image = torch.unsqueeze(b, 0)

#     input = Variable(tensor_image)
#     output = model(input)
#     print(output)
#     _, preds = torch.max(output.data, 1)

#     return preds.item()

@app.get("/test")
async def root():
    return {"message" : "this is test"}


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port = 8000)
