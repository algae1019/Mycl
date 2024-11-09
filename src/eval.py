'''
모델 평가
'''

import torch
from torchvision import models, transforms
from PIL import Image
from utils import get_path
import torch.nn as nn


# 모델 불러오기
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model_load_path = get_path("models", "resnet50_mycl.pth")
model.load_state_dict(torch.load(model_load_path))
model.eval()


# 이미지 전처리 함수
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)

    return transform(image).unsqueeze(0)  # 배치 차원 추가


# 예측 함수
def predict(image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return "fire" if predicted.item() == 0 else "no_fire"
    

# 테스트 예측
test_image_path = get_path("data", "val", "fire", "sample_fire_image.jpg")
print(predict(test_image_path))