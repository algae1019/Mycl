'''
predict.py - 슬라이딩 윈도우 방식 예측
'''

import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image
from utils import get_path


# 모델 정의 및 FC 레이어 수정
class MyResNet50(nn.Module):
    def __init__(self):
        super(MyResNet50, self).__init__()
        self.model = models.resnet50(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
    
    def forward(self, x):
        x = self.model(x)
    

    # 슬라이딩 윈도우 예측 함수
def sliding_window_prediction(model, image_path, patch_size=50, stride=25):
    model.eval()
    device = next(model.parameters()).device

    # 이미지 불러오기 및 전처리
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 패치 예측 결과를 저장할 맵 초기화
    output_map = np.zeros((image_tensor.shape[2], image_tensor.shape[3]))

    # 슬라이딩 윈도우를 통한 패치별 예측 수행
    with torch.no_grad():
        for y in range(0, image_tensor.shape[2] - patch_size + 1, stride):
            for x in range(0, image_tensor.shape[3] - patch_size + 1, stride):
                # 현재 위치에서 패치 추출
                patch = image_tensor[:, :, y:y + patch_size, x:x + patch_size].to(device)
                
                # 패치 예측 수행
                output = model(patch)
                prediction = torch.sigmoid(output).item()

                # 패치 중앙 위치에 예측 결과 저장
                center_y, center_x = y + patch_size // 2, x + patch_size // 2
                output_map[center_y, center_x] = prediction

    return output_map


# 모델 로드 및 경로 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MyResNet50()
model_load_path = get_path("models", "resnet50_mycl.pth")
model.load_state_dict(torch.load(model_load_path, map_location=device))
model = model.to(device)
print(f"모델 로드 완료: {model_load_path}")


# 예측 수행
test_image_path = get_path("data", "test", "sample_image.jpg")
output_map = sliding_window_prediction(model, test_image_path, patch_size=50, stride=25)
print("슬라이딩 윈도우 예측 결과 맵:")
print(output_map)


# 예측 결과 저장
output_predictions_path = get_path("outputs", "predictions")
np.save(output_predictions_path, output_map)
print(f"예측 결과 저장 완료: {output_predictions_path}")