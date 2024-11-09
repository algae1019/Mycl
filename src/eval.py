'''
모델 평가
'''

import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
from utils import get_path


# 커스텀 모델 정의
class MyResNet50(nn.Module):
    def __init__(self):
        super(MyResNet50, self).__init__()
        self.model = models.resnet50(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
    
    def forward(self, x):
        x = self.model(x)
        return x


# 모델 불러오기 및 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MyResNet50()
model_load_path = get_path("models", "resnet50_mycl.pth")
model.load_state_dict(torch.load(model_load_path, map_location=device))
model = model.to(device)
model.eval()
print(f"모델 로드 완료: {model_load_path}")


# 이미지 전처리 함수
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)
    return transform(image).unsqueeze(0).to(device)  # 배치 차원 추가


# 예측 함수
def predict(image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image)
        probability = torch.sigmoid(output).item()
        return probability
    

# 테스트 예측
test_image_path = get_path("data", "val", "fire", "sample_fire_image.jpg")
fire_probability = predict(test_image_path)
print(f"Fire Probability: {fire_probability}")


# 로그 파일에 저장
logs_path = get_path("outputs", "logs")
with open(logs_path, "w") as f:
    f.write(f"Fire Probability: {fire_probability}\n")
print(f"평가 결과 저장 완료: {logs_path}")