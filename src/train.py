'''
Resnet50 모델 정의 및 훈련 스크립트
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import yaml
from data_loader import get_loader
from utils import get_path


# 설정 파일 .yaml 로드
config_load_path = get_path("config", "config.yaml")
with open (config_load_path, "r") as f:
    config = yaml.safe_load(f)


# 경로 설정
train_data_path = get_path(config['paths']['train_data'])
val_data_path = get_path(config['paths']['val_data'])
model_save_path = get_path(config['paths']['model_save_path'])



# 데이터 로더 불러오기
train_loader, val_loader = get_loader(
    config['train']['batch_size'],
    config['paths']['train_data'],
    config['paths']['val_data']
    )


# 모델 정의 및 수정
model = models.resnet50(pretrained=config['model']['pretrained'])
model.fc = nn.Linear(model.fc.in_features, config['model']['num_classes'])
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')


# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config['train']['learning_rate'])


# 모델 훈련
for epoch in range(config['train']['num_epochs']):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to('cuda'), labels.to('cuda')

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{config['train']['num_epochs']}], Loss: {running_loss/len(train_loader)}")

    # 검증 정확도 계산
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy}%")

# 모델 저장
torch.save(model.state_dict(), config['paths']['model_save_path'])