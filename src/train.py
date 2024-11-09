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
from tqdm import tqdm


# 설정 파일 .yaml 로드
config_load_path = get_path("config", "config.yaml")
with open (config_load_path, "r") as f:
    config = yaml.safe_load(f)


# 모델 정의 및 수정 (FC 레이어 정의)
class MyResNet50(nn.Module):
    def __init__(self):
        super(MyResNet50, self).__init__()
        self.model = models.resnet50(pretrained=config['model']['pretrained'])
        self.model.fc = nn.Linear(self.model.fc.in_features, config['model']['num_classes'])

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    # 경로 설정
    train_data_path = get_path(config['paths']['train_data'])
    val_data_path = get_path(config['paths']['val_data'])
    model_save_path = get_path(config['paths']['model_save_path'])



    # 데이터 로더 불러오기
    train_loader, val_loader = get_loader(
        config['train']['batch_size'],
        train_data_path,
        val_data_path
        )


    model = MyResNet50()

    # 특정 레이어 고정
    for param in model.parameters():
        param.requires_grad = False


    # 마지막 FC 레이어는 학습할 수 있도록 설정
    for param in model.model.fc.parameters():
        param.requires_grad = True


    # 모델을 GPU or CPU 로 이동
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)


    # 손실 함수 및 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.model.fc.parameters(), lr=config['train']['learning_rate'])


    # 모델 훈련
    for epoch in range(config['train']['num_epochs']):
        model.train()
        running_loss = 0.0

        with tqdm(train_loader, unit='batch') as tepoch:
            tepoch.set_description(f"Epoch [{epoch+1}/{config['train']['num_epochs']}]")
            for images, labels in tepoch:
                images, labels = images.to(device), labels.to(device).long()

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                tepoch.set_postfix(loss=running_loss / len(train_loader))

        print(f"Epoch [{epoch+1}/{config['train']['num_epochs']}], Loss: {running_loss/len(train_loader)}")

        # 검증 정확도 계산
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                prediction = torch.argmax(outputs, dim=1)
                correct += (prediction == labels).sum().item()
                total += labels.size(0)

        accuracy = 100 * correct / total
        print(f"Validation Accuracy: {accuracy}%")

    # 모델 저장
    torch.save(model.state_dict(), model_save_path)
    print(f"모델 저장 완료 : {model_save_path}")