'''
데이터 전처리 및 로더
'''

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_loader(batch_size, train_data_path, val_data_path):
    # 이미지 전처리 설정
    transform = transforms.Compose([
        transforms.Resize(256),     # 이미지 크기 256x256 조정
        transforms.CenterCrop(224), # 중앙 224x224 크기 자르기
        transforms.ToTensor(),      # 이미지를 텐서로 변환
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])    # 정규화
    ])

    # 학습 및 검증 데이터셋 생성
    train_dataset = datasets.ImageFolder(root=train_data_path, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_data_path, transform=transform)

    # 데이터 로더 설정
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    return train_loader, val_loader