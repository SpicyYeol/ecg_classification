#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 20:23:13 2024

@author: yun

TCN (2D)
"""
#%%
import numpy as np
from scipy.signal import chirp
import torch.optim as optim
import matplotlib.pyplot as plt
from stockwell import st
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import f1_score, confusion_matrix, classification_report, cohen_kappa_score, matthews_corrcoef
import seaborn as sns
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import time
import random
from torch.nn.utils import weight_norm
from ranger21 import Ranger21
import logging


def setup_logging(log_file="training.log"):
    # 기존 핸들러 제거
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # 새 핸들러 추가
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:%(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
# 로깅 설정 적용
setup_logging("training.log")
logging.info("Logging setup complete!")

#%%
# Random seed 고정
def set_seed(seed):
    torch.manual_seed(seed)  # CPU 연산 시드 고정
    torch.cuda.manual_seed(seed)  # GPU 연산 시드 고정
    torch.cuda.manual_seed_all(seed)  # 모든 GPU에 동일한 시드 적용
    np.random.seed(seed)  # NumPy 시드 고정
    random.seed(seed)  # Python 랜덤 시드 고정
    torch.backends.cudnn.deterministic = True  # 연산의 결정론적 결과 보장
    torch.backends.cudnn.benchmark = False  # 특정 환경에서 속도 저하 가능성 있음

# 시드 설정
set_seed(5148)

#%% parameter
# TCN 모델 초기화
input_size = 2  # 입력 채널 수 (실수부와 허수부)
output_size = 5  # 출력 크기 (예시로 10개 클래스)
num_channels = [32, 32, 64]  # TCN 레이어 채널 수
kernel_size = 2
dropout = 0.2
fmin = 0  # Hz (stockwell freq min)
fmax = 15  # Hz (stockwell freq max)
signal_length=10 #(lasting second of signal)

#%%
class ResNetLikeCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(ResNetLikeCNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3)  # [B, 64, 76, 500]
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # [B, 64, 38, 250]

        # Residual Blocks
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

        # Fully Connected Layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        for _ in range(1, blocks):
            layers.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
#%%
class CustomECGDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None,transform_dir="./MITI_npy"):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.transform_dir=transform_dir
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        try:
            original_file = os.path.basename(self.file_paths[idx]).replace('.csv', '_transformed.npy')
            transformed_path = os.path.join(self.transform_dir, original_file)
            signal_tensor = torch.tensor(np.load(transformed_path), dtype=torch.float32)
            label = self.labels[idx]
        except Exception as e:
            print(f"Error at index {idx}: {str(e)}. Returning zero tensor.")
            signal_tensor = torch.zeros((2, 151, 1000), dtype=torch.float32)
            label = -1
        return signal_tensor, label

#%% main

# 초기화: 데이터를 저장할 리스트
data_list = []  # 신호 데이터를 저장할 리스트
labels_list = []  # 각 신호의 레이블을 저장할 리스트
file_dict = {"N": 0, "S": 1, "V": 2,"Q":3,"F":4}  # 폴더명과 레이블 매핑
file_paths = []
labels = []
for file_folder_name, label in file_dict.items():
    file_path = "./smote_MIT/" + file_folder_name + "/"
    for file in os.listdir(file_path):
        file_paths.append(file_path + file)
        labels.append(label)



print("loading...")

# 1차 분할: train (80%), valid+test (20%)
train_data, valid_test_data, train_labels, valid_test_labels = train_test_split(
    file_paths, labels, test_size=0.2, random_state=42, stratify=labels,shuffle=True
)

# 2차 분할: valid (50% of 20% = 10%), test (50% of 20% = 10%)
valid_data, test_data, valid_labels, test_labels = train_test_split(
    valid_test_data, valid_test_labels, test_size=0.5, random_state=42, stratify=valid_test_labels
)

# TensorDataset으로 변환
train_dataset = CustomECGDataset(train_data, train_labels)
valid_dataset = CustomECGDataset(valid_data, valid_labels)
test_dataset = CustomECGDataset(test_data, test_labels)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 초기화 및 GPU로 이동
model = ResNetLikeCNN(num_classes=5)

# 훈련 파라미터
best_val_loss = float('inf')
best_val_accuracy=0
save_path = os.path.join('revision_models', "2DResNet2.pth")
os.makedirs('revision_models', exist_ok=True)
trigger_times = 0
epochs=50
accumulation_steps=4
patience=8
num_workers = 8 #min(8, cpu_count())  # 시스템에 따라 조정
batch_size=4

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)
# 디바이스 및 모델 설정

model.to(device)

# 혼합 정밀도 훈련을 위한 설정
scaler = torch.cuda.amp.GradScaler()

# 손실 함수 및 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = Ranger21(model.parameters(), lr=0.0005, weight_decay = 1e-5,
                 eps = 1e-8, use_warmup=True,
                 use_cheb=False,
                 use_madgrad = False,
                 num_epochs = epochs,
                 using_gc=True, num_batches_per_epoch = len(train_loader))

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 훈련 루프
for epoch in range(epochs):
    start_time = time.time()
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    optimizer.zero_grad()
    for i, (inputs, labels_batch) in enumerate(train_loader):
        inputs = inputs.to(device, non_blocking=True)
        #print(inputs.shape)
        labels_batch = labels_batch.to(device, non_blocking=True)

        with torch.amp.autocast(device_type='cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels_batch)
            loss = loss / accumulation_steps  # 그래디언트 누적을 위한 손실 스케일링

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        train_loss += loss.item() * inputs.size(0) * accumulation_steps  # 스케일링 복원
        _, preds = torch.max(outputs, 1)
        train_total += labels_batch.size(0)
        train_correct += (preds == labels_batch).sum().item()

    train_accuracy = 100 * train_correct / train_total
    logging.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / train_total:.4f}, "
                 f"Train Accuracy: {train_accuracy:.2f}%")

    # 검증 루프
    model.eval()
    valid_loss = 0.0
    valid_correct = 0
    valid_total = 0
    with torch.no_grad():
        for inputs, labels_batch in valid_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)

            with torch.amp.autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels_batch)

            valid_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            valid_total += labels_batch.size(0)
            valid_correct += (preds == labels_batch).sum().item()

    valid_accuracy = 100 * valid_correct / valid_total
    logging.info(f"Validation Loss: {valid_loss / valid_total:.4f}, Validation Accuracy: {valid_accuracy:.2f}%")

    scheduler.step()

    # 베스트 모델 저장
    if valid_loss < best_val_loss or valid_accuracy>best_val_accuracy:
        best_val_loss = valid_loss
        best_val_accuracy=valid_accuracy
        trigger_times = 0
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss / train_total,
        }, save_path)
        logging.info(f"Epoch {epoch + 1}에서 모델 저장됨 (Validation Loss: {valid_loss / valid_total:.4f})")
    else:
        trigger_times += 1
        logging.info(f"Validation Loss 개선 없음: {trigger_times}회 연속")

    # 조기 종료
    if trigger_times >= patience:
        logging.info("조기 종료 조건 만족. 훈련을 종료합니다.")
        break

    elapsed_time = time.time() - start_time
    logging.info(f"Epoch 시간: {elapsed_time // 60:.0f}분 {elapsed_time % 60:.0f}초")

# 모델 테스트
model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels_batch in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels_batch = labels_batch.to(device, non_blocking=True)

        with torch.amp.autocast(device_type='cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels_batch)

        test_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        test_total += labels_batch.size(0)
        test_correct += (preds == labels_batch).sum().item()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())

# 전체 테스트 정확도
test_accuracy = 100 * test_correct / test_total

# 추가 평가지표 계산
f1 = f1_score(all_labels, all_preds, average='macro')  # 매크로 평균 F1 스코어
# 클래스별 F1 스코어
f1_per_class = f1_score(all_labels, all_preds, average=None)
# 클래스별 F1 스코어
f1_per_class = f1_score(all_labels, all_preds, average=None)

# 클래스별 이름 (필요하면 추가)
class_names = test_loader.dataset.classes if hasattr(test_loader.dataset, 'classes') else range(len(f1_per_class))
sensitivity = confusion_matrix(all_labels, all_preds, labels=range(5)).diagonal() / \
              confusion_matrix(all_labels, all_preds, labels=range(5)).sum(axis=1)
ppv = confusion_matrix(all_labels, all_preds, labels=range(5)).diagonal() / \
      confusion_matrix(all_labels, all_preds, labels=range(5)).sum(axis=0)
kappa = cohen_kappa_score(all_labels, all_preds)  # Cohen's Kappa
mcc = matthews_corrcoef(all_labels, all_preds)  # Matthews Correlation Coefficient

# 로그 출력
logging.info(f"Test Loss: {test_loss / test_total:.4f}, Test Accuracy: {test_accuracy:.2f}%")
logging.info(f"F1 Score (macro): {f1:.4f}")
for i, f1 in enumerate(f1_per_class):
    logging.info(f"F1 Score for class '{class_names[i]}': {f1:.4f}")
logging.info(f"Sensitivity: {sensitivity}")
logging.info(f"PPV: {ppv}")
logging.info(f"Cohen's Kappa: {kappa:.4f}")
logging.info(f"MCC: {mcc:.4f}")


