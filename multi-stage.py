import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ecg_cluster import load_and_preprocess_ecg_data


# 1. Transformer 기반 MLM 모델 정의
class ECGTransformerMLMModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, max_len=300):
        super(ECGTransformerMLMModel, self).__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, d_model))  # Positional encoding
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.output_fc = nn.Linear(d_model, input_dim)

    def forward(self, x):
        x = self.input_fc(x)
        x = x + self.positional_encoding[:, :x.size(1), :]  # Add positional encoding
        x = self.transformer_encoder(x)
        x = self.output_fc(x)
        return x

    def extract_features(self, x):
        x = self.input_fc(x)
        x = x + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        return x  # Transformer의 은닉 상태를 반환


# 2. Transformer 기반 분류 모델 정의 (MLM 특성을 입력으로)
class ECGTransformerClassificationModel(nn.Module):
    def __init__(self, d_model, num_classes):
        super(ECGTransformerClassificationModel, self).__init__()
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = x.mean(dim=1)  # 시퀀스의 평균을 사용하여 전체 특징을 얻음
        out = self.fc(x)
        return out


# 3. 데이터 마스킹 함수
def mask_data(ecg_data, mask_ratio=0.15):
    data = ecg_data.clone()
    num_mask = int(mask_ratio * data.size(1))
    mask_indices = np.random.choice(data.size(1), num_mask, replace=False)
    data[:, mask_indices, :] = 0  # 마스킹 (0으로 설정)
    return data, mask_indices


# 4. 모델 학습 및 평가

# 기본 설정
input_dim = 1  # ECG 시퀀스의 단일 차원 입력
d_model = 64  # Transformer 모델 차원
nhead = 8  # Multi-head attention
num_layers = 4  # Transformer 레이어 수
dim_feedforward = 128  # FFN의 차원
seq_length = 300  # ECG 시퀀스 길이
batch_size = 32
num_epochs = 10
num_classes = 4  # a, b, c, d에 대한 4개 클래스

# N_DATA = [2,4,6]  # 완료 2,4,6
OFFSET = None
DEBUG = False
LEARNING_RATE = 0.001
PLOT = False
dtype = 1
CLUSTERING = True

# Step 1: 데이터 로딩 및 전처리
ecg_data_A = load_and_preprocess_ecg_data(OFFSET, [2], dtype, DEBUG, CLUSTERING, PLOT)
ecg_data_B = load_and_preprocess_ecg_data(OFFSET, [4], dtype, DEBUG, CLUSTERING, PLOT)

# 샘플 ECG 데이터 (랜덤 데이터로 대체)
# ecg_data_A = torch.randn(batch_size, seq_length, input_dim)  # A Dataset (a, b, c, d)
# ecg_data_B = torch.randn(batch_size, seq_length, input_dim)  # B Dataset (e, f, g)

# 샘플 레이블 (랜덤 생성된 예시 레이블)
label_to_idx_A = {label: idx for idx, label in enumerate(set(d['label'] for d in ecg_data_A))}
data_A = torch.stack([torch.tensor(d['data']).unsqueeze(-1) for d in ecg_data_A])
labels_A = torch.tensor([label_to_idx_A[d['label']] for d in ecg_data_A])

labels_B = torch.randint(0, 3, (batch_size,))  # e, f, g 레이블

# Transformer 기반 MLM 모델 학습
mlm_model = ECGTransformerMLMModel(input_dim, d_model, nhead, num_layers, dim_feedforward)
criterion_mlm = nn.MSELoss()
optimizer_mlm = optim.Adam(mlm_model.parameters(), lr=0.001)

# MLM 학습
masked_data_A, _ = mask_data(ecg_data_A)
for epoch in range(num_epochs):
    mlm_model.train()
    optimizer_mlm.zero_grad()
    outputs = mlm_model(masked_data_A)  # 마스킹된 데이터 입력
    loss = criterion_mlm(outputs, ecg_data_A)  # 원래 데이터와 비교하여 손실 계산
    loss.backward()
    optimizer_mlm.step()
    print(f'MLM Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 학습된 MLM 모델에서 특징 추출
with torch.no_grad():
    features_A = mlm_model.extract_features(ecg_data_A)
    features_B = mlm_model.extract_features(ecg_data_B)

# Transformer 기반 분류 모델 학습 (MLM 모델의 특징을 사용)
classification_model_A = ECGTransformerClassificationModel(d_model, num_classes)
criterion_class = nn.CrossEntropyLoss()
optimizer_class_A = optim.Adam(classification_model_A.parameters(), lr=0.001)

# A Dataset 분류 학습
for epoch in range(num_epochs):
    classification_model_A.train()
    optimizer_class_A.zero_grad()
    outputs_A = classification_model_A(features_A)  # A Dataset의 MLM 특징을 사용하여 분류
    loss_class = criterion_class(outputs_A, labels_A)
    loss_class.backward()
    optimizer_class_A.step()
    print(f'Classification Epoch [{epoch + 1}/{num_epochs}], A Dataset Loss: {loss_class.item():.4f}')

# B Dataset에 대한 분류 모델 학습
classification_model_B = ECGTransformerClassificationModel(d_model, 3)  # B Dataset은 e, f, g
optimizer_class_B = optim.Adam(classification_model_B.parameters(), lr=0.001)

for epoch in range(num_epochs):
    classification_model_B.train()
    optimizer_class_B.zero_grad()
    outputs_B = classification_model_B(features_B)  # B Dataset의 MLM 특징을 사용하여 분류
    loss_class_B = criterion_class(outputs_B, labels_B)
    loss_class_B.backward()
    optimizer_class_B.step()
    print(f'Classification Epoch [{epoch + 1}/{num_epochs}], B Dataset Loss: {loss_class_B.item():.4f}')

# 5. 다단계 학습 및 매핑

# A, B Dataset의 분류 모델 특징 벡터 추출
with torch.no_grad():
    a_features = classification_model_A(features_A).detach().numpy()
    b_features = classification_model_B(features_B).detach().numpy()

# 코사인 유사도를 사용한 매핑 추정
similarity_matrix = cosine_similarity(b_features, a_features)

# 유사도 기반으로 B Dataset의 e, f, g 레이블을 A Dataset의 a, b, c, d로 매핑
mapped_labels = np.argmax(similarity_matrix, axis=1)

print("Mapped Labels:", mapped_labels)
