# -*- coding: utf-8 -*-
# Relational Transformer for ECG Classification without Data Augmentation
import warnings
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pyts.image import MarkovTransitionField, RecurrencePlot, GramianAngularField
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models
from sklearn.metrics import precision_recall_fscore_support
from ecg_cluster import load_and_preprocess_ecg_data  # 사용자 정의 모듈
import heartpy as hp
import random

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)

# 경고 메시지 숨기기
warnings.filterwarnings("ignore")

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

# 하이퍼파라미터 및 설정
LEARNING_RATE = 3e-4
BATCH_SIZE = 32
NUM_EPOCHS = 100
NUM_LAYERS = 2
NHEAD = 8  # d_model=1024일 때 nhead=8
D_MODEL = 1024  # 이미지 특징(768) + 원시 신호 특징(256)
DROPOUT_RATE = 0.5
PATIENCE = 10  # 조기 종료를 위한 patience

OFFSET = None
DEBUG = False
dtype = 1
CLUSTERING = True
PLOT = False

# 데이터 전처리 함수들 (데이터 증강 제외)
def normalize_signal(signal):
    min_val = np.min(signal)
    max_val = np.max(signal)
    normalized_signal = (signal - min_val) / (max_val - min_val + 1e-8)
    return normalized_signal

def align_ecg_signals(signals, labels, sample_rate=300, window_size=600):
    aligned_signals = []
    aligned_labels = []
    for idx, (signal, label) in enumerate(zip(signals, labels)):
        try:
            working_data, measures = hp.process(signal, sample_rate=sample_rate, report_time=False)
            r_peaks = working_data['peaklist']
            if len(r_peaks) == 0:
                raise ValueError("R-peaks not found")
            # Align signal based on first R-peak
            first_r = r_peaks[0]
            start = first_r - window_size // 2
            end = first_r + window_size // 2
            if start < 0:
                # 신호 시작 부분 패딩
                segment = np.pad(signal[0:end], (abs(start), 0), 'constant')
            elif end > len(signal):
                # 신호 끝 부분 패딩
                segment = np.pad(signal[start:], (0, end - len(signal)), 'constant')
            else:
                segment = signal[start:end]
            aligned_signals.append(segment)
            aligned_labels.append(label)
        except Exception as e:
            logging.error(f"Error aligning signal index {idx}: {e}")
            continue  # 신호와 레이블 모두를 건너뜀
    return aligned_signals, aligned_labels

def segment_ecg_signals(signals, window_size=300, sample_rate=100):
    # 이미 R-피크 기준으로 정렬된 신호를 사용
    # 추가적인 세그멘테이션이 필요한 경우, 예를 들어 여러 윈도우로 분할
    segmented_signals = []
    for idx, signal in enumerate(signals):
        try:
            # 여기서는 각 신호가 이미 window_size를 만족한다고 가정
            segmented_signals.append(signal)
        except Exception as e:
            logging.error(f"Error segmenting signal index {idx}: {e}")
            continue
    return segmented_signals

def normalize_signals(signals):
    normalized_signals = []
    for idx, signal in enumerate(signals):
        try:
            normalized = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
            normalized_signals.append(normalized)
        except Exception as e:
            logging.error(f"Error normalizing signal index {idx}: {e}")
            continue
    return normalized_signals

def transform_to_images(signals):
    # 신호를 RP, GASF, GADF, MTF 이미지로 변환
    transformers = {
        'RP': RecurrencePlot(),
        'GASF': GramianAngularField(method='summation'),
        'GADF': GramianAngularField(method='difference'),
        'MTF': MarkovTransitionField(n_bins=4),
    }
    transformed_images = []
    for idx, signal in enumerate(signals):
        try:
            transformed = []
            for key in ['RP', 'GASF', 'GADF', 'MTF']:
                transformer = transformers[key]
                img = transformer.transform(signal.reshape(1, -1))[0]
                # 이미지 정규화
                img = (img - np.mean(img)) / (np.std(img) + 1e-8)
                transformed.append(img)
            # 채널 차원으로 스택
            transformed = np.stack(transformed, axis=0)  # Shape: (4, H, W)
            transformed_images.append(transformed)
        except Exception as e:
            logging.error(f"Error transforming signal index {idx} to images: {e}")
            continue
    return transformed_images

# 1D CNN 기반 Raw Signal Feature Extractor
class RawSignalFeatureExtractor(nn.Module):
    def __init__(self, input_channels=1, output_dim=256, dropout_rate=0.3):
        super(RawSignalFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: (batch_size, 1, signal_length)
        x = self.relu(self.bn1(self.conv1(x)))  # (batch_size, 64, L)
        x = self.pool(x)                         # (batch_size, 64, L/2)
        x = self.relu(self.bn2(self.conv2(x)))  # (batch_size, 128, L/2)
        x = self.pool2(x)                        # (batch_size, 128, L/4)
        x = self.relu(self.bn3(self.conv3(x)))  # (batch_size, 256, L/4)
        x = self.pool3(x)                        # (batch_size, 256, 1)
        x = x.view(x.size(0), -1)               # (batch_size, 256)
        x = self.dropout(x)
        x = self.fc(x)                           # (batch_size, output_dim)
        return x

# Attention 기반 특징 결합 모듈
class AttentionFusion(nn.Module):
    def __init__(self, feature_dim1, feature_dim2, fusion_dim=256):
        super(AttentionFusion, self).__init__()
        self.query = nn.Linear(feature_dim1, fusion_dim)
        self.key = nn.Linear(feature_dim2, fusion_dim)
        self.value = nn.Linear(feature_dim2, fusion_dim)
        self.fc = nn.Linear(fusion_dim, fusion_dim)

    def forward(self, feat1, feat2):
        # feat1: (batch_size, feature_dim1)
        # feat2: (batch_size, feature_dim2)
        Q = self.query(feat1)  # (batch_size, fusion_dim)
        K = self.key(feat2)    # (batch_size, fusion_dim)
        V = self.value(feat2)  # (batch_size, fusion_dim)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(K.size(-1))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        out = self.fc(attn_output)
        return out

# 수정된 Feature Extractor 클래스에 RawSignalFeatureExtractor 통합
class FeatureExtractor(nn.Module):
    def __init__(self, image_output_dim=768, raw_output_dim=256, fusion_dim=256, dropout_rate=0.3):
        super(FeatureExtractor, self).__init__()
        # 이미지 기반 Feature Extractor (ResNet18)
        self.image_backbone = models.resnet18(pretrained=True)
        self.image_backbone.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.image_backbone.conv1.weight[:, 3, :, :] = self.image_backbone.conv1.weight[:, :3, :, :].mean(dim=1)

        # Freeze early layers
        for name, param in self.image_backbone.named_parameters():
            if "layer3" in name or "layer4" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Extract features from layer2 and layer4
        self.layer2 = self.image_backbone.layer2
        self.layer4 = self.image_backbone.layer4

        # Pooling layers
        self.pool_layer2 = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_layer4 = nn.AdaptiveAvgPool2d((1, 1))

        # Image feature combination
        self.image_fc = nn.Linear(128 + 512, image_output_dim)  # layer2: 128 channels (ResNet18), layer4: 512 channels

        self.image_dropout = nn.Dropout(dropout_rate)

        # Raw signal Feature Extractor
        self.raw_signal_extractor = RawSignalFeatureExtractor(input_channels=1, output_dim=raw_output_dim, dropout_rate=dropout_rate)

        # Attention-based Fusion
        self.fusion = AttentionFusion(feature_dim1=image_output_dim, feature_dim2=raw_output_dim, fusion_dim=fusion_dim)

        # Combined Feature
        self.combined_fc = nn.Linear(fusion_dim, fusion_dim)  # fusion_dim
        self.combined_dropout = nn.Dropout(dropout_rate)

    def forward(self, image, raw_signal):
        # 이미지 처리
        x = self.image_backbone.conv1(image)
        x = self.image_backbone.bn1(x)
        x = self.image_backbone.relu(x)
        x = self.image_backbone.maxpool(x)

        x = self.image_backbone.layer1(x)
        x = self.layer2(x)  # Output of layer2
        feat_local = self.pool_layer2(x).view(x.size(0), -1)  # Shape: (batch_size, 128)

        x = self.image_backbone.layer3(x)
        x = self.layer4(x)  # Output of layer4
        feat_global = self.pool_layer4(x).view(x.size(0), -1)  # Shape: (batch_size, 512)

        # Concatenate image features
        image_combined = torch.cat((feat_local, feat_global), dim=1)  # Shape: (batch_size, 640)
        image_combined = self.image_dropout(image_combined)
        image_combined = self.image_fc(image_combined)  # Shape: (batch_size, 768)

        # 원시 신호 처리
        raw_features = self.raw_signal_extractor(raw_signal)  # Shape: (batch_size, 256)

        # 특징 결합 via Attention-based Fusion
        fused_features = self.fusion(image_combined, raw_features)  # Shape: (batch_size, 256)

        # Combined Feature
        combined = self.combined_dropout(fused_features)
        combined = self.combined_fc(combined)  # Shape: (batch_size, 256)
        return combined

# Residual Connections 적용된 Transformer 블록
class TransformerEncoderLayerWithResidual(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super(TransformerEncoderLayerWithResidual, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention with residual connection
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask,
                                 key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # Feedforward with residual connection
        src2 = self.linear2(self.dropout2(F.relu(self.linear1(src))))
        src = src + src2
        src = self.norm2(src)
        return src

# Transformer 모델 정의
class ECGTransformerWithResidual(nn.Module):
    def __init__(self, num_classes, d_model=256, nhead=8, num_layers=2, dropout_rate=0.3):
        super(ECGTransformerWithResidual, self).__init__()
        self.feature_extractor = FeatureExtractor(image_output_dim=768, raw_output_dim=256, fusion_dim=256, dropout_rate=dropout_rate)
        self.feature_norm = nn.LayerNorm(d_model)
        self.d_model = d_model

        self.transformer_encoder = nn.ModuleList([
            TransformerEncoderLayerWithResidual(d_model, nhead, dim_feedforward=1024, dropout=dropout_rate)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, images, raw_signals, attention_mask):
        """
        images: (batch_size, max_length, 4, H, W)
        raw_signals: (batch_size, max_length, 1, signal_length)
        attention_mask: (batch_size, max_length)
        """
        batch_size, max_length, C, H, W = images.size()
        raw_signals = raw_signals.view(batch_size * max_length, 1, -1)  # (batch_size * max_length, 1, signal_length)
        images = images.view(batch_size * max_length, C, H, W)          # (batch_size * max_length, C, H, W)

        # 특징 추출
        features = self.feature_extractor(images, raw_signals)           # (batch_size * max_length, 256)
        features = self.feature_norm(features)
        features = features.view(batch_size, max_length, self.d_model)  # (batch_size, max_length, d_model)

        # Transformer expects input as (sequence_length, batch_size, d_model)
        features = features.permute(1, 0, 2)  # (max_length, batch_size, d_model)

        # Transformer 인코더 통과
        for layer in self.transformer_encoder:
            features = layer(features, src_key_padding_mask=~attention_mask)

        # 출력의 평균을 취함 (패딩을 고려)
        features = features.permute(1, 0, 2)  # (batch_size, max_length, d_model)
        attention_mask = attention_mask.unsqueeze(-1)  # (batch_size, max_length, 1)
        masked_output = features * attention_mask  # (batch_size, max_length, d_model)
        sum_output = masked_output.sum(dim=1)     # (batch_size, d_model)
        valid_lengths = attention_mask.sum(dim=1) + 1e-8  # (batch_size, 1)
        avg_output = sum_output / valid_lengths     # (batch_size, d_model)
        logits = self.fc(avg_output)               # (batch_size, num_classes)
        return logits

# Dataset 클래스 수정: 이미지와 원시 신호 모두 반환
class ECGCombinedDataset(Dataset):
    def __init__(self, images, raw_signals, labels, augment=False):
        """
        images: list of numpy arrays with shape (4, H, W)
        raw_signals: list of numpy arrays with shape (1, signal_length)
        labels: list or array of labels
        augment: whether to apply additional image augmentations (현재 사용하지 않음)
        """
        self.images = images
        self.raw_signals = raw_signals
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]          # (4, H, W)
        raw_signal = self.raw_signals[idx]  # (1, signal_length)
        label = self.labels[idx]

        # # 추가적인 이미지 증강 (현재 비활성화)
        # if self.augment:
        #     if random.random() > 0.5:
        #         image = np.flip(image, axis=2)  # W 축을 기준으로 뒤집기

        # 텐서로 변환
        image = torch.tensor(image, dtype=torch.float32)
        raw_signal = torch.tensor(raw_signal, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return image, raw_signal, label

# Label Smoothing 적용된 CrossEntropyLoss (선택 사항)
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        smooth_label = torch.full_like(pred, self.smoothing / (pred.size(1) - 1))
        smooth_label.scatter_(1, target.unsqueeze(1), confidence)
        return torch.mean(torch.sum(-smooth_label * F.log_softmax(pred, dim=1), dim=1))

# 모델 가중치 초기화 함수
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

# 학습 함수
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for batch_idx, batch in enumerate(dataloader):
        images, raw_signals, labels = batch
        images = images.to(device, non_blocking=True)
        raw_signals = raw_signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        try:
            attention_mask = torch.ones(images.size(0), images.size(1), dtype=torch.bool).to(device)
            outputs = model(images, raw_signals, attention_mask)
            loss = criterion(outputs, labels)
        except ValueError as e:
            logging.error(f"Forward Error: {e}")
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels).item()
        total_samples += labels.size(0)

        del images, raw_signals, labels, outputs, loss
        torch.cuda.empty_cache()

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    return epoch_loss, epoch_acc

# 검증 함수
def validate(model, dataloader, criterion, device):
    model.eval()
    val_running_loss = 0.0
    val_running_corrects = 0
    val_total_samples = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            images, raw_signals, labels = batch
            images = images.to(device, non_blocking=True)
            raw_signals = raw_signals.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            try:
                attention_mask = torch.ones(images.size(0), images.size(1), dtype=torch.bool).to(device)
                outputs = model(images, raw_signals, attention_mask)
                loss = criterion(outputs, labels)
            except ValueError as e:
                logging.error(f"Forward Error: {e}")
                continue

            val_running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_running_corrects += torch.sum(preds == labels).item()
            val_total_samples += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            del images, raw_signals, labels, outputs, loss
            torch.cuda.empty_cache()

    val_epoch_loss = val_running_loss / val_total_samples
    val_epoch_acc = val_running_corrects / val_total_samples

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    logging.info(f"Validation Loss: {val_epoch_loss:.4f}, Validation Acc: {val_epoch_acc:.4f}, "
                 f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    return val_epoch_loss, val_epoch_acc, precision, recall, f1

# 메인 함수
def main():
    # 데이터 로딩 및 전처리
    try:
        ecg_data_A = load_and_preprocess_ecg_data(OFFSET, [2], dtype, DEBUG, CLUSTERING, PLOT)[:100]
    except Exception as e:
        logging.error(f"Data loading error: {e}")
        return

    # 레이블 인코딩
    label_to_idx_A = {label: idx for idx, label in enumerate(set(d['label'] for d in ecg_data_A))}
    data = [d['data'] for d in ecg_data_A]
    labels = np.array([label_to_idx_A[d['label']] for d in ecg_data_A])

    num_classes = len(label_to_idx_A)
    logging.info(f"Number of classes: {num_classes}")
    print(f"Number of classes: {num_classes}")

    # 데이터 무결성 검사
    finite_mask = np.isfinite(labels)
    data = [d for d, f in zip(data, finite_mask) if f]
    labels = labels[finite_mask]

    if not np.all(np.isfinite(labels)):
        logging.error("레이블에 비정상적인 값(NaN 또는 Inf)이 포함되어 있습니다.")
        return

    # 4. 데이터 전처리
    # 4.1. 신호 정렬
    aligned_signals, aligned_labels = align_ecg_signals(data, labels, sample_rate=100)
    logging.info(f"Aligned signals: {len(aligned_signals)}")

    # 4.2. 윈도우 분할 (이미 aligned_signals are window_size)
    segmented_signals = segment_ecg_signals(aligned_signals, window_size=50, sample_rate=100)
    logging.info(f"Segmented signals: {len(segmented_signals)}")

    # 4.3. 신호 정규화
    normalized_signals = normalize_signals(segmented_signals)
    logging.info(f"Normalized signals: {len(normalized_signals)}")

    # 4.4. 이미지 변환
    transformed_images = transform_to_images(normalized_signals)
    logging.info(f"Transformed images: {len(transformed_images)}")

    # 4.5. 레이블 조정 (데이터 증강 후 레이블 일치)
    # 증강된 신호와 레이블의 개수가 일치하는지 확인
    if len(transformed_images) != len(aligned_labels):
        logging.error("Transformed images and aligned labels length mismatch.")
        return

    # 5. 학습 및 검증 데이터 분할
    train_images, val_images, train_labels, val_labels = train_test_split(
        transformed_images, aligned_labels, test_size=0.2, random_state=42, stratify=aligned_labels
    )
    logging.info(f"Train samples: {len(train_images)}, Validation samples: {len(val_images)}")

    # 6. 클래스 가중치 계산
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    if torch.isnan(class_weights_tensor).any() or torch.isinf(class_weights_tensor).any():
        logging.error("class_weights_tensor에 NaN 또는 INF 값이 포함되어 있습니다.")
        return

    # 7. 원시 신호 준비 (원시 신호는 aligned_signals)
    # aligned_signals는 list of numpy arrays with shape (300,)
    # raw_signals는 list of numpy arrays with shape (1, 300)
    raw_signals = [signal.reshape(1, -1) for signal in normalized_signals]
    # Split train and val raw_signals accordingly
    train_raw_signals = [raw_signals[i] for i in train_labels]
    val_raw_signals = [raw_signals[i] for i in val_labels]

    # 8. Dataset 및 DataLoader 설정
    train_dataset = ECGCombinedDataset(train_images, train_raw_signals, train_labels, augment=False)
    val_dataset = ECGCombinedDataset(val_images, val_raw_signals, val_labels, augment=False)

    # 클래스 불균형을 해결하기 위한 WeightedRandomSampler 사용
    class_counts = np.bincount(train_labels)
    class_weights_sampler = 1. / class_counts
    sample_weights = class_weights_sampler[train_labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    num_workers = 0  # Unix 환경에서는 4 이상으로 설정 가능
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=False,  # WeightedRandomSampler 사용 시 shuffle=False
        collate_fn=None,  # 기본 collate_fn 사용
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=None,
        num_workers=num_workers,
        pin_memory=True
    )

    # 9. 모델, 손실 함수, 옵티마이저, 스케줄러 설정
    model = ECGTransformerWithResidual(
        num_classes=num_classes,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dropout_rate=DROPOUT_RATE
    ).to(device)

    initialize_weights(model.fc)

    # 손실 함수: CrossEntropyLoss에 클래스 가중치 적용
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    # Label Smoothing 사용 시:
    # criterion = LabelSmoothingLoss(smoothing=0.1)

    # 학습 가능한 파라미터만 옵티마이저에 전달
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=1e-5)

    # 스케줄러: ReduceLROnPlateau 사용
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    # 10. 학습 루프
    best_val_acc = 0.0
    best_model_state = None
    trigger_times = 0

    for epoch in range(NUM_EPOCHS):
        logging.info(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )
        logging.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        val_loss, val_acc, precision, recall, f1 = validate(
            model,
            val_loader,
            criterion,
            device
        )

        # 스케줄러 업데이트: 검증 정확도를 기준으로 학습률 조정
        scheduler.step(val_acc)

        # 모델 성능 기록
        if val_acc > best_val_acc:
            logging.info(f"Val Acc: {val_acc:.4f}, Best Acc: {best_val_acc:.4f}")
            print(f"Val Acc: {val_acc:.4f}, Best Acc: {best_val_acc:.4f}")
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            trigger_times = 0
            logging.info("Validation accuracy improved. Saving best model.")
            print("Validation accuracy improved. Saving best model.")
        else:
            trigger_times += 1
            logging.info(f"No improvement in validation accuracy for {trigger_times} epochs.")
            print(f"No improvement in validation accuracy for {trigger_times} epochs.")

        # 조기 종료
        if trigger_times >= PATIENCE:
            logging.info("Early stopping triggered.")
            print("Early stopping triggered.")
            break

    # 최적의 모델 로드
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logging.info(f"Loaded best model with validation accuracy: {best_val_acc:.4f}")
        print(f"Loaded best model with validation accuracy: {best_val_acc:.4f}")

    # 모델 저장
    torch.save(model.state_dict(), 'best_ecg_transformer_model.pth')
    logging.info("Best model saved to 'best_ecg_transformer_model.pth'.")
    print("Best model saved to 'best_ecg_transformer_model.pth'.")

    # 추가 평가 및 시각화 (선택 사항)
    # 예시: 첫 번째 배치의 첫 번째 샘플 시각화
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            images, raw_signals, labels = batch
            images = images.to(device, non_blocking=True)
            raw_signals = raw_signals.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # 모델의 출력 계산
            attention_mask = torch.ones(images.size(0), images.size(1), dtype=torch.bool).to(device)
            outputs = model(images, raw_signals, attention_mask)
            _, preds = torch.max(outputs, 1)

            # 텐서를 CPU로 이동하여 시각화 준비
            images_np = images.cpu().numpy()
            labels_np = labels.cpu().numpy()
            preds_np = preds.cpu().numpy()

            # 첫 번째 샘플의 시각화를 위한 처리
            sample_idx = 0
            sequence_length = images_np.shape[1]  # max_length

            # 시각화
            plt.figure(figsize=(12, 6))
            for i in range(sequence_length):
                plt.subplot(4, (sequence_length // 4) + 1, i + 1)
                plt.imshow(images_np[sample_idx, i, 0, :, :], cmap='gray')
                plt.axis('off')
            plt.suptitle(f"True Label: {labels_np[sample_idx]}, Predicted: {preds_np[sample_idx]}")
            plt.show()

            break  # 하나의 배치만 시각화

if __name__ == '__main__':
    main()
