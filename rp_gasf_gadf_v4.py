# -*- coding: utf-8 -*-
import warnings
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pyts.image import MarkovTransitionField, RecurrencePlot, GramianAngularField
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from ecg_cluster import load_and_preprocess_ecg_data
import heartpy as hp

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
NHEAD = 8
D_MODEL = 256
DROPOUT_RATE = 0.5
PATIENCE = 10  # 조기 종료를 위한 patience

OFFSET = None
DEBUG = False
dtype = 1
CLUSTERING = True
PLOT = False

# 데이터 증강 함수
def augment_ecg_signal(signal, max_shift=50):
    try:
        # 신호 시프트
        shift = np.random.randint(-max_shift, max_shift)
        if shift > 0:
            shifted_signal = np.pad(signal, (shift, 0), 'constant')[:len(signal)]
        else:
            shifted_signal = np.pad(signal, (0, -shift), 'constant')[-shift:]
        # 노이즈 추가
        noise = np.random.normal(0, 0.01, shifted_signal.shape)
        noisy_signal = shifted_signal + noise
        return noisy_signal
    except Exception as e:
        logging.error(f"Error augmenting signal: {e}")
        return None

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

def augment_signals(signals, labels, max_shift=50):
    augmented_signals = []
    augmented_labels = []
    for idx, (signal, label) in enumerate(zip(signals, labels)):
        augmented_signal = augment_ecg_signal(signal, max_shift=max_shift)
        if augmented_signal is not None:
            augmented_signals.append(augmented_signal)
            augmented_labels.append(label)
    return augmented_signals, augmented_labels

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

# Dataset 클래스
class ECGImageDataset(Dataset):
    def __init__(self, images, labels):
        """
        images: list of numpy arrays with shape (4, H, W)
        labels: list or array of labels
        """
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # (4, H, W)
        label = self.labels[idx]
        # 텐서로 변환
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return image, label

class ECGDataset(Dataset):
    def __init__(self, data, labels, augment=False):
        self.data = data
        self.labels = labels
        self.augment = augment

        self.transformers = {
            'RP': RecurrencePlot(),
            'GASF': GramianAngularField(method='summation'),
            'GADF': GramianAngularField(method='difference'),
            'MTF': MarkovTransitionField(n_bins=4),
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            signals = self.data[idx]  # Shape: (N_i, 300)
            label = self.labels[idx]

            if self.augment:
                signals = [augment_ecg_signal(signal) for signal in signals]
                signals = [normalize_signal(signal) for signal in signals]

            transformed_signals = []
            for signal in signals:
                transformed = []
                for key in ['RP', 'GASF', 'GADF', 'MTF']:
                    transformer = self.transformers[key]
                    img = transformer.transform(signal.reshape(1, -1))[0]
                    transformed.append(img)
                transformed = np.stack(transformed, axis=0)
                transformed = (transformed - np.mean(transformed)) / (np.std(transformed) + 1e-8)
                transformed_signals.append(transformed)

            transformed_signals = [torch.tensor(ts, dtype=torch.float32) for ts in transformed_signals]
            label = torch.tensor(label, dtype=torch.long)

            return transformed_signals, label
        except Exception as e:
            logging.error(f"Error in __getitem__ for index {idx}: {e}")
            # 예외를 다시 발생시켜 DataLoader가 이를 처리하게 합니다.
            raise e

# 커스텀 collate_fn
def collate_fn(batch):
    labels = []
    all_transformed_signals = []
    lengths = []
    for transformed_signals, label in batch:
        if transformed_signals is None:
            continue  # 에러가 발생한 샘플은 건너뜁니다.
        labels.append(label)
        all_transformed_signals.append(transformed_signals)
        lengths.append(len(transformed_signals))

    if not all_transformed_signals:
        return None, None, None

    max_length = max(lengths)

    padded_sequences = []
    attention_masks = []
    for signals in all_transformed_signals:
        N_i = len(signals)
        pad_size = max_length - N_i
        if pad_size > 0:
            pad = [torch.zeros_like(signals[0]) for _ in range(pad_size)]
            signals = signals + pad
        padded_sequences.append(torch.stack(signals))  # Shape: (max_length, 4, H, W)
        attention_mask = [1] * N_i + [0] * pad_size
        attention_masks.append(attention_mask)

    padded_sequences = torch.stack(padded_sequences)  # Shape: (batch_size, max_length, 4, H, W)
    attention_masks = torch.tensor(attention_masks, dtype=torch.bool)  # Shape: (batch_size, max_length)
    labels = torch.stack(labels)  # Shape: (batch_size,)

    # 텐서를 GPU로 이동시키지 않습니다. 학습 루프에서 이동시킵니다.
    return padded_sequences, attention_masks, labels




# Feature Extractor 정의
class FeatureExtractor(nn.Module):
    def __init__(self, output_dim=256, dropout_rate=0.3):
        super(FeatureExtractor, self).__init__()
        self.backbone = models.resnet34(pretrained=True)
        self.backbone.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.backbone.conv1.weight[:, 3, :, :] = self.backbone.conv1.weight[:, :3, :, :].mean(dim=1)
        for name, param in self.backbone.named_parameters():
            if "layer3" in name or "layer4" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# Transformer 모델 정의
class ECGTransformer(nn.Module):
    def __init__(self, num_classes, d_model=256, nhead=8, num_layers=2, dropout_rate=0.3):
        super(ECGTransformer, self).__init__()
        self.feature_extractor = FeatureExtractor(output_dim=d_model, dropout_rate=dropout_rate)
        self.feature_norm = nn.LayerNorm(d_model)
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=dropout_rate,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, attention_mask):
        batch_size, max_length, C, H, W = x.size()
        x = x.view(batch_size * max_length, C, H, W)
        x = self.feature_extractor(x)
        x = self.feature_norm(x)
        x = x.view(batch_size, max_length, self.d_model)
        src_key_padding_mask = ~attention_mask
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        attention_mask = attention_mask.unsqueeze(-1)
        masked_output = x * attention_mask
        sum_output = masked_output.sum(dim=1)
        valid_lengths = attention_mask.sum(dim=1) + 1e-8
        avg_output = sum_output / valid_lengths
        logits = self.fc(avg_output)
        return logits

# 학습 함수
# 학습 함수
# 학습 함수 내에서 attention_mask를 수정하는 부분
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for batch_idx, batch in enumerate(dataloader):
        inputs, labels = batch
        inputs = inputs.unsqueeze(1)  # (batch_size, 1, ...)
        attention_mask = torch.ones(inputs.size(0), inputs.size(1), dtype=torch.bool)  # (batch_size, seq_length)

        # 텐서를 올바르게 확장하여 2D로 변환
        attention_mask = attention_mask.to(device)
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        try:
            outputs = model(inputs, attention_mask)
            loss = criterion(outputs, labels)
        except ValueError as e:
            logging.error(f"Forward Error: {e}")
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels).item()
        total_samples += labels.size(0)

        del inputs, attention_mask, labels, outputs, loss
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
            inputs, labels = batch
            inputs = inputs.unsqueeze(1)  # (batch_size, 1, ...)

            # attention_mask를 batch_size와 seq_length에 맞춰서 2D로 확장
            attention_mask = torch.ones(inputs.size(0), inputs.size(1), dtype=torch.bool).to(device)

            inputs = inputs.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            try:
                outputs = model(inputs, attention_mask)  # model에 2D src_key_padding_mask 전달
                loss = criterion(outputs, labels)
            except ValueError as e:
                logging.error(f"Forward Error: {e}")
                continue

            val_running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_running_corrects += torch.sum(preds == labels).item()
            val_total_samples += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            del inputs, attention_mask, labels, outputs, loss
            torch.cuda.empty_cache()

    val_epoch_loss = val_running_loss / val_total_samples
    val_epoch_acc = val_running_corrects / val_total_samples

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    logging.info(f"Validation Loss: {val_epoch_loss:.4f}, Validation Acc: {val_epoch_acc:.4f}, "
                 f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    return val_epoch_loss, val_epoch_acc, precision, recall, f1


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
    segmented_signals = segment_ecg_signals(aligned_signals, window_size=300, sample_rate=100)
    logging.info(f"Segmented signals: {len(segmented_signals)}")

    # 4.3. 신호 증강
    augmented_signals, augmented_labels = augment_signals(segmented_signals, aligned_labels,  max_shift=10)
    logging.info(f"Augmented signals: {len(augmented_signals)}")

    # 4.4. 신호 정규화
    normalized_signals = normalize_signals(augmented_signals)
    logging.info(f"Normalized signals: {len(normalized_signals)}")

    # 4.5. 이미지 변환
    transformed_images = transform_to_images(normalized_signals)
    logging.info(f"Transformed images: {len(transformed_images)}")

    # 4.6. 레이블 조정 (데이터 증강 후 레이블 일치)
    # 증강된 신호와 레이블의 개수가 일치하는지 확인
    if len(transformed_images) != len(augmented_labels):
        logging.error("Transformed images and augmented labels length mismatch.")
        return


    # 5. 학습 및 검증 데이터 분할
    train_images, val_images, train_labels, val_labels = train_test_split(
        transformed_images, augmented_labels, test_size=0.2, random_state=42, stratify=augmented_labels
    )
    logging.info(f"Train samples: {len(train_images)}, Validation samples: {len(val_images)}")

    # 6. 클래스 가중치 계산
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    if torch.isnan(class_weights_tensor).any() or torch.isinf(class_weights_tensor).any():
        logging.error("class_weights_tensor에 NaN 또는 INF 값이 포함되어 있습니다.")
        return

    # 7. Dataset 및 DataLoader 설정
    train_dataset = ECGImageDataset(train_images, train_labels)
    val_dataset = ECGImageDataset(val_images, val_labels)

    # 학습 및 검증 데이터 분할
    # train_data, val_data, train_labels, val_labels = train_test_split(
    #     data, labels, test_size=0.2, random_state=42, stratify=labels
    # )
    #
    # # 클래스 가중치 계산
    # class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    # class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    #
    # if torch.isnan(class_weights_tensor).any() or torch.isinf(class_weights_tensor).any():
    #     logging.error("class_weights_tensor에 NaN 또는 INF 값이 포함되어 있습니다.")
    #     return
    #
    # # Dataset 및 DataLoader 설정
    # train_dataset = ECGDataset(train_data, train_labels, augment=True)
    # val_dataset = ECGDataset(val_data, val_labels, augment=False)

    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=BATCH_SIZE,
    #     shuffle=True,
    #     collate_fn=collate_fn,
    #     num_workers=0,  # 멀티프로세싱 활용 (Windows에서는 num_workers=0을 권장할 수 있음)
    #     pin_memory=True
    # )
    #
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=BATCH_SIZE,
    #     shuffle=False,
    #     collate_fn=collate_fn,
    #     num_workers=0,
    #     pin_memory=True
    # )

    num_workers = 0  # Unix 환경에서는 4 이상으로 설정 가능
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
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

    # 모델, 손실 함수, 옵티마이저, 스케줄러 설정
    model = ECGTransformer(
        num_classes=num_classes,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dropout_rate=DROPOUT_RATE
    ).to(device)

    initialize_weights(model.fc)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # 학습 가능한 파라미터만 옵티마이저에 전달
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=1e-5)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

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

        # 스케줄러 업데이트
        scheduler.step()

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
            inputs, labels = batch
            inputs = inputs.unsqueeze(1)  # (batch_size, 1, ...)

            # attention_mask를 2D로 확장 (batch_size, seq_length)
            attention_mask = torch.ones(inputs.size(0), inputs.size(1), dtype=torch.bool).to(device)

            # 텐서들 장치로 이동
            inputs = inputs.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # 모델의 출력 계산
            outputs = model(inputs, attention_mask)
            _, preds = torch.max(outputs, 1)

            # 텐서를 CPU로 이동하여 시각화 준비
            inputs_np = inputs.cpu().numpy()
            attention_mask_np = attention_mask.cpu().numpy()
            labels_np = labels.cpu().numpy()
            preds_np = preds.cpu().numpy()

            # 첫 번째 샘플의 시각화를 위한 처리
            sample_idx = 0
            sequence_length = int(attention_mask_np[sample_idx].sum())  # 마스크의 유효 길이 계산

            # 시각화
            plt.figure(figsize=(12, 6))
            for i in range(sequence_length):
                plt.subplot(4, (sequence_length // 4) + 1, i + 1)
                plt.imshow(inputs_np[sample_idx, i, 0, :, :], cmap='gray')
                plt.axis('off')
            plt.suptitle(f"True Label: {labels_np[sample_idx]}, Predicted: {preds_np[sample_idx]}")
            plt.show()

            break  # 하나의 배치만 시각화

if __name__ == '__main__':
    main()
