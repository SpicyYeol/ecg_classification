import warnings
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from ecg_cluster import load_and_preprocess_ecg_data  # 데이터 로딩 함수

# 경고 메시지 숨기기
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

OFFSET = None
DEBUG = False
LEARNING_RATE = 1e-4
PLOT = False
dtype = 1
CLUSTERING = True

option = ['TRAIN']


# ============================================
# 데이터 증강 및 정규화 함수
# ============================================

def augment_ecg_signal(signal):
    """Gaussian noise 추가"""
    noise = np.random.normal(0, 0.01, signal.shape)
    augmented_signal = signal + noise
    return augmented_signal


def normalize_signal(signal):
    """신호를 [0,1] 범위로 정규화"""
    min_val = np.min(signal)
    max_val = np.max(signal)
    normalized_signal = (signal - min_val) / (max_val - min_val + 1e-8)
    return normalized_signal


# ============================================
# Dataset 클래스 정의 (원시 ECG 데이터 사용)
# ============================================

class ECGDataset(Dataset):
    def __init__(self, data, labels, augment=False):
        """
        Args:
            data (list of np.ndarray): 각 샘플은 (N_i, 300) 형태의 배열.
            labels (np.ndarray): 각 샘플의 레이블.
            augment (bool): 데이터 증강 여부.
        """
        self.data = data
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            waveform (torch.Tensor): (N_i, 300) 형태의 텐서.
            label (torch.Tensor): 정수형 레이블.
        """
        waveform = self.data[idx]  # Shape: (N_i, 300)
        label = self.labels[idx]

        if self.augment:
            # 데이터 증강: Gaussian noise 추가 및 정규화
            waveform = [augment_ecg_signal(s) for s in waveform]
            waveform = [normalize_signal(s) for s in waveform]

        # 텐서로 변환
        waveform = torch.tensor(waveform, dtype=torch.float32)  # Shape: (N_i, 300)

        return waveform, torch.tensor(label, dtype=torch.long)


# ============================================
# Custom Collate Function (패딩 및 마스크 생성)
# ============================================

def collate_fn(batch):
    """
    Args:
        batch (list of tuples): [(waveform, label), ...]
    Returns:
        padded_waveforms (torch.Tensor): (batch_size, max_N, 300)
        attention_mask (torch.Tensor): (batch_size, max_N)
        labels (torch.Tensor): (batch_size,)
    """
    waveforms, labels = zip(*batch)  # waveforms는 (N_i, 300)의 리스트

    # 패딩
    padded_waveforms = pad_sequence(waveforms, batch_first=True, padding_value=0)  # (batch_size, max_N, 300)

    # 어텐션 마스크 생성: 1은 유효 데이터, 0은 패딩
    lengths = torch.tensor([w.shape[0] for w in waveforms], dtype=torch.long)
    max_length = padded_waveforms.size(1)
    attention_mask = torch.arange(max_length).expand(len(lengths), max_length) < lengths.unsqueeze(1)
    attention_mask = attention_mask.to(device)  # (batch_size, max_N)

    # 레이블 텐서
    labels = torch.stack(labels).to(device)  # (batch_size,)

    # 패딩된 웨이브폼을 디바이스로 이동
    padded_waveforms = padded_waveforms.to(device)  # (batch_size, max_N, 300)

    return padded_waveforms, attention_mask, labels


# ============================================
# Positional Encoding 정의
# ============================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # 짝수 인덱스
        pe[:, 1::2] = torch.cos(position * div_term)  # 홀수 인덱스

        self.register_buffer('pe', pe)  # 학습되지 않음

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (batch_size, seq_length, d_model)
        Returns:
            x (torch.Tensor): (batch_size, seq_length, d_model) + positional encoding
        """
        # print(f"PositionalEncoding input shape: {x.shape}")  # 디버깅용 출력
        seq_length = x.size(1)
        if seq_length > self.pe.size(0):
            raise ValueError(f"Sequence length {seq_length} exceeds maximum length {self.pe.size(0)}")
        pe_slice = self.pe[:seq_length, :].unsqueeze(0)  # (1, seq_length, d_model)
        # print(f"PositionalEncoding pe_slice shape: {pe_slice.shape}")  # 디버깅용 출력
        x = x + pe_slice
        # print(f"PositionalEncoding output shape: {x.shape}")  # 디버깅용 출력
        return x


# ============================================
# Custom Transformer Encoder Layer 정의
# ============================================

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super(CustomTransformerEncoderLayer, self).__init__(*args, **kwargs)
        self.attn_weights = None

    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
        # 어텐션 가중치 초기화
        self.attn_weights = None

        # Self-attention
        src2, attn_output_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            **kwargs  # 추가 인자 전달
        )
        self.attn_weights = attn_output_weights
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feedforward network
        src2 = self.linear1(src)  # (batch_size, seq_length, dim_feedforward)
        src2 = self.activation(src2)  # 활성화 함수 적용
        src2 = self.dropout(src2)  # 드롭아웃 적용
        src2 = self.linear2(src2)  # (batch_size, seq_length, d_model)
        src2 = self.dropout2(src2)  # 드롭아웃 적용
        src = src + src2  # Residual connection
        src = self.norm2(src)

        return src


# ============================================
# Transformer 기반 Feature Extractor 정의
# ============================================

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim=1, d_model=256, num_heads=8, num_layers=3, dropout=0.1, max_len=300):
        """
        Args:
            input_dim (int): 각 타임 스텝의 차원 (여기서는 1).
            d_model (int): 임베딩 차원.
            num_heads (int): 어텐션 헤드 수.
            num_layers (int): Transformer 인코더 레이어 수.
            dropout (float): 드롭아웃 비율.
            max_len (int): 최대 시퀀스 길이.
        """
        super(FeatureExtractor, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.max_len = max_len

        # 임베딩 레이어
        self.embedding = nn.Linear(input_dim, d_model)

        # 포지셔널 인코딩
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_len)

        # Transformer 인코더
        encoder_layer = CustomTransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, mask):
        """
        Args:
            x (torch.Tensor): (batch_size * max_N, 300)
            mask (torch.Tensor): (batch_size * max_N, 300)
        Returns:
            features (torch.Tensor): (batch_size * max_N, d_model)
        """
        # print(f"FeatureExtractor input x shape: {x.shape}")  # 디버깅용 출력
        x = x.unsqueeze(-1)  # (batch_size * max_N, 300, 1)
        # print(f"FeatureExtractor after unsqueeze: {x.shape}")  # 디버깅용 출력

        x = self.embedding(x)  # (batch_size * max_N, 300, d_model)
        # print(f"FeatureExtractor after embedding: {x.shape}")  # 디버깅용 출력

        x = self.positional_encoding(x)  # (batch_size * max_N, 300, d_model)
        # print(f"FeatureExtractor after positional encoding: {x.shape}")  # 디버깅용 출력

        # Transformer 인코더
        # src_key_padding_mask: (batch_size * max_N, 300)
        x = self.transformer_encoder(x, src_key_padding_mask=~mask)  # (batch_size * max_N, 300, d_model)
        # print(f"FeatureExtractor after transformer encoder: {x.shape}")  # 디버깅용 출력

        # 평균 풀링
        x = x.masked_fill(~mask.unsqueeze(-1), 0)  # 패딩 부분을 0으로 설정
        sum_x = x.sum(dim=1)  # (batch_size * max_N, d_model)
        lengths = mask.sum(dim=1).unsqueeze(-1) + 1e-8  # (batch_size * max_N, 1)
        avg_x = sum_x / lengths  # (batch_size * max_N, d_model)
        # print(f"FeatureExtractor after pooling: {avg_x.shape}")  # 디버깅용 출력

        return avg_x


# ============================================
# Transformer 기반 분류기 정의
# ============================================

class ClassificationTransformer(nn.Module):
    def __init__(self, d_model=256, num_heads=8, num_layers=3, num_classes=4, dropout=0.1):
        """
        Args:
            d_model (int): 임베딩 차원.
            num_heads (int): 어텐션 헤드 수.
            num_layers (int): Transformer 인코더 레이어 수.
            num_classes (int): 분류할 클래스 수.
            dropout (float): 드롭아웃 비율.
        """
        super(ClassificationTransformer, self).__init__()
        self.d_model = d_model

        # 포지셔널 인코딩
        self.positional_encoding = PositionalEncoding(d_model, max_len=500)  # max_len은 필요에 따라 조정

        # Transformer 인코더
        encoder_layer = CustomTransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 분류기
        self.fc = nn.Linear(d_model, num_classes)

        # 저장된 어텐션 가중치를 위한 변수
        self.attention_weights = []

    def forward(self, x, mask):
        """
        Args:
            x (torch.Tensor): (batch_size, max_N, d_model)
            mask (torch.Tensor): (batch_size, max_N)
        Returns:
            logits (torch.Tensor): (batch_size, num_classes)
        """
        # 어텐션 가중치 초기화
        self.attention_weights = []

        # print(f"ClassificationTransformer input x shape: {x.shape}")  # 디버깅용 출력
        # 포지셔널 인코딩 추가
        x = self.positional_encoding(x)  # (batch_size, max_N, d_model)
        # print(f"ClassificationTransformer after positional encoding: {x.shape}")  # 디버깅용 출력

        # Transformer 인코더
        x = self.transformer_encoder(x, src_key_padding_mask=~mask)  # (batch_size, max_N, d_model)
        # print(f"ClassificationTransformer after transformer encoder: {x.shape}")  # 디버깅용 출력

        # 어텐션 가중치 수집
        for layer in self.transformer_encoder.layers:
            if hasattr(layer, 'attn_weights') and layer.attn_weights is not None:
                self.attention_weights.append(layer.attn_weights.detach().cpu())

        # 평균 풀링
        x = x.masked_fill(~mask.unsqueeze(-1), 0)  # 패딩 부분을 0으로 설정
        sum_x = x.sum(dim=1)  # (batch_size, d_model)
        lengths = mask.sum(dim=1).unsqueeze(-1) + 1e-8  # (batch_size, 1)
        avg_x = sum_x / lengths  # (batch_size, d_model)
        # print(f"ClassificationTransformer after pooling: {avg_x.shape}")  # 디버깅용 출력

        # 분류기
        logits = self.fc(avg_x)  # (batch_size, num_classes)
        # print(f"ClassificationTransformer logits: {logits.shape}")  # 디버깅용 출력

        return logits


# ============================================
# 전체 모델 정의
# ============================================

class ECGModel(nn.Module):
    def __init__(self, num_classes, d_model=256, num_heads=8, num_layers=3, dropout_rate=0.1):
        super(ECGModel, self).__init__()
        self.feature_extractor = FeatureExtractor(
            input_dim=1,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout_rate,
            max_len=300
        )
        self.classifier = ClassificationTransformer(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout_rate
        )

    def forward(self, x, mask):
        """
        Args:
            x (torch.Tensor): (batch_size, max_N, 300)
            mask (torch.Tensor): (batch_size, max_N)
        Returns:
            logits (torch.Tensor): (batch_size, num_classes)
        """
        batch_size, max_N, seq_length = x.size()
        # print(f"ECGModel forward - Input x shape: {x.shape}")  # 디버깅용 출력

        # Reshape x from [batch_size, max_N, 300] to [batch_size * max_N, 300]
        x = x.view(batch_size * max_N, seq_length)
        # print(f"ECGModel forward - Reshaped x for FeatureExtractor: {x.shape}")  # 디버깅용 출력

        # Create feature_mask as all True since each waveform has fixed length 300
        feature_mask = torch.ones(batch_size * max_N, seq_length, device=x.device, dtype=torch.bool)
        # print(f"ECGModel forward - Feature mask shape: {feature_mask.shape}")  # 디버깅용 출력

        # Feature Extractor
        features = self.feature_extractor(x, feature_mask)  # (batch_size * max_N, d_model)
        # print(f"ECGModel forward - Features from FeatureExtractor: {features.shape}")  # 디버깅용 출력

        # Reshape features back to [batch_size, max_N, d_model]
        features = features.view(batch_size, max_N, -1)
        # print(f"ECGModel forward - Reshaped features for ClassificationTransformer: {features.shape}")  # 디버깅용 출력

        # Classification Transformer
        logits = self.classifier(features, mask)  # (batch_size, num_classes)
        # print(f"ECGModel forward - Logits shape: {logits.shape}")  # 디버깅용 출력

        return logits

# ============================================
# Attention Map 시각화 함수
# ============================================

def visualize_attention(model, dataloader, num_layers=3, num_heads=8):
    """
    Args:
        model (ECGModel): 학습된 모델.
        dataloader (DataLoader): 검증 데이터 로더.
        num_layers (int): Transformer 인코더 레이어 수.
        num_heads (int): 어텐션 헤드 수.
    """
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs, mask, labels = batch
            inputs = inputs.to(device)
            mask = mask.to(device)
            labels = labels.to(device)

            outputs = model(inputs, mask)
            _, preds = torch.max(outputs, 1)

            # 첫 번째 샘플의 어텐션 가중치 추출
            attention_weights = model.classifier.attention_weights  # 리스트 of (num_heads, seq_length, seq_length)
            if not attention_weights:
                print("No attention weights found.")
                return

            # 각 레이어별로 시각화
            for layer_idx, layer_attn in enumerate(attention_weights):
                # layer_attn: (num_heads, seq_length, seq_length)
                # 첫 번째 헤드
                first_head_attn = layer_attn[0].numpy()  # (seq_length, seq_length)

                plt.figure(figsize=(6, 5))
                plt.imshow(first_head_attn, cmap='viridis')
                plt.title(f"Attention Weights - Layer {layer_idx + 1}, Head 1")
                plt.xlabel("Key Position")
                plt.ylabel("Query Position")
                plt.colorbar()
                plt.show()

            # 첫 번째 샘플의 예측 결과 출력
            print(f"True Label: {labels[0].item()}, Predicted: {preds[0].item()}")

            break  # 첫 번째 배치만 시각화

# ============================================
# Training 및 Validation 함수 정의
# ============================================

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device, scaler=None):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for inputs, mask, labels in dataloader:
        inputs = inputs.to(device)  # (batch_size, max_N, 300)
        mask = mask.to(device)      # (batch_size, max_N)
        labels = labels.to(device)  # (batch_size,)

        optimizer.zero_grad()

        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(inputs, mask)  # (batch_size, num_classes)
                loss = criterion(outputs, labels)
        else:
            outputs = model(inputs, mask)
            loss = criterion(outputs, labels)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels)
        total_samples += labels.size(0)

        # 메모리 관리
        del inputs, mask, labels, outputs, loss, preds
        torch.cuda.empty_cache()

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples

    return epoch_loss, epoch_acc.item()

def validate(model, dataloader, criterion, device):
    model.eval()
    val_running_loss = 0.0
    val_running_corrects = 0
    val_total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs, mask, labels = batch
            inputs = inputs.to(device)
            mask = mask.to(device)
            labels = labels.to(device)

            outputs = model(inputs, mask)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_running_corrects += torch.sum(preds == labels)
            val_total_samples += labels.size(0)

            # 메모리 관리
            del inputs, mask, labels, outputs, loss, preds
            torch.cuda.empty_cache()

    val_epoch_loss = val_running_loss / val_total_samples
    val_epoch_acc = val_running_corrects.double() / val_total_samples

    return val_epoch_loss, val_epoch_acc.item()

# ============================================
# Main 함수 정의
# ============================================

def main():
    # ============================================
    # 하이퍼파라미터 설정
    # ============================================

    num_epochs = 50
    learning_rate = 5e-5
    batch_size = 64
    num_layers = 4
    num_heads = 4
    d_model = 128
    dropout_rate = 0.1
    num_classes = 4  # 실제 클래스 수로 설정

    # ============================================
    # 데이터 로딩 및 전처리
    # ============================================

    ecg_data_A = load_and_preprocess_ecg_data(OFFSET, [2], dtype, DEBUG, CLUSTERING, PLOT)#[:1500]

    # 레이블 인덱스 생성
    label_to_idx_A = {label: idx for idx, label in enumerate(set(d['label'] for d in ecg_data_A))}
    data = [d['data'] for d in ecg_data_A]  # list of np.ndarray, each (N_i, 300)
    labels = np.array([label_to_idx_A[d['label']] for d in ecg_data_A])

    # 데이터 무결성 검사
    if not np.all(np.isfinite(labels)):
        raise ValueError("레이블에 비정상적인 값(NaN 또는 Inf)이 포함되어 있습니다.")

    # ============================================
    # 데이터 분할
    # ============================================

    train_data, val_data, train_labels, val_labels = train_test_split(
        data, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # ============================================
    # 클래스 가중치 계산
    # ============================================

    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # 클래스 가중치 무결성 검사
    if torch.isnan(class_weights_tensor).any() or torch.isinf(class_weights_tensor).any():
        raise ValueError("class_weights_tensor에 NaN 또는 INF 값이 포함되어 있습니다.")

    # ============================================
    # Dataset 및 DataLoader 생성
    # ============================================

    train_dataset = ECGDataset(train_data, train_labels, augment=True)
    val_dataset = ECGDataset(val_data, val_labels, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # CUDA IPC 경고 방지를 위해 0으로 설정
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # ============================================
    # 모델, 손실 함수, 옵티마이저, 스케줄러 설정
    # ============================================

    model = ECGModel(
        num_classes=num_classes,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout_rate=dropout_rate
    ).to(device)

    # 가중치 초기화
    def initialize_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0, std=0.1)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    model.apply(initialize_weights)

    # 손실 함수 정의 (클래스 가중치 적용)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # 옵티마이저 정의
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # 학습률 스케줄러 정의
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # ============================================
    # 학습 루프 (조기 종료 포함)
    # ============================================

    best_val_acc = 0.0
    best_model_state = None
    patience = 5
    trigger_times = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # 학습
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            scaler=None  # 혼합 정밀도 비활성화
        )
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # 검증
        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
            device
        )
        print(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}")

        # 성능 향상 체크
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            trigger_times = 0
            print("Validation accuracy improved. Saving best model.")
        else:
            trigger_times += 1
            print(f"No improvement in validation accuracy for {trigger_times} epochs.")

        # 조기 종료
        if trigger_times >= patience:
            print("Early stopping triggered.")
            break

    # ============================================
    # 최상의 모델 로드 및 저장
    # ============================================

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation accuracy: {best_val_acc:.4f}")

    # 모델 저장
    torch.save(model.state_dict(), 'best_ecg_transformer_model.pth')
    print("Best model saved to 'best_ecg_transformer_model.pth'.")

    # ============================================
    # 평가 및 시각화
    # ============================================

    # Attention Map 시각화
    visualize_attention(model, val_loader, num_layers=num_layers, num_heads=num_heads)

if __name__ == '__main__':
    main()
