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
import numpy as np
from sklearn.model_selection import train_test_split

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


# Modify the CombinedLoss to include class weights
class CombinedLossWeighted(nn.Module):
    def __init__(self, class_weights, classification_weight=1.0, reconstruction_weight=1.0):
        super(CombinedLossWeighted, self).__init__()
        self.classification_weight = classification_weight
        self.reconstruction_weight = reconstruction_weight
        self.classification_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.reconstruction_loss = nn.MSELoss()

    def forward(self, class_output, labels, recon_waveform, original_waveform):
        loss_cls = self.classification_loss(class_output, labels)
        loss_recon = self.reconstruction_loss(recon_waveform, original_waveform)
        loss = self.classification_weight * loss_cls + self.reconstruction_weight * loss_recon
        return loss, loss_cls, loss_recon

def split_dataset_for_deep_learning(
    image_dataset,
    source_mapping,
    labels_per_segment,
    grouped_segments,
    grouped_labels,
    train_size=0.8,
    random_state=42
):
    """
    Splits the dataset into training and validation sets based on source signals.

    Parameters:
    - image_dataset (np.ndarray): Array of shape (N, 4, H, W) containing transformed images.
    - source_mapping (np.ndarray): Array of shape (N,) mapping each segment to its source signal index.
    - labels_per_segment (np.ndarray): Array of shape (N,) containing labels for each segment.
    - grouped_segments (list of lists):
        Each sublist contains ECG segments from the same source signal.
        Example: [[seg1, seg2, seg3], [seg4, seg5, seg6], ...]
    - grouped_labels (list or array):
        List of labels corresponding to each source signal.
        Example: [label0, label1, label2, ...]
    - train_size (float): Proportion of the dataset to include in the training set (between 0.0 and 1.0).
    - random_state (int): Controls the shuffling applied to the data before applying the split.

    Returns:
    - train_set (dict): Dictionary containing training images, ECG segments, and labels.
    - val_set (dict): Dictionary containing validation images, ECG segments, and labels.
    """
    # Step 1: Identify unique source signals
    unique_sources = np.unique(source_mapping)
    print(f"Total unique source signals: {len(unique_sources)}")

    # Step 2: Split source signals into train and validation
    train_sources, val_sources = train_test_split(
        unique_sources,
        train_size=train_size,
        random_state=random_state,
        shuffle=True,
        stratify=None  # Adjust if stratification based on labels is needed
    )
    print(f"Training source signals: {len(train_sources)}")
    print(f"Validation source signals: {len(val_sources)}")

    # Step 3: Identify segment indices for training and validation
    train_indices = np.isin(source_mapping, train_sources)
    val_indices = np.isin(source_mapping, val_sources)

    # Step 4: Extract images and labels for training and validation
    train_images = image_dataset[train_indices]
    val_images = image_dataset[val_indices]

    train_labels = labels_per_segment[train_indices]
    val_labels = labels_per_segment[val_indices]

    # Step 5: Extract ECG segments for training and validation
    # Since grouped_segments is a list of lists, we need to map back
    # First, create a mapping from source index to segments
    # Assume that the source indices in source_mapping correspond to indices in grouped_segments
    # i.e., source_mapping contains integers from 0 to len(grouped_segments)-1
    # Verify this assumption
    if not np.all(unique_sources == np.arange(len(grouped_segments))):
        raise ValueError("source_mapping indices do not correspond to grouped_segments indices.")

    # Create a list to hold ECG segments for train and validation
    train_ecg_segments = []
    val_ecg_segments = []

    # Create a list to hold labels for train and validation (redundant with train_labels and val_labels)
    # But useful if needed separately
    train_grouped_labels = []
    val_grouped_labels = []

    # Iterate through each source signal in training sources and collect their segments
    for src_idx in train_sources:
        segments = grouped_segments[src_idx]
        train_ecg_segments.extend(segments)
        train_grouped_labels.append(grouped_labels[src_idx])

    # Similarly for validation
    for src_idx in val_sources:
        segments = grouped_segments[src_idx]
        val_ecg_segments.extend(segments)
        val_grouped_labels.append(grouped_labels[src_idx])

    # Optional: Verify that the number of ECG segments matches
    # train_ecg_segments should have the same number as train_images
    assert len(train_ecg_segments) == len(train_images), "Mismatch in training segments count."
    assert len(val_ecg_segments) == len(val_images), "Mismatch in validation segments count."

    # Prepare the output dictionaries
    train_set = {
        'images': train_images,                # Shape: (N_train, 4, H, W)
        'ecg_segments': np.array(train_ecg_segments),  # Shape: (N_train, segment_length)
        'labels': train_labels                  # Shape: (N_train,)
    }

    val_set = {
        'images': val_images,                  # Shape: (N_val, 4, H, W)
        'ecg_segments': np.array(val_ecg_segments),      # Shape: (N_val, segment_length)
        'labels': val_labels                    # Shape: (N_val,)
    }

    print(f"Training set: {train_set['images'].shape[0]} segments")
    print(f"Validation set: {val_set['images'].shape[0]} segments")

    return train_set, val_set

# 데이터 전처리 함수들 (데이터 증강 제외)
def normalize_signal(signal):
    min_val = np.min(signal)
    max_val = np.max(signal)
    normalized_signal = (signal - min_val) / (max_val - min_val + 1e-8)
    return normalized_signal

def align_ecg_signals(signals, labels, sample_rate=300, window_size=600, pre_r_peak=50, min_segments=3):
    """
    Align and segment ECG signals based on R-peaks.

    Parameters:
    - signals (list or array): List of ECG signals.
    - labels (list or array): Corresponding labels for each ECG signal.
    - sample_rate (int): Sampling rate of the ECG signals.
    - window_size (int): Number of data points in each window.
    - pre_r_peak (int): Number of data points before the R-peak to start the window.
    - min_segments (int): Minimum number of segments required per signal.

    Returns:
    - aligned_signals (list): List of segmented ECG windows.
    - aligned_labels (list): Corresponding labels for each segmented window.
    """
    grouped_segments = []  # List to hold lists of segments per signal
    grouped_labels = []  # List to hold labels per signal

    for idx, (signal, label) in enumerate(zip(signals, labels)):
        try:
            # Process the ECG signal to detect R-peaks
            working_data, measures = hp.process(signal, sample_rate=sample_rate, report_time=False)
            r_peaks = working_data['peaklist']

            # Check if the signal has the required number of R-peaks
            if len(r_peaks) < min_segments:
                raise ValueError(f"Not enough R-peaks: found {len(r_peaks)}, required {min_segments}")

            signal_segments = []  # List to hold segments for the current signal

            for seg_idx, r_peak in enumerate(r_peaks):
                start = r_peak - pre_r_peak
                end = start + window_size

                # Handle cases where the window exceeds signal boundaries
                if start < 0:
                    # Pad the beginning with zeros if start index is negative
                    pad_width = abs(start)
                    segment = np.pad(signal[0:end], (pad_width, 0), 'constant', constant_values=(0, 0))
                elif end > len(signal):
                    # Pad the end with zeros if end index exceeds signal length
                    pad_width = end - len(signal)
                    segment = np.pad(signal[start:], (0, pad_width), 'constant', constant_values=(0, 0))
                else:
                    # Extract the segment normally
                    segment = signal[start:end]

                signal_segments.append(segment)

                # Optional: Limit to a certain number of segments per signal
                # Uncomment the following lines if you want only the first 'min_segments' segments
                # if seg_idx + 1 >= min_segments:
                #     break

            # Ensure that at least 'min_segments' are present
            if len(signal_segments) < min_segments:
                raise ValueError(
                    f"After processing, only {len(signal_segments)} segments found, required {min_segments}")

            # Append to grouped lists
            grouped_segments.append(signal_segments)
            grouped_labels.append(label)


            logging.info(f"Processed and saved segments for signal index {idx}.")

        except Exception as e:
                logging.error(f"Error aligning signal index {idx}: {e}")
                continue  # Skip this signal and move to the next

    return grouped_segments, grouped_labels

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


def transform_to_images_grouped_with_labels(grouped_segments, grouped_labels):
    """
    Transforms grouped ECG segments into image representations and structures the dataset,
    returning image data along with their corresponding source mappings and labels.

    Parameters:
    - grouped_segments (list of lists):
        Each sublist contains ECG segments from the same source signal.
        Example: [[seg1, seg2, seg3], [seg4, seg5, seg6], ...]
    - grouped_labels (list or array):
        List of labels corresponding to each source signal.
        Example: [label0, label1, label2, ...]

    Returns:
    - image_dataset (np.ndarray):
        Array of shape (N, 4, H, W) containing transformed images.
    - source_mapping (np.ndarray):
        Array of shape (N,) mapping each segment to its source signal index.
    - labels_per_segment (np.ndarray):
        Array of shape (N,) containing labels for each segment.
    """
    # Initialize transformers
    transformers = {
        'RP': RecurrencePlot(),
        'GASF': GramianAngularField(method='summation'),
        'GADF': GramianAngularField(method='difference'),
        'MTF': MarkovTransitionField(n_bins=4),
    }

    transformed_images = []
    source_mapping = []  # To keep track of source signal index for each segment
    labels_per_segment = []  # To keep track of labels for each segment

    for source_idx, (segments, label) in enumerate(zip(grouped_segments, grouped_labels)):
        for seg_idx, signal in enumerate(segments):
            try:
                transformed = []
                for key in ['RP', 'GASF', 'GADF', 'MTF']:
                    transformer = transformers[key]
                    img = transformer.transform(signal.reshape(1, -1))[0]

                    # Image normalization: Standard Score (Z-score)
                    img = (img - np.mean(img)) / (np.std(img) + 1e-8)
                    transformed.append(img)

                # Stack images along the channel dimension to get shape (4, H, W)
                transformed = np.stack(transformed, axis=0)
                transformed_images.append(transformed)
                source_mapping.append(source_idx)  # Map to source signal index
                labels_per_segment.append(label)  # Assign label to segment

            except Exception as e:
                logging.error(f"Error transforming segment {seg_idx} from source {source_idx}: {e}")
                continue  # Skip this segment and proceed

    # Convert lists to NumPy arrays
    image_dataset = np.array(transformed_images)  # Shape: (N, 4, H, W)
    source_mapping = np.array(source_mapping)  # Shape: (N,)
    labels_per_segment = np.array(labels_per_segment)  # Shape: (N,)

    return image_dataset, source_mapping, labels_per_segment

# Define the Image Encoder

class ECGArrhythmiaClassifier(nn.Module):
    def __init__(self, num_classes, waveform_length=600, image_size=(4, 100, 100)):
        super(ECGArrhythmiaClassifier, self).__init__()

        # Image Branch: Using ResNet18 as feature extractor
        self.image_size = image_size
        self.image_model = models.resnet18(pretrained=True)
        # Modify the first convolutional layer to accept 4 channels instead of 3
        self.image_model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Remove the final fully connected layer
        self.image_feature_extractor = nn.Sequential(
            *list(self.image_model.children())[:-1])  # Output: (batch, 512, 1, 1)
        self.image_embedding_size = 512

        # Waveform Branch: 1D CNN
        self.waveform_model = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Output: (batch, 64, 1)
        )
        self.waveform_embedding_size = 64

        # Fusion Layer
        self.fusion = nn.Linear(self.image_embedding_size + self.waveform_embedding_size, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)
        )

        # Reconstruction Head
        self.reconstructor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, waveform_length)
        )

        # Attention Layer (Self-Attention on Fused Embedding)
        # self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=16, num_heads=8, batch_first=True)

    def forward(self, image, waveform):
        """
        Forward pass for the model.

        Parameters:
        - image: Tensor of shape (batch_size, 4, H, W)
        - waveform: Tensor of shape (batch_size, 600)

        Returns:
        - class_output: Tensor of shape (batch_size, num_classes)
        - reconstructed_waveform: Tensor of shape (batch_size, 600)
        - attention_weights: Tensor of shape (batch_size, num_heads, seq_len, seq_len)
        """
        # Image Branch
        img_features = self.image_feature_extractor(image)  # (batch, 512, 1, 1)
        img_features = img_features.view(img_features.size(0), -1)  # (batch, 512)

        # Waveform Branch
        waveform = waveform.unsqueeze(1)  # (batch, 1, 600)
        wf_features = self.waveform_model(waveform)  # (batch, 64, 1)
        wf_features = wf_features.view(wf_features.size(0), -1)  # (batch, 64)

        # Fusion
        fused = torch.cat((img_features, wf_features), dim=1)  # (batch, 512 + 64 = 576)
        fused = self.fusion(fused)  # (batch, 256)
        fused = self.relu(fused)
        fused = self.dropout(fused)

        # Attention (Self-Attention requires sequence length; reshape accordingly)
        # Here, we'll treat the fused embedding as a sequence of length 1
        # To make attention meaningful, consider splitting the fused embedding into multiple tokens
        # For simplicity, we'll expand the fused embedding to a sequence of fixed length
        # Example: Split into 16 tokens of size 16 each
        seq_len = 16
        token_dim = 16  # 16 * 16 = 256
        fused = fused.view(fused.size(0), seq_len, token_dim)  # (batch, seq_len, token_dim)

        attn_output, attn_weights = self.attention(fused, fused, fused)  # attn_output: (batch, seq_len, 256)

        # Aggregate attended features
        attn_output = attn_output.mean(dim=1)  # (batch, 256)

        # Classification Output
        class_output = self.classifier(attn_output)  # (batch, num_classes)

        # Reconstruction Output
        reconstructed_waveform = self.reconstructor(attn_output)  # (batch, 600)

        return class_output, reconstructed_waveform, attn_weights


class ECGDataset(torch.utils.data.Dataset):
    def __init__(self, grouped_segments, grouped_labels, transform=None, image_normalize=None):
        """
        Parameters:
        - grouped_segments: List of lists. Each sublist contains segments (images and waveforms) from one ECG signal.
                           Example: [[segment1, segment2, ...], [segment1, segment2, ...], ...]
                           Each segment is a dict with keys: 'images' (4xHxW tensor), 'waveform' (1x600 tensor)
        - grouped_labels: List of labels corresponding to each ECG signal.
        - transform: Optional transformations to apply to image data.
        - image_normalize: Optional normalization to apply to image data.
        """
        self.grouped_segments = grouped_segments
        self.grouped_labels = grouped_labels
        self.transform = transform
        self.image_normalize = image_normalize

    def __len__(self):
        return len(self.grouped_labels)

    def __getitem__(self, idx):
        """
        Returns:
        - images: Tensor of shape (num_segments, 4, H, W)
        - waveforms: Tensor of shape (num_segments, 600)
        - label: Integer label
        """
        segments = self.grouped_segments[idx]
        label = self.grouped_labels[idx]

        # Extract images and waveforms
        images = [segment['images'] for segment in segments]  # List of (4, H, W) tensors
        waveforms = [segment['waveform'] for segment in segments]  # List of (600,) tensors

        # Stack into tensors
        images = torch.stack(images)  # (num_segments, 4, H, W)
        waveforms = torch.stack(waveforms)  # (num_segments, 600)

        # Apply transformations if any
        if self.transform:
            # Apply transform to each image in the batch
            images = torch.stack([self.transform(img) for img in images])

        if self.image_normalize:
            # Normalize each image channel
            images = self.image_normalize(images)

        return images, waveforms, label

def grouped_collate_fn(batch):
    """
    Parameters:
    - batch: List of tuples (images, waveforms, label)
        - images: Tensor of shape (num_segments, 4, H, W)
        - waveforms: Tensor of shape (num_segments, 600)
        - label: Integer

    Returns:
    - batch_images: Padded Tensor of shape (batch_size, max_num_segments, 4, H, W)
    - batch_waveforms: Padded Tensor of shape (batch_size, max_num_segments, 600)
    - labels: Tensor of shape (batch_size,)
    - segment_mask: Tensor of shape (batch_size, max_num_segments), indicating valid segments
    """
    images, waveforms, labels = zip(*batch)

    # Determine max number of segments in the batch
    max_num_segments = max([img.size(0) for img in images])

    # Pad images and waveforms
    padded_images = []
    padded_waveforms = []
    segment_mask = []

    for img, wf in zip(images, waveforms):
        num_segments = img.size(0)
        pad_segments = max_num_segments - num_segments

        if pad_segments > 0:
            # Pad images with zeros
            pad_img = torch.zeros(pad_segments, *img.size()[1:], dtype=img.dtype)
            img = torch.cat([img, pad_img], dim=0)

            # Pad waveforms with zeros
            pad_wf = torch.zeros(pad_segments, wf.size(1), dtype=wf.dtype)
            wf = torch.cat([wf, pad_wf], dim=0)

            # Mask
            mask = torch.cat([torch.ones(num_segments), torch.zeros(pad_segments)])
        else:
            mask = torch.ones(max_num_segments)

        padded_images.append(img)
        padded_waveforms.append(wf)
        segment_mask.append(mask)

    batch_images = torch.stack(padded_images)  # (batch_size, max_num_segments, 4, H, W)
    batch_waveforms = torch.stack(padded_waveforms)  # (batch_size, max_num_segments, 600)
    labels = torch.tensor(labels, dtype=torch.long)  # (batch_size,)
    segment_mask = torch.stack(segment_mask)  # (batch_size, max_num_segments)

    return batch_images, batch_waveforms, labels, segment_mask

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
# Training and Validation Functions
def train_one_epoch(model, dataloader, optimizer, criterion_cls, criterion_rec, device):
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_rec_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        images, raw_signals, labels = batch
        images = images.to(device)
        ecgs = raw_signals.to(device)
        labels = labels.to(device)

        logits, reconstructed_ecg, attn_weights, top_segment_indices = model(images, ecgs)

        # Classification Loss
        loss_cls = criterion_cls(logits, labels)

        # Reconstruction Loss
        B = images.size(0)
        top_ecgs = ecgs[torch.arange(B), top_segment_indices, :].unsqueeze(1)  # (B, 1, 300)
        loss_rec = criterion_rec(reconstructed_ecg, top_ecgs)

        # Total Loss with weighting for the reconstruction loss
        loss = loss_cls + 0.1 * loss_rec

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cls_loss += loss_cls.item()
        total_rec_loss += loss_rec.item()

    avg_loss = total_loss / len(dataloader)
    avg_cls_loss = total_cls_loss / len(dataloader)
    avg_rec_loss = total_rec_loss / len(dataloader)
    print(
        f"Train Loss: {avg_loss:.4f}, Classification Loss: {avg_cls_loss:.4f}, Reconstruction Loss: {avg_rec_loss:.4f}")
    return avg_loss

# 검증 함수
def validate(model, dataloader, criterion_cls, criterion_rec, device):
    model.eval()
    total_loss = 0
    total_cls_loss = 0
    total_rec_loss = 0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images, raw_signals, labels = batch
            images = images.to(device)
            ecgs = raw_signals.to(device)
            labels = labels.to(device)

            logits, reconstructed_ecg, attn_weights, top_segment_indices = model(images, ecgs)

            # Classification Loss
            loss_cls = criterion_cls(logits, labels)

            # Reconstruction Loss
            B = images.size(0)
            top_ecgs = ecgs[torch.arange(B), top_segment_indices, :].unsqueeze(1)  # (B, 1, 300)
            loss_rec = criterion_rec(reconstructed_ecg, top_ecgs)

            # Total Loss with weighting for the reconstruction loss
            loss = loss_cls + 0.1 * loss_rec

            total_loss += loss.item()
            total_cls_loss += loss_cls.item()
            total_rec_loss += loss_rec.item()

            # Calculate accuracy (for multi-label, you might use a different metric)
            preds = (torch.sigmoid(logits) > 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_samples += labels.numel()

        avg_loss = total_loss / len(dataloader)
        avg_cls_loss = total_cls_loss / len(dataloader)
        avg_rec_loss = total_rec_loss / len(dataloader)
        accuracy = total_correct / total_samples
        print(
            f"Validation Loss: {avg_loss:.4f}, Classification Loss: {avg_cls_loss:.4f}, Reconstruction Loss: {avg_rec_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy

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
    aligned_signals, aligned_labels = align_ecg_signals(data, labels, sample_rate=100,window_size=300)
    logging.info(f"Aligned signals: {len(aligned_signals)}")

    # 4.3. 신호 정규화
    normalized_signals = normalize_signals(aligned_signals)
    logging.info(f"Normalized signals: {len(normalized_signals)}")

    image_dataset, source_mapping, labels_per_segment = transform_to_images_grouped_with_labels(
        grouped_segments=normalized_signals,
        grouped_labels=aligned_labels
    )

    # Step 3: Split Dataset into Training and Validation
    train_set, val_set = split_dataset_for_deep_learning(
        image_dataset=image_dataset,
        source_mapping=source_mapping,
        labels_per_segment=labels_per_segment,
        grouped_segments=normalized_signals,
        grouped_labels=aligned_labels,
        train_size=0.8,
        random_state=42
    )

    # 6. 클래스 가중치 계산
    class_weights = compute_class_weight('balanced', classes=np.unique(train_set['labels']), y=train_set['labels'])
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    if torch.isnan(class_weights_tensor).any() or torch.isinf(class_weights_tensor).any():
        logging.error("class_weights_tensor에 NaN 또는 INF 값이 포함되어 있습니다.")
        return

    # 7. 원시 신호 준비 (원시 신호는 aligned_signals)
    # aligned_signals는 list of numpy arrays with shape (300,)
    # raw_signals는 list of numpy arrays with shape (1, 300)
    # raw_signals = [signal.reshape(1, -1) for signal in normalized_signals]
    # # Split train and val raw_signals accordingly
    # train_raw_signals = [raw_signals[i] for i in train_labels]
    # val_raw_signals = [raw_signals[i] for i in val_labels]

    # train_set = {
    #     'images': train_images,                # Shape: (N_train, 4, H, W)
    #     'ecg_segments': np.array(train_ecg_segments),  # Shape: (N_train, segment_length)
    #     'labels': train_labels                  # Shape: (N_train,)
    # }

    # 8. Dataset 및 DataLoader 설정
    train_dataset = ECGDataset(train_set['images'], train_set['ecg_segments'], train_set['labels'])
    val_dataset = ECGDataset(val_set['images'], val_set['ecg_segments'], val_set['labels'])

    # 클래스 불균형을 해결하기 위한 WeightedRandomSampler 사용
    class_counts = np.bincount(train_set['labels'])
    class_weights_sampler = 1. / class_counts
    sample_weights = class_weights_sampler[train_set['labels']]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    num_workers = 0  # Unix 환경에서는 4 이상으로 설정 가능
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=False,  # WeightedRandomSampler 사용 시 shuffle=False
        collate_fn=grouped_collate_fn,  # 기본 collate_fn 사용
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=grouped_collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    embed_dim = 128
    num_classes = 10  # Number of arrhythmia types
    num_segments = 5
    batch_size = 8
    num_epochs = 10

    # Initialize Model, Optimizer, and Loss Functions
    model = ECGArrhythmiaClassifier(embed_dim, num_classes).to(device)

    class_weights = torch.ones(num_classes).to(device)  # Adjust as needed for imbalanced data
    criterion = CombinedLossWeighted(class_weights=class_weights, classification_weight=1.0, reconstruction_weight=1.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 50

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_cls_loss = 0.0
        running_recon_loss = 0.0

        for images, waveforms, labels in train_loader:
            images = images.to(device)
            waveforms = waveforms.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            class_output, recon_waveform, _ = model(images, waveforms)
            loss, loss_cls, loss_recon = criterion(class_output, labels, recon_waveform, waveforms)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_cls_loss += loss_cls.item() * images.size(0)
            running_recon_loss += loss_recon.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_cls_loss = running_cls_loss / len(train_loader.dataset)
        epoch_recon_loss = running_recon_loss / len(train_loader.dataset)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_cls_loss = 0.0
        val_recon_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, waveforms, labels in val_loader:
                images = images.to(device)
                waveforms = waveforms.to(device)
                labels = labels.to(device)

                class_output, recon_waveform, _ = model(images, waveforms)
                loss, loss_cls, loss_recon = criterion(class_output, labels, recon_waveform, waveforms)

                val_loss += loss.item() * images.size(0)
                val_cls_loss += loss_cls.item() * images.size(0)
                val_recon_loss += loss_recon.item() * images.size(0)

                # Classification Accuracy
                _, predicted = torch.max(class_output, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_cls_loss = val_cls_loss / len(val_loader.dataset)
        val_epoch_recon_loss = val_recon_loss / len(val_loader.dataset)
        val_accuracy = correct / total

        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"Train Loss: {epoch_loss:.4f} (Cls: {epoch_cls_loss:.4f}, Recon: {epoch_recon_loss:.4f}) | "
              f"Val Loss: {val_epoch_loss:.4f} (Cls: {val_epoch_cls_loss:.4f}, Recon: {val_epoch_recon_loss:.4f}) | "
              f"Val Acc: {val_accuracy * 100:.2f}%")

if __name__ == '__main__':
    main()
