# ecg_classification.py

import logging
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from sklearn.utils import class_weight
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import heartpy as hp
from ecg_cluster import load_and_preprocess_ecg_data
from pyts.image import MarkovTransitionField, RecurrencePlot, GramianAngularField
from sklearn.model_selection import train_test_split
import cv2
from stockwell import st
# ===============================
# 1. Setting Seeds for Reproducibility
# ===============================

def apply_2d_dct(image):
    dct = cv2.dct(np.float32(image))
    return dct

def split_dataset_for_deep_learning(
        image_dataset,
        source_mapping,
        labels_per_segment,
        grouped_segments,
        grouped_labels,
        train_size=0.8,
        random_state=42,
        stratify_labels=None,
        return_grouped=False
):
    """
    Splits the dataset into training and validation sets based on source signals.

    Parameters:
    - image_dataset (np.ndarray): Array of shape (N, 4, H, W) containing transformed images.
    - source_mapping (np.ndarray): Array of shape (N,) mapping each segment to its source signal index.
    - labels_per_segment (np.ndarray): Array of shape (N,) containing labels for each segment.
    - grouped_segments (list of lists):
        Each sublist contains ECG segments from the same source signal.
        Example: [[seg1, seg2, ...], [seg4, seg5, ...], ...]
    - grouped_labels (list or array):
        List of labels corresponding to each source signal.
        Example: [label0, label1, label2, ...]
    - train_size (float): Proportion of the dataset to include in the training set (between 0.0 and 1.0).
    - random_state (int): Controls the shuffling applied to the data before applying the split.
    - stratify_labels (array-like, optional): Labels to use for stratification. Must be the same length as `grouped_labels`.
        If None, no stratification is performed.
    - return_grouped (bool): If True, returns grouped data. If False, returns per-segment data.

    Returns:
    - If return_grouped is False:
        - train_set (dict): Dictionary containing training images, ECG segments, and labels.
        - val_set (dict): Dictionary containing validation images, ECG segments, and labels.
    - If return_grouped is True:
        - train_grouped_segments (list of lists): Segments grouped for training.
        - val_grouped_segments (list of lists): Segments grouped for validation.
        - train_grouped_labels (list or array): Labels for training groups.
        - val_grouped_labels (list or array): Labels for validation groups.
    """
    # Step 1: Verify that source_mapping indices correspond to grouped_segments indices
    unique_sources = np.unique(source_mapping)
    num_grouped_segments = len(grouped_segments)

    if not np.all(unique_sources < num_grouped_segments):
        raise ValueError(
            f"source_mapping contains source indices that exceed the number of grouped_segments. "
            f"Max source index in source_mapping: {unique_sources.max()}, "
            f"Number of grouped_segments: {num_grouped_segments}"
        )

    print(f"Total unique source signals: {len(unique_sources)}")

    # Step 2: Prepare labels for stratification if provided
    if stratify_labels is not None:
        if len(stratify_labels) != len(grouped_labels):
            raise ValueError(
                "Length of stratify_labels must match the number of grouped_labels."
            )
        stratify_split = stratify_labels
    else:
        stratify_split = None

    # Step 3: Split source signals into train and validation
    train_sources, val_sources = train_test_split(
        unique_sources,
        train_size=train_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify_split  # Can be None or labels per group
    )
    print(f"Training source signals: {len(train_sources)}")
    print(f"Validation source signals: {len(val_sources)}")

    # Step 4: Identify segment indices for training and validation
    train_indices = np.isin(source_mapping, train_sources)
    val_indices = np.isin(source_mapping, val_sources)

    # Step 5: Extract images and labels for training and validation
    train_images = image_dataset[train_indices]
    val_images = image_dataset[val_indices]

    train_labels = labels_per_segment[train_indices]
    val_labels = labels_per_segment[val_indices]

    # Step 6: Extract ECG segments for training and validation based on grouped_sources
    # Create mappings from source to segments
    # Assume that source_mapping contains integers from 0 to len(grouped_segments)-1
    # Each source index maps to grouped_segments[source_idx]

    if return_grouped:
        # Step 6: Collect grouped images based on train_sources and val_sources
        # Initialize lists to hold grouped images
        train_grouped_images = []
        val_grouped_images = []

        # Initialize lists to hold grouped segments and labels
        train_grouped_segments = []
        val_grouped_segments = []

        train_grouped_labels = []
        val_grouped_labels = []

        # Mapping from source index to segment indices
        # Assuming that source_mapping indices correspond to grouped_segments indices
        # i.e., source_mapping contains integers from 0 to len(grouped_segments)-1
        # and grouped_segments[src_idx] corresponds to source_mapping == src_idx

        # Create a dictionary for quick lookup of segment indices per source
        from collections import defaultdict
        source_to_indices = defaultdict(list)
        for idx, src_idx in enumerate(source_mapping):
            source_to_indices[src_idx].append(idx)

        # Collect training grouped data
        for src_idx in train_sources:
            segment_list = grouped_segments[src_idx]  # List of segments
            train_grouped_segments.append(segment_list)

            # Get all image indices for this source
            img_indices = source_to_indices[src_idx]
            group_images = image_dataset[img_indices]  # Shape: (num_segments_in_group, 4, H, W)
            train_grouped_images.append(group_images)

            # Get the label for this group
            label = grouped_labels[src_idx]
            train_grouped_labels.append(label)

        # Collect validation grouped data
        for src_idx in val_sources:
            segment_list = grouped_segments[src_idx]  # List of segments
            val_grouped_segments.append(segment_list)

            # Get all image indices for this source
            img_indices = source_to_indices[src_idx]
            group_images = image_dataset[img_indices]  # Shape: (num_segments_in_group, 4, H, W)
            val_grouped_images.append(group_images)

            # Get the label for this group
            label = grouped_labels[src_idx]
            val_grouped_labels.append(label)

        # Optional: Verify alignment
        # Number of segments in train_grouped_segments should equal number of images in train_grouped_images
        total_train_segments = sum([len(group) for group in train_grouped_segments])
        total_train_images = sum([group.shape[0] for group in train_grouped_images])
        assert total_train_segments == total_train_images, "Mismatch in training segments and images count."

        # Similarly for validation
        total_val_segments = sum([len(group) for group in val_grouped_segments])
        total_val_images = sum([group.shape[0] for group in val_grouped_images])
        assert total_val_segments == total_val_images, "Mismatch in validation segments and images count."

        print(f"Training set: {total_train_segments} segments across {len(train_grouped_segments)} groups")
        print(f"Validation set: {total_val_segments} segments across {len(val_grouped_segments)} groups")

        return (
            train_grouped_images,
            val_grouped_images,
            train_grouped_segments,
            val_grouped_segments,
            train_grouped_labels,
            val_grouped_labels
        )
    # Step 7: Flatten the grouped segments and labels to match per-segment data
    # This step is optional and depends on whether you need per-segment data
    # Here, we assume that each group has consistent labels across segments
    # Therefore, labels_per_segment can be derived from grouped_labels

    # Create a list to hold training and validation segments
    train_ecg_segments = []
    val_ecg_segments = []

    # Create labels per segment based on grouped_labels
    # Assuming all segments in a group share the same label
    for src_idx in train_sources:
        segments = grouped_segments[src_idx]
        train_ecg_segments.extend(segments)

    for src_idx in val_sources:
        segments = grouped_segments[src_idx]
        val_ecg_segments.extend(segments)

    # Optional: Verify alignment
    if len(train_ecg_segments) != len(train_images):
        raise ValueError(
            f"Number of training ECG segments ({len(train_ecg_segments)}) does not match "
            f"number of training images ({len(train_images)})."
        )

    if len(val_ecg_segments) != len(val_images):
        raise ValueError(
            f"Number of validation ECG segments ({len(val_ecg_segments)}) does not match "
            f"number of validation images ({len(val_images)})."
        )

    # Step 8: Prepare labels per segment based on grouped_labels
    # Assuming labels_per_segment corresponds to grouped_labels
    # If labels_per_segment are already correctly aligned, you can retain them
    # Otherwise, derive them from grouped_labels
    # Here, we derive labels_per_segment from grouped_labels for consistency

    # Initialize empty lists
    train_labels_per_segment = []
    val_labels_per_segment = []

    for src_idx in train_sources:
        label = grouped_labels[src_idx]
        num_segments = len(grouped_segments[src_idx])
        train_labels_per_segment.extend([label] * num_segments)

    for src_idx in val_sources:
        label = grouped_labels[src_idx]
        num_segments = len(grouped_segments[src_idx])
        val_labels_per_segment.extend([label] * num_segments)

    # Convert lists to numpy arrays
    train_labels_per_segment = np.array(train_labels_per_segment)
    val_labels_per_segment = np.array(val_labels_per_segment)

    # Step 9: Prepare the output dictionaries
    train_set = {
        'images': train_images,  # Shape: (N_train, 4, H, W)
        'ecg_segments': np.array(train_ecg_segments),  # Shape: (N_train, segment_length)
        'labels': train_labels_per_segment  # Shape: (N_train,)
    }

    val_set = {
        'images': val_images,  # Shape: (N_val, 4, H, W)
        'ecg_segments': np.array(val_ecg_segments),  # Shape: (N_val, segment_length)
        'labels': val_labels_per_segment  # Shape: (N_val,)
    }

    print(f"Training set: {train_set['images'].shape[0]} segments")
    print(f"Validation set: {val_set['images'].shape[0]} segments")

    return train_set, val_set

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
                for key in ['RP', 'GASF',  'MTF']:
                    transformer = transformers[key]
                    img = transformer.transform(signal.reshape(1, -1))[0]
                    # img = apply_2d_dct(img)

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


    # 360 144 / 180 = 324
    # 100 40 / 50


def align_ecg_signals(signals, labels, sample_rate=300, window_size=600, min_segments=3):
    """
    Align and segment ECG signals based on a fixed window size.

    Parameters:
    - signals (list or array): List of ECG signals.
    - labels (list or array): Corresponding labels for each ECG signal.
    - sample_rate (int): Sampling rate of the ECG signals.
    - window_size (int): Number of data points in each window.
    - min_segments (int): Minimum number of segments required per signal.

    Returns:
    - aligned_signals (list): List of segmented ECG windows.
    - aligned_labels (list): Corresponding labels for each segmented window.
    """
    grouped_segments = []  # List to hold lists of segments per signal
    grouped_labels = []  # List to hold labels per signal

    for idx, (signal, label) in enumerate(zip(signals, labels)):
        try:
            signal_segments = []  # List to hold segments for the current signal

            # Segment the signal into fixed windows of `window_size`
            num_segments = len(signal) // window_size

            # If fewer segments than required, raise an error
            if num_segments < min_segments:
                raise ValueError(f"Not enough segments: found {num_segments}, required {min_segments}")

            for seg_idx in range(num_segments):
                start = seg_idx * window_size
                end = start + window_size

                # Handle cases where the window exceeds signal boundaries
                if end > len(signal):
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


def set_seed(seed=42):
    """
    Sets the seed for generating random numbers.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)


# ===============================
# 2. Defining the Custom Dataset
# ===============================

class GroupedECGDataset(Dataset):
    def __init__(self, grouped_segments, grouped_waveforms, grouped_labels, transform=None, image_normalize=None):
        """
        Initializes the dataset with grouped segments and labels.

        Parameters:
        - grouped_segments: List of lists. Each sublist contains segments from one ECG signal.
                           Each segment is a dict with keys: 'images' (4xHxW tensor), 'waveform' (1x600 tensor)
        - grouped_labels: List of integer labels corresponding to each ECG signal.
        - transform: Optional torchvision transforms to apply to images.
        - image_normalize: Optional normalization to apply to images.
        """
        self.grouped_segments = grouped_segments
        self.grouped_waveforms = grouped_waveforms
        self.grouped_labels = grouped_labels
        self.transform = transform
        self.image_normalize = image_normalize

    def __len__(self):
        return len(self.grouped_labels)

    def __getitem__(self, idx):
        """
        Retrieves the grouped segments and label for a given index.

        Returns:
        - images: Tensor of shape (num_segments, 4, H, W)
        - waveforms: Tensor of shape (num_segments, 600)
        - label: Integer label
        """
        segments = torch.tensor(self.grouped_segments[idx],dtype=torch.float32)
        waveform = torch.tensor(self.grouped_waveforms[idx],dtype=torch.float32)
        label =  torch.tensor(self.grouped_labels[idx],dtype=torch.long)

        # Extract images and waveforms
        # images = [segment for segment in segments]  # List of (4, H, W) tensors
        # waveforms = [segment for segment in waveform]  # List of (600,) tensors

        # Stack into tensors
        # images = torch.stack(images)  # (num_segments, 4, H, W)
        # waveforms = torch.stack(waveforms)  # (num_segments, 600)

        # Apply transformations if any
        # if self.transform:
            # Apply transform to each image in the batch
            # images = torch.stack([self.transform(img) for img in images])

        # if self.image_normalize:
            # Normalize each image channel
            # images = self.image_normalize(images)

        return segments, waveform, label


# ===============================
# 3. Defining the Custom Collate Function
# ===============================

def grouped_collate_fn(batch):
    """
    Custom collate function to handle grouped data.

    Parameters:
    - batch: List of tuples (images, ecg_segments, label)
        - images: Tensor of shape (num_segments, 4, H, W)
        - ecg_segments: List of ECG segments
        - label: Integer

    Returns:
    - batch_images: Padded Tensor of shape (batch_size, max_num_segments, 4, H, W)
    - batch_waveforms: Padded Tensor of shape (batch_size, max_num_segments, 600)
    - labels: Tensor of shape (batch_size,)
    - segment_mask: Tensor of shape (batch_size, max_num_segments), indicating valid segments
    """
    images, ecg_segments, labels = zip(*batch)

    # Determine max number of segments in the batch
    max_num_segments = max([img.size(0) for img in images])

    # Initialize lists for padded data
    padded_images = []
    padded_waveforms = []
    segment_mask = []

    for img, seg in zip(images, ecg_segments):
        num_segments = img.size(0)
        pad_segments = max_num_segments - num_segments

        if pad_segments > 0:
            # Pad images with zeros on the same device as img (should be CPU)
            pad_img = torch.zeros(pad_segments, img.size(1), img.size(2), img.size(3), dtype=img.dtype,
                                  device=img.device)
            img = torch.cat([img, pad_img], dim=0)

            # Pad waveforms with zeros (assuming each segment's ECG waveform is of length 600)
            seg_waveforms = torch.tensor(seg, dtype=torch.float32, device='cpu')  # Ensure on CPU
            pad_wf = torch.zeros(pad_segments, seg_waveforms.size(1), dtype=seg_waveforms.dtype, device='cpu')
            seg_waveforms = torch.cat([seg_waveforms, pad_wf], dim=0)

            # Create mask
            mask = torch.cat([torch.ones(num_segments, device='cpu'), torch.zeros(pad_segments, device='cpu')])
        else:
            seg_waveforms = torch.tensor(seg, dtype=torch.float32, device='cpu')  # Ensure on CPU
            mask = torch.ones(max_num_segments, device='cpu')

        padded_images.append(img)  # (max_num_segments, 4, H, W)
        padded_waveforms.append(seg_waveforms)  # (max_num_segments, 600)
        segment_mask.append(mask)  # (max_num_segments,)

    # Stack all padded data
    batch_images = torch.stack(padded_images)  # (batch_size, max_num_segments, 4, H, W)
    batch_waveforms = torch.stack(padded_waveforms)  # (batch_size, max_num_segments, 600)
    labels = torch.tensor(labels, dtype=torch.long, device='cpu')  # (batch_size,)
    segment_mask = torch.stack(segment_mask)  # (batch_size, max_num_segments)

    return batch_images, batch_waveforms, labels, segment_mask


# ===============================
# 4. Defining the Hierarchical Model
# ===============================

class HierarchicalECGClassifier(nn.Module):
    def __init__(self, num_classes, waveform_length=600, image_size=(4, 100, 100)):
        """
        Initializes the hierarchical ECG classifier.

        Parameters:
        - num_classes: Number of arrhythmia classes.
        - waveform_length: Length of the ECG waveform.
        - image_size: Tuple representing the image dimensions (C, H, W).
        """
        super(HierarchicalECGClassifier, self).__init__()

        # Segment-Level Encoder
        self.segment_encoder = SegmentEncoder(image_size=image_size, embed_dim=256)

        # Aggregation Layer: Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=0.1,
                                                   activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Classification and Reconstruction Heads
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)
        )

        self.reconstructor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, waveform_length)
        )

    def forward(self, images, waveforms, segment_mask):
        """
        Forward pass of the hierarchical model.

        Parameters:
        - images: Tensor of shape (batch_size, max_num_segments, 4, H, W)
        - waveforms: Tensor of shape (batch_size, max_num_segments, 600)
        - segment_mask: Tensor of shape (batch_size, max_num_segments)

        Returns:
        - class_output: Tensor of shape (batch_size, num_classes)
        - reconstructed_waveform: Tensor of shape (batch_size, 600)
        """
        batch_size, max_num_segments, C, H, W = images.size()

        # Reshape to process all segments collectively
        images = images.view(batch_size * max_num_segments, C, H, W)  # (B*M, 4, H, W)
        waveforms = waveforms.view(batch_size * max_num_segments, -1)  # (B*M, 600)

        # Encode each segment
        segment_embeddings = self.segment_encoder(images, waveforms)  # (B*M, 256)

        # Reshape back to (batch_size, max_num_segments, embed_dim)
        segment_embeddings = segment_embeddings.view(batch_size, max_num_segments, -1)  # (B, M, 256)

        # Prepare for Transformer: (max_num_segments, batch_size, embed_dim)
        transformer_input = segment_embeddings.permute(1, 0, 2)  # (M, B, 256)

        # Apply Transformer Encoder
        transformer_output = self.transformer_encoder(transformer_input)  # (M, B, 256)

        # Reshape back to (batch_size, max_num_segments, embed_dim)
        transformer_output = transformer_output.permute(1, 0, 2)  # (B, M, 256)

        # Apply mask: zero out padded segments
        segment_mask = segment_mask.unsqueeze(-1)  # (B, M, 1)
        transformer_output = transformer_output * segment_mask  # (B, M, 256)

        # Aggregate embeddings: Mean pooling over valid segments
        sum_embeddings = torch.sum(transformer_output, dim=1)  # (B, 256)
        num_valid_segments = torch.sum(segment_mask, dim=1).clamp(min=1)  # (B, 1)
        aggregated_embedding = sum_embeddings / num_valid_segments  # (B, 256)

        # Classification and Reconstruction
        class_output = self.classifier(aggregated_embedding)  # (B, num_classes)
        reconstructed_waveform = self.reconstructor(aggregated_embedding)  # (B, 600)

        return class_output, reconstructed_waveform


# ===============================
# 5. Defining the Segment Encoder
# ===============================

class SegmentEncoder(nn.Module):
    def __init__(self, image_size=(4, 100, 100), embed_dim=256):
        """
        Initializes the segment-level encoder.

        Parameters:
        - image_size: Tuple representing the image dimensions (C, H, W).
        - embed_dim: Dimension of the output embedding.
        """
        super(SegmentEncoder, self).__init__()
        self.embed_dim = embed_dim

        # Image Branch: Modified ResNet18
        self.image_model = models.resnet18(pretrained=True)
        # Modify the first convolutional layer to accept 4 channels instead of 3
        self.image_model.conv1 = nn.Conv2d(
            in_channels=image_size[0],
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        # Remove the final fully connected layer
        self.image_feature_extractor = nn.Sequential(*list(self.image_model.children())[:-1])  # (B*M, 512, 1, 1)

        # Waveform Branch: 1D CNN
        self.waveform_model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # (B*M, 64, 1)
        )

        # Fusion Layer
        self.fusion = nn.Linear(512 + 64, embed_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, image, waveform):
        """
        Forward pass for the segment encoder.

        Parameters:
        - image: Tensor of shape (B*M, 4, H, W)
        - waveform: Tensor of shape (B*M, 600)

        Returns:
        - embedding: Tensor of shape (B*M, embed_dim)
        """
        # Image Features
        img_features = self.image_feature_extractor(image)  # (B*M, 512, 1, 1)
        img_features = img_features.view(img_features.size(0), -1)  # (B*M, 512)

        # Waveform Features
        waveform = waveform.unsqueeze(1)  # (B*M, 1, 600)
        wf_features = self.waveform_model(waveform)  # (B*M, 64, 1)
        wf_features = wf_features.view(wf_features.size(0), -1)  # (B*M, 64)

        # Fusion
        fused = torch.cat((img_features, wf_features), dim=1)  # (B*M, 512 + 64 = 576)
        fused = self.fusion(fused)  # (B*M, embed_dim)
        fused = self.relu(fused)
        fused = self.dropout(fused)

        return fused


# ===============================
# 6. Defining the Combined Loss Function
# ===============================

class CombinedLossWeighted(nn.Module):
    def __init__(self, class_weights=None, classification_weight=1.0, reconstruction_weight=1.0):
        """
        Initializes the combined loss for multi-task learning.

        Parameters:
        - class_weights: Tensor of class weights for classification loss.
        - classification_weight: Weight for the classification loss component.
        - reconstruction_weight: Weight for the reconstruction loss component.
        """
        super(CombinedLossWeighted, self).__init__()
        self.classification_weight = classification_weight
        self.reconstruction_weight = reconstruction_weight

        self.classification_loss = nn.CrossEntropyLoss(
            weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
        self.reconstruction_loss = nn.MSELoss()

    def forward(self, class_output, labels, recon_waveform, original_waveform):
        """
        Computes the combined loss.

        Parameters:
        - class_output: Tensor of shape (B, num_classes)
        - labels: Tensor of shape (B,)
        - recon_waveform: Tensor of shape (B, 600)
        - original_waveform: Tensor of shape (B, 600)

        Returns:
        - loss: Combined loss scalar
        - loss_cls: Classification loss scalar
        - loss_recon: Reconstruction loss scalar
        """
        loss_cls = self.classification_loss(class_output, labels)
        loss_recon = self.reconstruction_loss(recon_waveform, original_waveform)
        loss = self.classification_weight * loss_cls + self.reconstruction_weight * loss_recon
        return loss, loss_cls, loss_recon


# ===============================
# 7. Creating Dummy Data (For Illustration)
# ===============================

def create_dummy_data(num_signals=100, max_segments=10, num_classes=5, image_size=(4, 100, 100), waveform_length=600):
    """
    Creates dummy grouped ECG data for illustration purposes.

    Parameters:
    - num_signals: Number of ECG signals.
    - max_segments: Maximum number of segments per signal.
    - num_classes: Number of arrhythmia classes.
    - image_size: Tuple representing the image dimensions (C, H, W).
    - waveform_length: Length of the ECG waveform.

    Returns:
    - grouped_segments: List of lists containing segments for each ECG signal.
    - grouped_labels: List of integer labels for each ECG signal.
    """
    grouped_segments = []
    grouped_labels = []

    for _ in range(num_signals):
        num_segments = random.randint(5, max_segments)  # Variable number of segments per signal
        segments = []
        for _ in range(num_segments):
            # Create dummy images: 4 channels, H=100, W=100
            images = torch.randn(image_size)  # (4, 100, 100)

            # Create dummy waveform: 600 data points
            waveform = torch.randn(waveform_length)  # (600,)

            segment = {
                'images': images,
                'waveform': waveform
            }
            segments.append(segment)

        grouped_segments.append(segments)

        # Assign a random label
        label = random.randint(0, num_classes - 1)
        grouped_labels.append(label)

    return grouped_segments, grouped_labels


# ===============================
# 8. Main Function: Putting It All Together
# ===============================

def main():
    # ===============================
    # 8.1. Parameters
    # ===============================
    num_classes = 5
    waveform_length = 300
    image_size = (4, 300, 300)
    batch_size = 4
    num_epochs = 20
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_accuracy = 0.0
    save_path = 'best_hierarchical_ecg_model.pth'

    # ===============================
    # 8.2. Create Dummy Data
    # ===============================
    # train_grouped_segments, train_grouped_labels = create_dummy_data(num_signals=200, max_segments=10, num_classes=num_classes, image_size=image_size, waveform_length=waveform_length)
    # val_grouped_segments, val_grouped_labels = create_dummy_data(num_signals=50, max_segments=10, num_classes=num_classes, image_size=image_size, waveform_length=waveform_length)

    OFFSET = None
    DEBUG = False
    dtype = 1
    CLUSTERING = True
    PLOT = False

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
    # 100
    aligned_signals, aligned_labels = align_ecg_signals(data, labels, sample_rate=100, window_size=100)
    logging.info(f"Aligned signals: {len(aligned_signals)}")

    # 4.3. 신호 정규화
    normalized_signals = normalize_signals(aligned_signals)
    logging.info(f"Normalized signals: {len(normalized_signals)}")

    image_dataset, source_mapping, labels_per_segment = transform_to_images_grouped_with_labels(
        grouped_segments=normalized_signals,
        grouped_labels=aligned_labels
    )

    # Step 3: Split Dataset into Training and Validation
    train_grouped_images, val_grouped_images, train_grouped_segments, val_grouped_segments, train_grouped_labels, val_grouped_labels = split_dataset_for_deep_learning(
        image_dataset=image_dataset,
        source_mapping=source_mapping,
        labels_per_segment=labels_per_segment,
        grouped_segments=normalized_signals,
        grouped_labels=aligned_labels,
        train_size=0.8,
        random_state=42,
        return_grouped=True
    )

    # ===============================
    # 8.3. Define Transforms and Normalization
    # ===============================
    image_transforms = transforms.Compose([
        transforms.RandomRotation(degrees=10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop(size=(100, 100), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    ])

    image_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406, 0.406],  # Example values; adjust as needed
        std=[0.229, 0.224, 0.225, 0.225]
    )

    # ===============================
    # 8.4. Create Dataset Instances
    # ===============================
    train_dataset = GroupedECGDataset(
        grouped_segments=train_grouped_images,
        grouped_waveforms=train_grouped_segments,
        grouped_labels=train_grouped_labels,
        transform=image_transforms,
        image_normalize=image_normalize
    )

    val_dataset = GroupedECGDataset(
        grouped_segments=val_grouped_images,
        grouped_waveforms=val_grouped_segments,
        grouped_labels=val_grouped_labels,
        transform=None,  # No augmentation for validation
        image_normalize=image_normalize
    )

    # ===============================
    # 8.5. Create DataLoaders
    # ===============================
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=grouped_collate_fn,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=grouped_collate_fn,
        num_workers=0,
        pin_memory=True
    )

    # ===============================
    # 8.6. Initialize the Model
    # ===============================
    model = HierarchicalECGClassifier(num_classes=num_classes, waveform_length=waveform_length, image_size=image_size)
    model = model.to(device)

    # ===============================
    # 8.7. Compute Class Weights (Optional)
    # ===============================
    cw = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(train_grouped_labels),
        y=train_grouped_labels
    )
    class_weights = torch.tensor(cw, dtype=torch.float).to(device)

    # ===============================
    # 8.8. Define the Loss Function and Optimizer
    # ===============================
    criterion = CombinedLossWeighted(
        class_weights=class_weights,
        classification_weight=1.0,
        reconstruction_weight=0.2
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ===============================
    # 8.9. Training Loop
    # ===============================
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_cls_loss = 0.0
        running_recon_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            images, waveforms, labels, segment_mask = batch
            images = images.to(device).float()
            waveforms = waveforms.to(device).float()
            labels = labels.to(device).long()
            segment_mask = segment_mask.to(device).float()

            optimizer.zero_grad()

            class_output, recon_waveform = model(images, waveforms, segment_mask)

            # For reconstruction loss, using the mean waveform across segments as target
            # Alternatively, use the first segment's waveform or another strategy
            target_waveform = waveforms[:, 0, :]  # (B, 600)

            loss, loss_cls, loss_recon = criterion(class_output, labels, recon_waveform, target_waveform)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_cls_loss += loss_cls.item() * images.size(0)
            running_recon_loss += loss_recon.item() * images.size(0)

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_cls_loss = running_cls_loss / len(train_loader.dataset)
        epoch_recon_loss = running_recon_loss / len(train_loader.dataset)

        # ===============================
        # 8.10. Validation Phase
        # ===============================
        model.eval()
        val_loss = 0.0
        val_cls_loss = 0.0
        val_recon_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                images, waveforms, labels, segment_mask = batch
                images = images.to(device).float()
                waveforms = waveforms.to(device).float()
                labels = labels.to(device).long()
                segment_mask = segment_mask.to(device).float()

                class_output, recon_waveform = model(images, waveforms, segment_mask)

                target_waveform = waveforms[:, 0, :]  # (B, 600)

                loss, loss_cls, loss_recon = criterion(class_output, labels, recon_waveform, target_waveform)

                val_loss += loss.item() * images.size(0)
                val_cls_loss += loss_cls.item() * images.size(0)
                val_recon_loss += loss_recon.item() * images.size(0)

                # Calculate accuracy
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

        # ===============================
        # 8.11. Save the Best Model
        # ===============================
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), save_path)
            print(f"Saved Best Model at Epoch {epoch + 1} with Val Acc: {best_accuracy * 100:.2f}%")

    print("Training Completed.")

    # ===============================
    # 8.12. Optional: Loading the Best Model for Evaluation
    # ===============================
    # model.load_state_dict(torch.load(save_path))
    # model.eval()

    # ===============================
    # 8.13. Optional: Visualization Functions
    # ===============================

    def visualize_segment_importance(model, images, waveforms, segment_mask, device):
        """
        Visualizes the importance of each segment in an ECG signal based on embedding norms.

        Parameters:
        - model: Trained HierarchicalECGClassifier model.
        - images: Tensor of shape (1, max_num_segments, 4, H, W)
        - waveforms: Tensor of shape (1, max_num_segments, 600)
        - segment_mask: Tensor of shape (1, max_num_segments)
        - device: Torch device
        """
        model.eval()
        with torch.no_grad():
            images = images.to(device).float()
            waveforms = waveforms.to(device).float()
            segment_mask = segment_mask.to(device).float()

            # Forward pass
            class_output, recon_waveform = model(images, waveforms, segment_mask)

            # Extract segment embeddings
            batch_size, max_num_segments, C, H, W = images.size()
            images_reshaped = images.view(batch_size * max_num_segments, C, H, W)  # (B*M, 4, H, W)
            waveforms_reshaped = waveforms.view(batch_size * max_num_segments, -1)  # (B*M, 600)
            segment_embeddings = model.segment_encoder(images_reshaped, waveforms_reshaped)  # (B*M, 256)
            segment_embeddings = segment_embeddings.view(batch_size, max_num_segments, -1)  # (B, M, 256)

            # Permute for Transformer: (B, M, 256) -> (M, B, 256)
            transformer_input = segment_embeddings.permute(1, 0, 2)  # (M, B, 256)

            # Apply Transformer Encoder
            transformer_output = model.transformer_encoder(transformer_input)  # (M, B, 256)

            # Reshape back to (B, M, 256)
            transformer_output = transformer_output.permute(1, 0, 2)  # (B, M, 256)

            # Apply mask: zero out padded segments
            transformer_output = transformer_output * segment_mask.unsqueeze(-1)  # (B, M, 256)

            # Calculate embedding norms
            embedding_norms = torch.norm(transformer_output, dim=2)  # (B, M)
            embedding_norms = embedding_norms.cpu().numpy()[0]  # Assuming batch_size=1
            segment_mask = segment_mask.cpu().numpy()[0]

            # Mask the importance
            embedding_norms = embedding_norms * segment_mask

            # Normalize for visualization
            embedding_norms = (embedding_norms - embedding_norms.min()) / (
                        embedding_norms.max() - embedding_norms.min() + 1e-8)

            # Plot
            plt.figure(figsize=(10, 4))
            plt.bar(range(len(embedding_norms)), embedding_norms)
            plt.xlabel("Segment Index")
            plt.ylabel("Embedding Norm")
            plt.title("Segment Importance Based on Embedding Norms")
            plt.show()

    # ===============================
    # 8.14. Optional: Running Visualization on a Sample
    # ===============================
    # Select a random sample from the validation set
    sample_idx = random.randint(0, len(val_dataset) - 1)
    sample_images, sample_waveforms, sample_label = val_dataset[sample_idx]
    sample_mask = torch.ones(sample_images.size(0))  # Assuming no padding for the sample

    # If the sample has padding, adjust the mask accordingly
    # For this dummy data, all samples have max_num_segments=10

    # Expand dimensions to create a batch of size 1
    sample_images = sample_images.unsqueeze(0)  # (1, M, 4, H, W)
    sample_waveforms = sample_waveforms.unsqueeze(0)  # (1, M, 600)
    sample_mask = sample_mask.unsqueeze(0)  # (1, M)

    visualize_segment_importance(model, sample_images, sample_waveforms, sample_mask, device)


if __name__ == "__main__":
    main()
