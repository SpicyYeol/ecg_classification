#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
혼동 행렬 및 분류 보고서 시각화 및 저장 기능이 추가된 ECG 분류 코드 (Efficient Transformer, DropBlock, RAdam 및 Ranger 도입)
"""

# Required packages:
# pip install linformer dropblock ranger-pytorch torch_optimizer tqdm

import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from stockwell import st
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import logging
import itertools
import json  # 분류 보고서 저장을 위한 임포트

# Additional imports for Efficient Transformer, DropBlock, and Optimizers
from linformer import Linformer  # Efficient Transformer
from dropblock import DropBlock2D  # DropBlock
from ranger import Ranger  # Ranger optimizer
from torch_optimizer import RAdam  # RAdam optimizer

from tqdm import tqdm  # tqdm 임포트

# 로깅 설정
def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:%(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

# 재현성을 위한 시드 설정
def set_seed(seed):
    torch.manual_seed(seed)                   # CPU
    torch.cuda.manual_seed(seed)              # GPU
    torch.cuda.manual_seed_all(seed)          # 모든 GPU
    np.random.seed(seed)                      # Numpy
    random.seed(seed)                         # Python random
    torch.backends.cudnn.deterministic = True # 결정론적 결과
    torch.backends.cudnn.benchmark = False    # 재현성을 위해 benchmark 비활성화

# Stockwell 변환 함수
def stockwell_transform(signal, fmin, fmax, signal_length):
    df = 1. / signal_length
    fmin_samples = int(fmin / df)
    fmax_samples = int(fmax / df)
    trans_signal = st.st(signal, fmin_samples, fmax_samples)
    return trans_signal

# 파일을 전처리하고 저장하는 함수 (병렬 처리 버전)
def preprocess_file(args):
    file_path, save_dir, fmin, fmax, signal_length = args
    try:
        # 신호 읽기
        signal = pd.read_csv(file_path, header=None).squeeze().to_numpy()

        # Stockwell 변환 적용
        trans_signal = stockwell_transform(signal, fmin, fmax, signal_length)

        # 실수부와 허수부 분리
        real_part = np.real(trans_signal)
        imag_part = np.imag(trans_signal)

        # 스택하고 저장
        transformed_signal = np.stack((real_part, imag_part), axis=0)
        filename = os.path.basename(file_path).replace('.csv', '_transformed.npy')
        save_path = os.path.join(save_dir, filename)
        np.save(save_path, transformed_signal)
    except Exception as e:
        logging.error(f"{file_path} 처리 중 오류 발생: {e}")

def preprocess_and_save(file_paths, save_dir, fmin, fmax, signal_length):
    os.makedirs(save_dir, exist_ok=True)
    args_list = [(file_path, save_dir, fmin, fmax, signal_length) for file_path in file_paths]
    num_processes = max(1, cpu_count() - 1)  # 한 개의 CPU 코어는 남겨둠

    with Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap_unordered(preprocess_file, args_list), total=len(args_list)))

# 커스텀 데이터셋 클래스
class CustomECGDataset(Dataset):
    def __init__(self, file_paths, labels, transform_dir):
        self.file_paths = file_paths
        self.labels = labels
        self.transform_dir = transform_dir

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        try:
            # 전처리된 데이터 로드
            original_file = os.path.basename(self.file_paths[idx]).replace('.csv', '_transformed.npy')
            transformed_path = os.path.join(self.transform_dir, original_file)
            signal = np.load(transformed_path)
            signal_tensor = torch.tensor(signal, dtype=torch.float32)

            # 레이블
            label = self.labels[idx]
            return signal_tensor, label
        except Exception as e:
            logging.error(f"인덱스 {idx}에서 데이터 로드 중 오류 발생: {e}")
            # 오류 발생 시 0 텐서와 기본 레이블 반환
            signal_tensor = torch.zeros((2, 151, 1000), dtype=torch.float32)  # 필요한 경우 차원 조정
            label = 0
            return signal_tensor, label

# 전처리가 필요한지 확인하는 함수
def check_preprocessed_data(file_paths, preprocessed_dir):
    preprocessed_files = os.listdir(preprocessed_dir) if os.path.exists(preprocessed_dir) else []
    expected_files = [os.path.basename(fp).replace('.csv', '_transformed.npy') for fp in file_paths]
    missing_files = set(expected_files) - set(preprocessed_files)
    return len(missing_files) == 0

# Efficient Transformer (Linformer) 클래스
class EfficientTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1, max_seq_len=256, k=256):
        super(EfficientTransformer, self).__init__()
        self.linformer = Linformer(
            dim=d_model,
            seq_len=max_seq_len,
            depth=num_layers,
            heads=nhead,
            k=k,
            one_kv_head=True,
            share_kv=True,
            reversible=False,
            # layer_norm_epsilon=1e-5,
            dropout=dropout,
            # attention_dropout=dropout
        )

    def forward(self, x):
        return self.linformer(x)

# DropBlock을 포함한 TemporalBlock2D
class TemporalBlock2D(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2, use_batchnorm=False, use_dropblock=False, dropblock_size=7):
        super(TemporalBlock2D, self).__init__()
        # "same" 패딩 계산
        pad_height = (kernel_size[0] - 1) * dilation[0] // 2
        pad_width = (kernel_size[1] - 1) * dilation[1] // 2
        padding = (pad_width, pad_width, pad_height, pad_height)  # (좌, 우, 상, 하)

        # 첫 번째 합성곱 층
        self.conv1 = nn.Conv2d(n_inputs, n_outputs, kernel_size, stride=stride,
                               padding=0, dilation=dilation)
        self.batchnorm1 = nn.BatchNorm2d(n_outputs) if use_batchnorm else nn.Identity()
        self.relu1 = nn.LeakyReLU(0.1, inplace=True)  # LeakyReLU 사용
        self.dropout1 = nn.Dropout(dropout)
        self.dropblock1 = DropBlock2D(block_size=dropblock_size, drop_prob=dropout) if use_dropblock else nn.Identity()

        # 두 번째 합성곱 층
        self.conv2 = nn.Conv2d(n_outputs, n_outputs, kernel_size, stride=stride,
                               padding=0, dilation=dilation)
        self.batchnorm2 = nn.BatchNorm2d(n_outputs) if use_batchnorm else nn.Identity()
        self.relu2 = nn.LeakyReLU(0.1, inplace=True)  # LeakyReLU 사용
        self.dropout2 = nn.Dropout(dropout)
        self.dropblock2 = DropBlock2D(block_size=dropblock_size, drop_prob=dropout) if use_dropblock else nn.Identity()

        # 잔차 연결
        self.downsample = nn.Conv2d(n_inputs, n_outputs, kernel_size=1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

        # 패딩 레이어
        self.pad = nn.ZeroPad2d(padding)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv1(out)
        out = self.batchnorm1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.dropblock1(out)

        out = self.pad(out)
        out = self.conv2(out)
        out = self.batchnorm2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.dropblock2(out)

        # 잔차 연결
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

# 모델 정의
class TemporalConvNet2D(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=(3, 3), dropout=0.2,
                 num_classes=3, use_batchnorm=False, use_transformer=False,
                 transformer_layers=4, transformer_heads=8, max_seq_len=256, use_dropblock=False, dropblock_size=7):
        super(TemporalConvNet2D, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = (2 ** i, 2 ** i)
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock2D(in_channels, out_channels, kernel_size,
                                       stride=1, dilation=dilation_size, dropout=dropout,
                                       use_batchnorm=use_batchnorm, use_dropblock=use_dropblock, dropblock_size=dropblock_size)]

        self.network = nn.Sequential(*layers)

        # Adaptive Pooling 레이어 추가
        self.adaptive_pool = nn.AdaptiveAvgPool2d((16, 16))  # 원하는 크기로 설정

        if use_transformer:
            # Efficient Transformer 사용
            self.transformer = EfficientTransformer(
                d_model=num_channels[-1],
                nhead=transformer_heads,
                num_layers=transformer_layers,
                dim_feedforward=2048,
                dropout=dropout,
                max_seq_len=256,
                k=256  # Linformer specific parameter
            )
        else:
            self.transformer = None

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_channels[-1], num_classes)  # Use num_channels[-1] as in_channels

    def forward(self, x):
        x = self.network(x)
        x = self.adaptive_pool(x)  # Adaptive Pooling 적용으로 크기 감소

        if self.transformer is not None:
            b, c, h, w = x.size()
            x = x.view(b, c, h * w).permute(0, 2, 1)  # (batch_size, seq_len, embedding_dim)
            x = self.transformer(x)  # Efficient Transformer
            x = x.mean(dim=1)  # 시퀀스 차원에 대해 평균
        else:
            x = self.global_avg_pool(x)
            x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x

# 실험 실행 함수
def run_experiment(config):
    # 로그 파일 설정
    experiment_name = config['experiment_name']
    experiment_dir = os.path.join('results', experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    log_file = os.path.join('logs', f"{experiment_name}.log")
    setup_logging(log_file)

    # 시드 설정
    seed = config.get('seed', 42)  # 고정된 시드 값 사용
    set_seed(seed)

    # 파라미터 설정
    input_size = config['input_size']
    output_size = config['output_size']
    num_channels = config['num_channels']
    kernel_size = config['kernel_size']
    dropout = config['dropout']
    fmin = config['fmin']
    fmax = config['fmax']
    signal_length = config['signal_length']
    target_length = config['target_length']
    use_batchnorm = config['use_batchnorm']
    use_transformer = config['use_transformer']
    transformer_layers = config.get('transformer_layers', 4)
    transformer_heads = config.get('transformer_heads', 8)
    batch_size = config['batch_size']
    accumulation_steps = config['accumulation_steps']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    epochs = config['epochs']
    patience = config['patience']
    root_path = config['root_path']
    preprocessed_dir = config['preprocessed_dir']
    augment = config.get('augment', False)  # 데이터 증강 여부
    use_dropblock = config.get('use_dropblock', True)  # DropBlock 사용 여부
    dropblock_size = config.get('dropblock_size', 7)  # DropBlock 크기
    optimizer_type = config.get('optimizer', 'Ranger')  # 옵티마이저 선택: 'RAdam' 또는 'Ranger'

    # 데이터 로드
    file_dict = {"N": 0, "S": 1, "V": 2}
    file_paths = []
    labels = []

    for folder_name, label in file_dict.items():
        folder_path = os.path.join(root_path, folder_name)
        if not os.path.isdir(folder_path):
            logging.warning(f"폴더 {folder_path} 가 존재하지 않습니다.")
            continue
        for file in os.listdir(folder_path)[:15000]:  # Changed to 1000 as per user's last code
            if file.endswith('.csv'):
                file_paths.append(os.path.join(folder_path, file))
                labels.append(label)

    if not file_paths:
        logging.error("데이터 파일을 찾을 수 없습니다. 데이터 디렉토리를 확인해주세요.")
        return

    # 전처리 및 변환된 데이터 저장
    # 전처리가 필요한지 확인
    if not check_preprocessed_data(file_paths, preprocessed_dir):
        logging.info("데이터를 전처리하고 저장합니다...")
        preprocess_and_save(file_paths, preprocessed_dir, fmin, fmax, signal_length)
    else:
        logging.info("전처리된 데이터가 이미 존재합니다. 전처리 단계를 건너뜁니다.")

    # 데이터 분할
    logging.info("데이터를 분할합니다...")
    train_data, valid_test_data, train_labels, valid_test_labels = train_test_split(
        file_paths, labels, test_size=0.2, random_state=42, stratify=labels, shuffle=True
    )

    valid_data, test_data, valid_labels, test_labels = train_test_split(
        valid_test_data, valid_test_labels, test_size=0.5, random_state=42, stratify=valid_test_labels
    )

    # 데이터셋 및 데이터로더 생성
    train_dataset = CustomECGDataset(train_data, train_labels, preprocessed_dir)
    valid_dataset = CustomECGDataset(valid_data, valid_labels, preprocessed_dir)
    test_dataset = CustomECGDataset(test_data, test_labels, preprocessed_dir)

    num_workers = min(8, cpu_count())  # 시스템에 따라 조정

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    # 디바이스 및 모델 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalConvNet2D(
        num_inputs=input_size,
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        num_classes=output_size,
        use_batchnorm=use_batchnorm,
        use_transformer=use_transformer,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        max_seq_len=256,
        use_dropblock=use_dropblock,
        dropblock_size=dropblock_size
    )
    if torch.cuda.device_count() > 1:
        logging.info(f"{torch.cuda.device_count()}개의 GPU를 사용합니다.")
        model = nn.DataParallel(model)
    model.to(device)

    # 혼합 정밀도 훈련을 위한 설정
    scaler = torch.cuda.amp.GradScaler()

    # 손실 함수
    criterion = nn.CrossEntropyLoss()

    # 옵티마이저 설정
    if optimizer_type == 'RAdam':
        optimizer = RAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        logging.info("RAdam 옵티마이저를 사용합니다.")
    elif optimizer_type == 'Ranger':
        optimizer = Ranger(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        logging.info("Ranger 옵티마이저를 사용합니다.")
    else:
        raise ValueError(f"지원하지 않는 옵티마이저 유형: {optimizer_type}")

    # 학습률 스케줄러: ReduceLROnPlateau 사용
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # 훈련 파라미터
    best_val_loss = float('inf')
    save_path = os.path.join('models', f"{experiment_name}_best.pth")
    os.makedirs('models', exist_ok=True)
    trigger_times = 0

    # 결과 기록을 위한 딕셔너리
    results = {
        'experiment_name': experiment_name,
        'num_channels': num_channels,
        'use_batchnorm': use_batchnorm,
        'use_transformer': use_transformer,
        'transformer_layers': transformer_layers,
        'transformer_heads': transformer_heads,
        'dropout': dropout,
        'use_dropblock': use_dropblock,
        'dropblock_size': dropblock_size,
        'optimizer': optimizer_type,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'seed': seed,  # 고정된 시드 값
        'train_loss': [],
        'train_accuracy': [],
        'valid_loss': [],
        'valid_accuracy': [],
        'test_accuracy': None,
        'confusion_matrix_file': None,
        'classification_report_file': None
    }

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
            labels_batch = labels_batch.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels_batch)
                loss = loss / accumulation_steps  # 그래디언트 누적을 위한 손실 스케일링

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

        results['train_loss'].append(train_loss / train_total)
        results['train_accuracy'].append(train_accuracy)

        # 검증 루프
        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0
        all_valid_preds = []
        all_valid_labels = []
        with torch.no_grad():
            for inputs, labels_batch in valid_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels_batch = labels_batch.to(device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels_batch)

                valid_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                valid_total += labels_batch.size(0)
                valid_correct += (preds == labels_batch).sum().item()
                all_valid_preds.extend(preds.cpu().numpy())
                all_valid_labels.extend(labels_batch.cpu().numpy())

        valid_accuracy = 100 * valid_correct / valid_total
        logging.info(f"Validation Loss: {valid_loss / valid_total:.4f}, Validation Accuracy: {valid_accuracy:.2f}%")

        results['valid_loss'].append(valid_loss / valid_total)
        results['valid_accuracy'].append(valid_accuracy)

        # 스케줄러 업데이트
        scheduler.step(valid_loss / valid_total)

        # 베스트 모델 저장
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
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

    # 테스트
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_test_preds = []
    all_test_labels = []
    with torch.no_grad():
        for inputs, labels_batch in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels_batch)

            test_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            test_total += labels_batch.size(0)
            test_correct += (preds == labels_batch).sum().item()

            all_test_preds.extend(preds.cpu().numpy())
            all_test_labels.extend(labels_batch.cpu().numpy())

    test_accuracy = 100 * test_correct / test_total
    logging.info(f"Test Loss: {test_loss / test_total:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    results['test_accuracy'] = test_accuracy

    # **혼동 행렬 및 분류 보고서 시각화 및 저장 추가**
    # 혼동 행렬 계산
    cm = confusion_matrix(all_test_labels, all_test_preds)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(file_dict.keys()))

    # 혼동 행렬 그림 저장
    cm_figure_file = os.path.join(experiment_dir, f"{experiment_name}_confusion_matrix.png")
    cm_display.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix for {experiment_name}")
    plt.savefig(cm_figure_file)
    plt.close()
    logging.info(f"혼동 행렬이 {cm_figure_file} 에 저장되었습니다.")

    # 분류 보고서 계산
    report = classification_report(all_test_labels, all_test_preds, target_names=list(file_dict.keys()), output_dict=True)
    report_file = os.path.join(experiment_dir, f"{experiment_name}_classification_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=4)
    logging.info(f"분류 보고서가 {report_file} 에 저장되었습니다.")

    # 결과 딕셔너리에 파일 경로 추가
    results['confusion_matrix_file'] = cm_figure_file
    results['classification_report_file'] = report_file

    # 결과 저장
    # 결과를 CSV 파일에 저장
    save_results(results, os.path.join(experiment_dir, 'experiment_results.csv'))

# 결과를 CSV 파일에 저장하는 함수
def save_results(results, results_file):
    # 결과를 데이터프레임으로 변환
    df = pd.DataFrame([results])
    if not os.path.isfile(results_file):
        df.to_csv(results_file, index=False)
    else:
        df_existing = pd.read_csv(results_file)
        df_combined = pd.concat([df_existing, df], ignore_index=True)
        df_combined.to_csv(results_file, index=False)

# 실험 설정 생성 함수
def generate_experiment_configs():
    # 하이퍼파라미터 그리드 정의
    num_channels_options = [
        [32, 64, 128],
        [64, 64, 128],
        # [16, 32],
        # [64, 128,  256],
        # [32, 64, 128, 256]
    ]
    use_batchnorm_options = [True]  # , False]
    use_transformer_options = [True]  # , False]
    dropout_options = [0.1]  # , 0.2, 0.3]
    learning_rate_options = [0.0005]  # [0.0001, 0.0005, 0.001]
    transformer_layers_options = [4, 6]  # 추가 실험: Transformer 레이어 수
    transformer_heads_options = [8, 16]  # 추가 실험: Transformer 헤드 수
    # seed_options 제거됨

    # DropBlock options
    use_dropblock_options = [True, False]
    dropblock_size_options = [7]  # [5, 7]

    # Optimizer options
    optimizer_options = ['RAdam', 'Ranger']  # 'RAdam' 또는 'Ranger'

    # 다른 하이퍼파라미터는 고정값 또는 필요에 따라 추가
    base_config = {
        'experiment_name': 'experiment',
        'input_size': 2,
        'output_size': 3,
        'kernel_size': (3, 3),
        'fmin': 0,
        'fmax': 15,
        'signal_length': 10,
        'target_length': 151,  # 고정된 신호 길이
        'batch_size': 16,       # 배치 크기 감소
        'accumulation_steps': 4,
        'weight_decay': 1e-5,
        'epochs': 50,
        'patience': 8,
        'root_path': r'F:\homes\icentia_pre',      # 실제 데이터 경로로 수정 필요
        'preprocessed_dir': r'F:\homes\icentia_npy', # 실제 저장 경로로 수정 필요
        'seed': 42,  # 고정된 시드 값 설정
        'augment': False,  # 데이터 증강 여부 (비활성화 as per earlier)
        'use_dropblock': True,  # DropBlock 사용 여부
        'dropblock_size': 7  # DropBlock 크기
    }

    # 하이퍼파라미터 조합 생성
    configs = []
    for num_channels, use_batchnorm, use_transformer, dropout, learning_rate, transformer_layers, transformer_heads, use_dropblock, dropblock_size, optimizer in itertools.product(
        num_channels_options, use_batchnorm_options, use_transformer_options,
        dropout_options, learning_rate_options, transformer_layers_options, transformer_heads_options,
        use_dropblock_options, dropblock_size_options, optimizer_options
    ):
        config = base_config.copy()
        config['num_channels'] = num_channels
        config['use_batchnorm'] = use_batchnorm
        config['use_transformer'] = use_transformer
        config['dropout'] = dropout
        config['learning_rate'] = learning_rate
        config['transformer_layers'] = transformer_layers
        config['transformer_heads'] = transformer_heads
        config['use_dropblock'] = use_dropblock
        config['dropblock_size'] = dropblock_size
        config['optimizer'] = optimizer
        # seed_options 제거로 인해 seed는 base_config에서 가져옵니다.
        config['experiment_name'] = f"exp_nc{len(num_channels)}_bn{use_batchnorm}_tf{use_transformer}_do{dropout}_lr{learning_rate}_tl{transformer_layers}_th{transformer_heads}_db{dropblock_size}_opt{optimizer}"
        configs.append(config)

    return configs

# 메인 실행 부분
if __name__ == "__main__":
    configs = generate_experiment_configs()
    for config in configs:
        run_experiment(config)
