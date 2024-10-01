#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
혼동 행렬 및 분류 보고서 시각화 및 저장 기능이 추가된 ECG 분류 코드 (Transformer 입력 크기 수정 및 WeightNorm 적용)
"""

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
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)  # 모든 GPU
    np.random.seed(seed)  # Numpy
    random.seed(seed)  # Python random
    torch.backends.cudnn.deterministic = True  # 결정론적 결과
    torch.backends.cudnn.benchmark = False  # 재현성을 위해 benchmark 비활성화


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

    from tqdm import tqdm  # tqdm 임포트
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
            signal_tensor = torch.tensor(np.load(transformed_path), dtype=torch.float32)

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


# 모델 정의
import torch.nn.utils.weight_norm as weight_norm


class TemporalBlock2D(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super(TemporalBlock2D, self).__init__()
        # "same" 패딩 계산
        pad_height = (kernel_size[0] - 1) * dilation[0] // 2
        pad_width = (kernel_size[1] - 1) * dilation[1] // 2
        padding = (pad_width, pad_width, pad_height, pad_height)  # (좌, 우, 상, 하)

        # 첫 번째 합성곱 층에 WeightNorm 적용
        self.conv1 = weight_norm(nn.Conv2d(n_inputs, n_outputs, kernel_size, stride=stride,
                                           padding=0, dilation=dilation))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # 두 번째 합성곱 층에 WeightNorm 적용
        self.conv2 = weight_norm(nn.Conv2d(n_outputs, n_outputs, kernel_size, stride=stride,
                                           padding=0, dilation=dilation))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 잔차 연결에 WeightNorm 적용
        self.downsample = None
        if n_inputs != n_outputs:
            self.downsample = weight_norm(nn.Conv2d(n_inputs, n_outputs, kernel_size=1))
        self.relu = nn.ReLU()

        # 패딩 레이어
        self.pad = nn.ZeroPad2d(padding)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.pad(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # 잔차 연결
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet2D(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=(3, 3), dropout=0.2,
                 num_classes=3, use_transformer=False):
        super(TemporalConvNet2D, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = (2 ** i, 2 ** i)
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock2D(in_channels, out_channels, kernel_size,
                                       stride=1, dilation=dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

        # Adaptive Pooling 레이어 추가
        self.adaptive_pool = nn.AdaptiveAvgPool2d((16, 16))  # 원하는 크기로 설정

        if use_transformer:
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=out_channels, nhead=4, batch_first=True),
                num_layers=2
            )
        else:
            self.transformer = None

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        x = self.network(x)
        x = self.adaptive_pool(x)  # Adaptive Pooling 적용으로 크기 감소

        if self.transformer is not None:
            b, c, h, w = x.size()
            x = x.view(b, c, h * w).permute(0, 2, 1)  # (batch_size, seq_len, embedding_dim)
            x = self.transformer(x)
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
    log_file = os.path.join('logs', f"{experiment_name}.log")
    os.makedirs('logs', exist_ok=True)
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
    use_transformer = config['use_transformer']
    batch_size = config['batch_size']
    accumulation_steps = config['accumulation_steps']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    epochs = config['epochs']
    patience = config['patience']
    root_path = config['root_path']
    preprocessed_dir = config['preprocessed_dir']

    # 데이터 로드
    file_dict = {"N": 0, "S": 1, "V": 2}
    file_paths = []
    labels = []

    for folder_name, label in file_dict.items():
        folder_path = os.path.join(root_path, folder_name)
        if not os.path.isdir(folder_path):
            logging.warning(f"폴더 {folder_path} 가 존재하지 않습니다.")
            continue
        for file in os.listdir(folder_path)[:10000]:
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
    model = TemporalConvNet2D(num_inputs=input_size, num_channels=num_channels, kernel_size=kernel_size,
                              dropout=dropout, num_classes=output_size, use_transformer=use_transformer)
    if torch.cuda.device_count() > 1:
        logging.info(f"{torch.cuda.device_count()}개의 GPU를 사용합니다.")
        model = nn.DataParallel(model)
    model.to(device)

    # 혼합 정밀도 훈련을 위한 설정
    scaler = torch.amp.GradScaler()

    # 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 훈련 파라미터
    best_val_loss = float('inf')
    save_path = os.path.join('models', f"{experiment_name}_best.pth")
    os.makedirs('models', exist_ok=True)
    trigger_times = 0

    # 결과 기록을 위한 딕셔너리
    results = {
        'experiment_name': experiment_name,
        'num_channels': num_channels,
        'use_transformer': use_transformer,
        'dropout': dropout,
        'learning_rate': learning_rate,
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

        results['train_loss'].append(train_loss / train_total)
        results['train_accuracy'].append(train_accuracy)

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

        results['valid_loss'].append(valid_loss / valid_total)
        results['valid_accuracy'].append(valid_accuracy)

        scheduler.step()

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

    test_accuracy = 100 * test_correct / test_total
    logging.info(f"Test Loss: {test_loss / test_total:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    results['test_accuracy'] = test_accuracy

    # **혼동 행렬 및 분류 보고서 시각화 및 저장 추가**
    # 혼동 행렬 계산
    cm = confusion_matrix(all_labels, all_preds)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(file_dict.keys()))

    # 혼동 행렬 그림 저장
    os.makedirs('results', exist_ok=True)
    cm_figure_file = os.path.join('results', f"{experiment_name}_confusion_matrix.png")
    cm_display.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix for {experiment_name}")
    plt.savefig(cm_figure_file)
    plt.close()
    logging.info(f"혼동 행렬이 {cm_figure_file} 에 저장되었습니다.")

    # 분류 보고서 계산
    report = classification_report(all_labels, all_preds, target_names=list(file_dict.keys()), output_dict=True)
    report_file = os.path.join('results', f"{experiment_name}_classification_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=4)
    logging.info(f"분류 보고서가 {report_file} 에 저장되었습니다.")

    # 결과 딕셔너리에 파일 경로 추가
    results['confusion_matrix_file'] = cm_figure_file
    results['classification_report_file'] = report_file

    # 결과 저장
    results_file = os.path.join('results', 'experiment_results.csv')
    os.makedirs('results', exist_ok=True)

    # 결과를 CSV 파일에 저장
    save_results(results, results_file)


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
        [16, 32],
        [32, 64, 128],
        [64, 128, 256],
        [32, 64, 128, 256]
    ]
    use_transformer_options = [True, False]
    dropout_options = [0.1, 0.2, 0.3]
    learning_rate_options = [0.001, 0.0005, 0.0001]

    # 다른 하이퍼파라미터는 고정값 또는 필요에 따라 추가
    base_config = {
        'experiment_name': 'experiment',
        'input_size': 2,
        'output_size': 3,
        'kernel_size': (3, 3),
        'fmin': 0,
        'fmax': 15,
        'signal_length': 10,
        'batch_size': 16,  # 배치 크기 감소
        'accumulation_steps': 4,
        'weight_decay': 1e-5,
        'epochs': 50,
        'patience': 5,
        'root_path': r'F:\homes\icentia_pre',
        'preprocessed_dir': r'F:\homes\icentia_npy',
        'seed': 42  # 고정된 시드 값 설정
    }

    # 하이퍼파라미터 조합 생성
    configs = []
    for num_channels, use_transformer, dropout, learning_rate in itertools.product(
            num_channels_options, use_transformer_options,
            dropout_options, learning_rate_options
    ):
        config = base_config.copy()
        config['num_channels'] = num_channels
        config['use_transformer'] = use_transformer
        config['dropout'] = dropout
        config['learning_rate'] = learning_rate
        config['experiment_name'] = f"exp_nc{len(num_channels)}_tf{use_transformer}_do{dropout}_lr{learning_rate}"
        configs.append(config)

    return configs


# 메인 실행 부분
if __name__ == "__main__":
    configs = generate_experiment_configs()
    for config in configs:
        run_experiment(config)
