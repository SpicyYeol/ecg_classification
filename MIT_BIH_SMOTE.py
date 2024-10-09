# 1D SMOTE file

import os
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

# 1. 각 폴더에서 데이터를 로드하는 함수
def load_1d_signals(file_path, folder_name):
    all_signals = []
    all_labels = []
    label_counts = {}

    for label, folder in enumerate(folder_name):
        folder_path = os.path.join(file_path, folder)
        count = 0
        for file in os.listdir(folder_path):
            if file.endswith('.csv'):
                file_full_path = os.path.join(folder_path, file)
                signal = np.loadtxt(file_full_path, delimiter=",")  # CSV 파일에서 1D 신호 로드
                all_signals.append(signal)
                all_labels.append(label)
                count += 1
        label_counts[folder] = count
    return np.array(all_signals), np.array(all_labels), label_counts

# 2. SMOTE 적용 후 데이터를 각 라벨별로 폴더에 저장하는 함수
def apply_smote_and_save_by_label(signals, labels, folder_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # 각 라벨별로 서브폴더 생성
    for folder in folder_name:
        label_folder = os.path.join(save_dir, folder)
        os.makedirs(label_folder, exist_ok=True)

    # SMOTE 적용 (자동으로 증강)
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    signals_resampled, labels_resampled = smote.fit_resample(signals, labels)

    # 라벨별로 데이터를 저장할 딕셔너리
    label_data = {label: [] for label in set(labels_resampled)}

    # 라벨별로 데이터 모으기
    for signal, label in zip(signals_resampled, labels_resampled):
        label_data[label].append(signal)

    # 라벨별로 데이터 저장
    for label, data in label_data.items():
        label_folder = os.path.join(save_dir, folder_name[label])  # 라벨에 맞는 서브폴더 경로
        for i, signal in enumerate(data):
            save_path = os.path.join(label_folder, f"smote_{folder_name[label]}_{i}.csv")
            # CSV 파일로 저장
            np.savetxt(save_path, signal, delimiter=",")
        print(f"Saved {len(data)} signals in {label_folder}.")

# 3. 실행 부분
file_path = "./MIT/"
folder_name = ["N", "S", "Q", "V"]  # 각 폴더 이름 (라벨 이름과 동일)
save_dir = "./smote_augmented_data_by_label"

# 1차원 신호 데이터 로드 및 각 클래스의 데이터 개수 파악
signals, labels, label_counts = load_1d_signals(file_path, folder_name)

# SMOTE 적용 및 라벨별로 데이터를 폴더에 저장 (자동으로 증강)
apply_smote_and_save_by_label(signals, labels, folder_name, save_dir)
