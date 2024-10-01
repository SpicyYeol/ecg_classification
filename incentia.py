#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 13:40:56 2024

@author: ssl
"""

import wfdb
import os
import torch
import numpy as np
from scipy import signal  # 변경: scipy.signal 임포트
from multiprocessing import Pool, Manager, cpu_count
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# GPU 사용 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 데이터 및 저장 디렉토리 설정
DATA_BASE_DIR = r'F:\icentia11k-single-lead-continuous-raw-electrocardiogram-dataset-1.0\icentia11k-single-lead-continuous-raw-electrocardiogram-dataset-1.0'
SAVE_BASE_DIR = r'F:\homes\icentia_pre'

# 로그 디렉토리 설정 및 생성
LOG_DIR = os.path.join(SAVE_BASE_DIR, 'log')
os.makedirs(LOG_DIR, exist_ok=True)

# %% 함수 정의
def process_record(args):
    i, signal_dict, exit_event = args
    if exit_event.is_set():
        return True

    folder_name1 = ["00", "01", "02", "03", "04", "05","06","07"]
    folder_name2 = str(i).zfill(5)

    data_directory = os.path.join(
        DATA_BASE_DIR,
        'p' + folder_name1[i // 1000],
        'p' + folder_name2
    )

    if not os.path.exists(data_directory):
        return True  # 디렉토리가 없으면 작업 완료로 간주

    record_files = [f for f in os.listdir(data_directory) if f.endswith('.dat')]

    pid = os.getpid()
    log_file = os.path.join(LOG_DIR, f'log_{pid}.txt')

    for record_file in record_files:
        record_name = os.path.splitext(record_file)[0]
        write_log(log_file, f"[PID {pid}] Reading {record_name}...")

        try:
            record_path = os.path.join(data_directory, record_name)
            # 신호와 주석을 동시에 읽기
            record = wfdb.rdrecord(record_path)
            annotation = wfdb.rdann(record_path, 'atr')

            # 데이터를 텐서로 변환하고 GPU로 이동
            original_signal = torch.tensor(record.p_signal.flatten(), dtype=torch.float32, device=device)
            original_fs = record.fs
        except Exception as e:
            write_log(log_file, f"[PID {pid}] Error reading {record_name}: {e}")
            continue

        target_fs = 100
        segment_length = int(50 * original_fs)  # 50초 분량
        step_size = segment_length  # 중첩 없이 처리
        resampled_segment_length = int(segment_length * target_fs / original_fs)

        segment_indices = range(0, len(original_signal) - segment_length + 1, step_size)
        for Sig_num in segment_indices:
            if exit_event.is_set():
                return True

            segment_signal = original_signal[Sig_num:Sig_num + segment_length]
            segment_signal = low_pass_filter(segment_signal, cutoff=50, fs=original_fs)
            segment_signal = resample_signal(segment_signal, original_fs, target_fs)
            segment_signal = detrend(segment_signal, 100)

            window_indices = range(0, resampled_segment_length - 1000 + 1, 1000)
            for Sig_num2 in window_indices:
                window = segment_signal[Sig_num2:Sig_num2 + 1000]
                if torch.max(torch.abs(window)) > 10:
                    continue

                start_sample = Sig_num + int(Sig_num2 * original_fs / target_fs)
                end_sample = start_sample + int(1000 * original_fs / target_fs)

                # 주석 정보에서 해당 구간의 심볼 추출
                ann_samples = torch.tensor(annotation.sample, device=device)
                sym_indices = torch.where((ann_samples >= start_sample) & (ann_samples <= end_sample))[0]
                if sym_indices.numel() == 0:
                    continue

                sym_list = [annotation.symbol[idx] for idx in sym_indices.cpu().numpy()]
                sym = get_priority_symbol(sym_list)

                if sym == "":
                    continue

                # 공유 변수에 락 없이 접근 (Manager의 dict는 원자성을 보장함)
                signal_dict[sym] += 1
                count = signal_dict[sym]

                # if count > 10000:
                #     continue

                window = iqr_min_max_normalize(window)

                # if all(signal_dict[s] >= 10000 for s in signal_dict.keys()):
                #     exit_event.set()
                #     return True

                save_directory = os.path.join(SAVE_BASE_DIR, sym)
                os.makedirs(save_directory, exist_ok=True)

                file_name = f"{sym}_{count}.csv"
                file_path = os.path.join(save_directory, file_name)
                # GPU에서 CPU로 데이터 이동하여 저장
                window_cpu = window.cpu().numpy()
                np.savetxt(file_path, window_cpu, delimiter=",")

    return True

def get_priority_symbol(sym_list):
    # 심볼 우선 순위 정의
    for sym in ["Q", "V", "S", "N"]:
        if sym in sym_list:
            return sym
    return ""

def write_log(log_file, message):
    with open(log_file, "a") as log:
        log.write(message + "\n")
        log.flush()

# 신호 처리 함수들
def iqr_min_max_normalize(data):
    # IQR 정규화
    Q1 = torch.quantile(data, 0.25)
    Q3 = torch.quantile(data, 0.75)
    IQR = Q3 - Q1
    data = (data - Q1) / IQR

    # Min-Max 정규화
    min_val = torch.min(data)
    max_val = torch.max(data)
    if max_val - min_val == 0:
        data = data - min_val
    else:
        data = (data - min_val) / (max_val - min_val)
    return data

def detrend(signal, Lambda):
    signal_length = len(signal)
    H = torch.eye(signal_length, device=device)
    e = torch.ones(signal_length, device=device)
    D = torch.diag(e, 0) - torch.diag(e[:-1], -1)
    D = D[:-1, :]
    filtered_signal = torch.matmul(
        (H - torch.inverse(H + (Lambda ** 2) * torch.matmul(D.T, D))),
        signal
    )
    return filtered_signal

def resample_signal(signal, original_fs, target_fs):
    num_samples = int(len(signal) * target_fs / original_fs)
    resampled_signal = torch.nn.functional.interpolate(
        signal.unsqueeze(0).unsqueeze(0), size=num_samples, mode='linear', align_corners=False
    ).squeeze()
    return resampled_signal

def low_pass_filter(signal, cutoff, fs, numtaps=101):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    # 필터 계수를 scipy.signal.firwin으로 계산
    fir_coeff = signal_firwin(numtaps, normal_cutoff)
    # 필터 계수를 텐서로 변환하고 GPU로 이동
    fir_coeff = torch.tensor(fir_coeff, dtype=torch.float32, device=device)
    # Convolution으로 필터 적용
    filtered_signal = torch.nn.functional.conv1d(
        signal.unsqueeze(0).unsqueeze(0),
        fir_coeff.flip(0).unsqueeze(0).unsqueeze(0),
        padding=numtaps // 2
    ).squeeze()
    return filtered_signal

def signal_firwin(numtaps, cutoff):
    # scipy.signal.firwin 함수를 사용하여 필터 계수 계산
    return signal.firwin(numtaps, cutoff)

# %% 병렬 처리 시작
if __name__ == "__main__":
    with Manager() as manager:
        shared_signal_dict = manager.dict({"N": 0, "V": 0, "S": 0, "Q": 0})
        exit_event = manager.Event()

        record_indices = list(range(7875))

        num_processes = min(cpu_count(), 8)  # 시스템에 맞게 프로세스 수 조정
        print("num_processes:", num_processes)

        args_iterable = [(i, shared_signal_dict, exit_event) for i in record_indices]

        try:
            with Pool(processes=num_processes) as pool:
                for _ in tqdm(pool.imap_unordered(process_record, args_iterable), total=len(record_indices), smoothing=0.1):
                    if exit_event.is_set():
                        pool.terminate()
                        break
        except Exception as e:
            print(f"An error occurred during parallel processing: {e}")
