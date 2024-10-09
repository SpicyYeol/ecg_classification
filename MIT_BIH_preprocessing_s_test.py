# 데이터 전처리

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 16:39:48 2024

@author: ssl
"""

import wfdb
import os
import scipy.signal as signal
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import resample
import torch
import numpy as np
from scipy.sparse import spdiags
import neurokit2 as nk
from multiprocessing import Pool, Manager, Event
import fcntl
# from scipy.signal import butter, filtfilt
from scipy.signal import firwin, lfilter
from collections import Counter
import pickle
import random
import sys

from scipy.interpolate import interp1d
from tqdm import tqdm

label_group_map = {'N': 'N', 'L': 'N', 'R': 'N', 'V': 'V', '/': 'Q', 'A': 'S', 'F': 'F', 'f': 'Q', 'j': 'S', 'a': 'S',
                   'E': 'V', 'J': 'S', 'e': 'S', 'Q': 'Q', 'S': 'S'}


# %%
def iqr_normalize(data):
    # 1사분위수(Q1)와 3사분위수(Q3) 계산
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    # IQR을 이용한 데이터 정규화
    normalized_data = (data - Q1) / IQR
    return normalized_data


# %%
def min_max_normalize(data):
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data


# %%
# detrend
def detrend(signal, Lambda):
    signal_length = len(signal)

    H = np.identity(signal_length)

    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot((H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), signal)
    return filtered_signal


# %%
# resample
def resample_signal(ori_signal, original_fs, target_fs):
    num_samples = int(len(ori_signal) * target_fs / original_fs)
    resampled_signal = resample(ori_signal, num_samples)
    return resampled_signal


# %%
def write_log_with_lock(log_file, message):
    with open(log_file, "a") as log:
        fcntl.flock(log, fcntl.LOCK_EX)  # 파일 잠금 설정
        log.write(message + "\n")
        log.flush()  # 로그를 즉시 디스크에 기록
        fcntl.flock(log, fcntl.LOCK_UN)  # 파일 잠금 해제


# %% butterworth filter (LPF)

# def low_pass_filter(signal, cutoff, fs, order=2):
#     """
#     Low-pass filter를 적용하는 함수

#     Parameters:
#     - signal: 입력 신호 (1D numpy array)
#     - cutoff: 컷오프 주파수 (Hz)
#     - fs: 샘플링 주파수 (Hz)
#     - order: 필터의 차수 (default = 2)

#     Returns:
#     - filtered_signal: 필터링된 신호 (1D numpy array)
#     """
#     nyquist = 0.5 * fs  # 나이퀴스트 주파수 계산
#     normal_cutoff = cutoff / nyquist  # 정규화된 컷오프 주파수
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)  # Butterworth 필터 설계
#     filtered_signal = filtfilt(b, a, signal)  # 필터 적용
#     return filtered_signal

def low_pass_filter(signal, cutoff, fs, numtaps=101):
    """
    FIR Low-pass filter를 적용하는 함수.
    Parameters:
    - signal: 입력 신호 (1D numpy array)
    - cutoff: 컷오프 주파수 (Hz)
    - fs: 샘플링 주파수 (Hz)
    - numtaps: 필터 계수의 수 (default=101)
    Returns:
    - filtered_signal: 필터링된 신호 (1D numpy array)
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    fir_coeff = firwin(numtaps, normal_cutoff)
    filtered_signal = lfilter(fir_coeff, [1.0], signal)
    return filtered_signal


# %%
def resample_unequal(ts, fs_in, fs_out):
    """
    interploration
    """
    fs_in, fs_out = int(fs_in), int(fs_out)
    if fs_out == fs_in:
        return ts
    else:
        x_old = np.linspace(0, 1, num=fs_in, endpoint=True)
        x_new = np.linspace(0, 1, num=fs_out, endpoint=True)
        y_old = ts
        f = interp1d(x_old, y_old, kind='linear')
        y_new = f(x_new)
        return y_new


if __name__ == "__main__":

    path = './mit-bih-arrhythmia-database-1.0.0'
    save_path = './MIT'
    # valid_lead = ['MLII', 'II', 'I', 'MLI', 'V5']
    valid_lead = ['MLII']
    fs_out = 100
    test_ratio = 0.2
    signal_dict = {"N": 0, "V": 0, "S": 0, "Q": 0, "F": 0}

    record_files = [f for f in os.listdir(path) if f.endswith('.dat')]
    for record_file in record_files:
        if all(value > 10 for value in signal_dict.values()):
            break
        record_name = os.path.splitext(record_file)[0]
        record = wfdb.rdrecord(os.path.join(path, record_name))
        # .atr 파일을 이용해 Annotation 객체 가져오기 (주석 파일)
        annotation = wfdb.rdann(os.path.join(path, record_name), 'atr')
        original_signal = record.p_signal
        original_signal = np.array(original_signal).T[0]
        original_fs = record.fs
        target_fs = 100
        for Sig_num in range(0, len(original_signal) - 5, 18000):
            segment_signal = original_signal[Sig_num:Sig_num + 18000]
            segment_signal = low_pass_filter(segment_signal, cutoff=30, fs=360, )
            segment_signal = resample_signal(segment_signal, original_fs, target_fs)
            segment_signal = detrend(segment_signal, 100)
            for Sig_num2 in range(0, 5000, 1000):
                window = segment_signal[Sig_num2:Sig_num2 + 1000]
                sym = ""
                for sym_num in range(len(annotation.symbol)):
                    if annotation.sample[sym_num] <= Sig_num + (3.6 * Sig_num2) + 3600 and annotation.sample[
                        sym_num] >= Sig_num + (3.6 * Sig_num2):
                        try:
                            temp = label_group_map[annotation.symbol[sym_num]]
                        except:
                            continue
                        if temp == "N" and (sym == "" or sym == "N"):
                            sym = "N"
                        elif temp == "S":
                            sym = "S"
                        elif temp == "V":
                            sym = "V"
                        elif temp == "Q":
                            sym = "Q"
                        elif temp == "F":
                            sym = "F"
                if sym == "":
                    continue
                signal_dict[sym] += 1

                window = min_max_normalize(iqr_normalize(window))
                # 파일 이름 생성 및 저장
                save_directory = './MIT/' + sym + '/'
                file_name = sym + "_" + str(signal_dict[sym]) + ".csv"
                np.savetxt(save_directory + file_name, window, delimiter=",")
