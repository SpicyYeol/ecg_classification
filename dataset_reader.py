import os
import glob
import scipy
import pandas as pd
import tqdm
import json
import numpy as np
import wfdb
import re
from numpy.distutils.conv_template import header
import ast

category_priority = {'S': 1, 'V': 2, 'Q': 3, 'N': 4}


def select_highest_priority_category(categories):
    # 유효한 카테고리만 필터링
    valid_categories = [cat for cat in categories if cat in category_priority]

    if not valid_categories:
        return 'Unknown'  # 유효한 카테고리가 없을 경우

    # 우선순위에 따라 정렬
    sorted_categories = sorted(valid_categories, key=lambda x: category_priority[x])

    # 우선순위가 가장 높은 카테고리 반환
    return sorted_categories[0]

def read_hea_file(hea_file_path,path):
    """
    .hea 파일을 읽어 헤더 정보를 파싱하는 함수

    Parameters:
    - hea_file_path: str, .hea 파일의 경로

    Returns:
    - header_info: dict, 헤더 정보가 담긴 딕셔너리
    """

    csv_file_path = 'ConditionNames_SNOMED-CT.csv'
    df = pd.read_csv(os.path.join(path,csv_file_path))
    df['Snomed_CT'] = df['Snomed_CT'].astype(str)
    snomed_to_acronym = dict(zip(df['Snomed_CT'], df['Acronym Name']))

    header_info = {}
    signal_info_list = []

    with open(hea_file_path, 'r') as f:
        lines = f.readlines()

        # 첫 번째 줄은 전체 기록에 대한 정보
        header_line = lines[0].strip()
        if header_line.startswith('#'):
            # 첫 줄이 주석인 경우 처리
            header_line = lines[1].strip()
            start_line = 2
        else:
            start_line = 1

        header_parts = header_line.split()
        header_info['record_name'] = header_parts[0]
        header_info['num_signals'] = int(header_parts[1]) if len(header_parts) > 1 else None
        # header_info['sampling_frequency'] = float(header_parts[2].split('/')[0]) if len(header_parts) > 2 else None
        # header_info['num_samples'] = int(header_parts[3]) if len(header_parts) > 3 else None
        # header_info['base_time'] = header_parts[4] if len(header_parts) > 4 else None
        # header_info['base_date'] = header_parts[5] if len(header_parts) > 5 else None
        numbers = re.findall(r'\d+', lines[15].strip())
        numbers = [num for num in numbers]

        replaced_names = []
        for num_str in numbers:
            acronym = snomed_to_acronym.get(num_str, None)
            if acronym:
                replaced_names.append(acronym)
            else:
                continue
                # replaced_names.append(num_str)  # 매칭되는 Acronym Name이 없으면 원래 숫자를 유지

        header_info['description'] = replaced_names
        # 신호별 정보 파싱
        for line in lines[start_line:]:
            line = line.strip()
            if not line or line.startswith('#'):
                continue  # 빈 줄이나 주석은 무시
            signal_parts = line.split()
            signal_info = {}

            # if len(signal_parts) >= 2:
            #     signal_info['filename'] = signal_parts[0]
            #     signal_info['format'] = signal_parts[1]
            # if len(signal_parts) >= 3:
            #     signal_info['gain'] = signal_parts[2]
            # if len(signal_parts) >= 4:
            #     signal_info['bit_resolution'] = int(signal_parts[3])
            # if len(signal_parts) >= 5:
            #     signal_info['zero_value'] = int(signal_parts[4])
            # if len(signal_parts) >= 6:
            #     signal_info['first_value'] = int(signal_parts[5])
            # if len(signal_parts) >= 7:
            #     signal_info['checksum'] = int(signal_parts[6])
            # if len(signal_parts) >= 8:
            #     signal_info['block_size'] = int(signal_parts[7])
            if len(signal_parts) >= 9:
                signal_info['description'] = ' '.join(signal_parts[8:])

            signal_info_list.append(signal_info)

        header_info['signals'] = signal_info_list

    return header_info

def find_mat_and_hea_files(directory):
    # 하위 디렉토리까지 모두 탐색하여 *.mat 및 *.hea 파일 찾기
    mat_files = []
    hea_files = []
    for root, dirs, files in os.walk(directory):
        mat_files.extend(glob.glob(os.path.join(root, '*.mat')))
        hea_files.extend(glob.glob(os.path.join(root, '*.hea')))

    # 파일 이름에서 확장자를 제거한 후 집합으로 변환
    mat_names = {os.path.splitext(file)[0] for file in mat_files}
    hea_names = {os.path.splitext(file)[0] for file in hea_files}

    # mat와 hea가 모두 존재하는 파일 이름 찾기
    common_names = list(mat_names & hea_names)

    return common_names

def find_files(directory, ext='*.mat'):
    # 하위 디렉토리까지 모두 탐색하여 *.mat 파일 찾기
    mat_files = []
    for root, dirs, files in os.walk(directory):
        for file in glob.glob(os.path.join(root, ext)):
            mat_files.append(file)
    return mat_files
# 데이터 로딩 함수 정의
def load_dataset_1(data_dict, src_dir, offset, labels_dict = None):
    mats = find_files(os.path.join(src_dir, data_dict['name']), '*.mat')
    all_contents = []
    for mat in mats[:offset]:
        mat_data = scipy.io.loadmat(mat)
        if 'ecg_raw' in mat_data:
            mat_contents = mat_data['ecg_raw'][0]
        elif 'ecg' in mat_data:
            mat_contents = mat_data['ecg'][0]
        else:
            print(f"Warning: '{mat}' does not contain 'ecg_raw' or 'ecg'")
            continue
        all_contents.append(mat_contents)
    return all_contents

# load af-classification dataset
def load_dataset_2(data_dict, src_dir, offset, labels_dict = None):
    mats = find_files(os.path.join(src_dir, data_dict['name'], 'training2017'), '*.mat')
    label_csv = os.path.join(src_dir, data_dict['name'], 'REFERENCE-v0.csv')
    label_data = pd.read_csv(label_csv)
    all_contents = []
    for mat in mats[:offset]:
        mat_data = scipy.io.loadmat(mat)
        label = label_data[label_data['filename'] == mat.split('\\')[-1].split('.')[0]].iloc[0]['label']
        if 'val' in mat_data:
            mat_contents = mat_data['val'][0]
        else:
            print(f"Warning: '{mat}' does not contain 'ecg_raw' or 'ecg'")
            continue
        all_contents.append({
            'data': mat_contents,
            'label': label
        })
    return all_contents

def load_dataset_3(data_dict, src_dir, offset, labels_dict):
    ann_path = os.path.join(src_dir, data_dict['name'], 'ptbxl_database.csv')
    df = pd.read_csv(ann_path)
    df_with_scp_codes = df[df['scp_codes'].notna()]
    heas = find_files(os.path.join(src_dir, data_dict['dataset_name'], 'records' + str(data_dict['fs'])), '*.hea')

    all_contents = []
    if offset == -1:
        offset = len(heas)
    for hea in tqdm(heas[:offset], desc='Loading data'):
        record = wfdb.rdrecord(hea[:-4])
        record_name = os.path.basename(hea)[:-4]
        record_metadata = df_with_scp_codes[df_with_scp_codes['filename_lr'].str.contains(record_name)]

        if not record_metadata.empty:
            scp_codes = record_metadata.iloc[0]['scp_codes']
            scp_codes_dict = json.loads(scp_codes.replace("'", "\""))
            one_hot_encoded = np.zeros(len(labels_dict), dtype=float)
            for code, value in scp_codes_dict.items():
                if code in labels_dict:
                    index = labels_dict[code]
                    one_hot_encoded[index] = value / 100.0

            ecg_data = record.p_signal[:, 1]
            all_contents.append({'ecg': ecg_data, 'labels': one_hot_encoded})
    return all_contents

def load_dataset_4(data_dict, src_dir, offset, labels_dict = None):
    heas = find_files(os.path.join(src_dir, data_dict['name'], 'records' + str(data_dict['fs'])), '*.hea')
    csv_file_path = os.path.join(src_dir, data_dict['name'],"ptbxl_database.csv")
    df = pd.read_csv(csv_file_path, usecols=['scp_codes', 'filename_hr'])

    def extract_filename_from_path(file_path):
        base_name = os.path.basename(file_path)
        filename_without_ext = os.path.splitext(base_name)[0]
        return filename_without_ext

    df['filename_hr_only'] = df['filename_hr'].apply(extract_filename_from_path)

    filename_to_scp_codes = dict(zip(df['filename_hr_only'], df['scp_codes']))

    def extract_filename(file_path):
        base_name = os.path.basename(file_path)  # 파일명.확장자 추출
        filename_without_ext = os.path.splitext(base_name)[0]  # 확장자 제거
        return filename_without_ext

    with open('dataset_4.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    code_to_category = {}

    for category, subcategories in dataset.items():
        for subcategory, codes in subcategories.items():
            for code, description in codes.items():
                code_to_category[code] = category  # 코드와 카테고리 매핑

    file_final_category_dict = {}
    all_contents = []
    for hea in heas[:offset]:
        record = wfdb.rdrecord(hea[:-4])
        #all_contents.append(record.p_signal[:,1])
        file_name = extract_filename(hea)
        matched_scp_codes = filename_to_scp_codes.get(file_name)
        matched_scp_codes = ast.literal_eval(matched_scp_codes)
        if matched_scp_codes:
            # matched_scp_codes의 코드들을 카테고리로 매핑
            categories = []
            for code in matched_scp_codes.keys():
                category = code_to_category.get(code)
                if category:
                    categories.append(category)
                else:
                    continue
                    #categories.append('Unknown')
            # 우선순위에 따라 최종 카테고리 선택
            final_category = select_highest_priority_category(categories)
            all_contents.append({
                'data': record.p_signal[:,1],
                'label': final_category[0]
            })
    return all_contents

def load_dataset_5(data_dict, src_dir, offset, labels_dict = None):
    # 여기에 다른 데이터셋에 대한 로직을 추가할 수 있습니다.
    return []

def load_dataset_6(data_dict, src_dir, offset, labels_dict = None):
    # 여기에 다른 데이터셋에 대한 로직을 추가할 수 있습니다.
    root_dir = os.path.join(src_dir,data_dict['name'], 'WFDBRecords')
    file_list = find_mat_and_hea_files(root_dir)

    with open('dataset_6.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    acronym_to_category = {}

    for category, subcategories in dataset.items():
        for subcategory, acronyms in subcategories.items():
            for acronym, description in acronyms.items():
                acronym_to_category[acronym] = category

    all_contents = []
    for mat in file_list[:offset]:
        mat_data = scipy.io.loadmat(mat + '.mat')
        hea_contents = read_hea_file(mat + '.hea',os.path.join(src_dir,data_dict['name']))
        mat_contents = mat_data['val'][2]
        for idx, sig_data in enumerate(hea_contents['signals']):
            if sig_data['description'] == 'II':
                mat_contents = mat_data['val'][2]

        label_tmp = hea_contents['description']
        label = []
        for name in label_tmp:
            category = acronym_to_category.get(name)
            if category:
                label.append(category)
            else:
                continue

        category_priority = {'S': 1, 'V': 2, 'Q': 3, 'N': 4}
        label = sorted(label, key=lambda x: category_priority[x])

        if len(label) == 0:
            label = ['N']

        all_contents.append({
            'data': mat_contents,
            'label': label[0]
        })


    return all_contents


def load_dataset_8(data_dict, src_dir, offset, labels_dict = None):
    # 여기에 다른 데이터셋에 대한 로직을 추가할 수 있습니다.
    return []
def load_generic_dataset(data_dict, src_dir, offset, labels_dict = None):
    # 여기에 다른 데이터셋에 대한 로직을 추가할 수 있습니다.
    return []
