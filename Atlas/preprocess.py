import pandas as pd
from pathlib import Path
import re


def txt_to_df(root_txt: Path):
    root_txt = Path(root_txt)
    root_df = root_txt.parent / (root_txt.stem + '.xlsx')
    with open(root_txt, 'r', encoding='gbk') as f:
        lines = f.readlines()
    spectrum_start_index, spectrum_end_index = None, None
    for i, line in enumerate(lines):
        if line.strip().startswith("波长"):
            spectrum_start_index = i
        if line.strip().startswith("结果"):
            spectrum_end_index = i
            break
    spectrum_data = lines[spectrum_start_index:spectrum_end_index]
    spectrum_data = [re.sub(r'\t+', '\t', line.strip()) for line in spectrum_data if line.strip()]
    header_columns = spectrum_data[0].split('\t')[:13]
    data_rows = []
    for row in spectrum_data[1:]:
        cells = row.split('\t')[:13]
        if len(cells) == 13:
            data_rows.append([float(cell) for cell in cells])
    data_rows = [list(row) for row in zip(*data_rows)]
    min_wavelength = data_rows[0][0]
    max_wavelengths = []
    for row in data_rows[1:]:
        max_wavelengths.append(row.index(max(row)) + min_wavelength)
    data = pd.read_excel(str(root_df), header=0)
    data['2nd_peak_wave'] = max_wavelengths
    return data


if __name__ == '__main__':
    df = txt_to_df(root_txt='./初始数据/目标：LSPR = 670 nm/20230331-2.txt')
    print(df)
