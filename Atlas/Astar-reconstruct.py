import math
import random
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

from preprocess import txt_to_df


class Astar_LSPR:
    def __init__(self, root_data: Path, seed: int=123):
        self.seed_everything(seed)
        root_data = Path(root_data)
        data = pd.read_excel(root_data, header=0)
        self.raw_data = self.preprocess(data)

    def seed_everything(self, seed: int=123):
        random.seed(seed)

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data[(data['wave_name']<'20230501') & (data['2nd_peak_wave'] >0) & (data['label']=='H')]
        data = data.sort_values(by='2nd_peak_wave', ascending=True)
        data = data.reset_index(drop=True)
        return data

    def run(self, target_list: list=[600], epoch: int=500):
        self.default_value = 20
        self.max_step_in_all_run = 50
        self.max_step_early_stopping = 200
        self.max_thres = 10  # stopping threshold
        for target in target_list:  # [600, 650, 700, 750, 800, 850, 900]
            for _ in tqdm(range(epoch)):
                self.run_once(target)

    def run_once(self, target, delta_cnt=5, HCL_thres=0.06, AgNO3_thres=0.06):
        cur_step = 1
        data = self.init_astar(select_near=1, select_far=5, target=target, thres=50)
        while cur_step < self.max_step_early_stopping and data['is_open'].sum() > 0:
            # print(f"[STEP {cur_step}] ---------------")
            data_ch = self.find_path(data, col='si_ucb', topk=1)
            data, real_peak_wave = self.update_zi(data, data_ch, cur_step, target)
            flag = False
            for wave in real_peak_wave:
                if abs(wave - target) < self.max_thres:
                    flag = True
                    break
            if flag:
                break
            data = self.update_si(data, data_ch, delta_cnt, HCL_thres, AgNO3_thres)
            data = self.update_ucb(data, cur_step)
            cur_step += 1
        if flag and cur_step < self.max_step_in_all_run:
            Path('./output').mkdir(parents=True, exist_ok=True)
            root_save = Path(f'./output/{self.__class__.__name__}_{target}.xlsx')
            data[data['is_close']==1].sort_values(by='step', ascending=True).to_excel(root_save, header=True, index=False)
            self.max_step_in_all_run = cur_step

    def init_astar(self, select_near=1, select_far=3, target=850, thres=100):
        data = self.raw_data.copy()
        data.loc[:, 'is_close'] = 0
        data.loc[:, 'is_open'] = 0
        data.loc[:, 'zi'] = 0
        data.loc[:, 'sn'] = 0
        data.loc[:, 'si'] = 0  # mean score
        data.loc[:, 'si_ucb'] = 0  # mean score of UCB
        data.loc[:, 'step'] = -1
        p_index = []
        if select_near:
            t_data = data[(data['2nd_peak_wave']>target-thres) & (data['2nd_peak_wave']<target+thres)]
            t_index = random.sample(t_data.index.to_list(), min(select_near, len(t_data)))
            p_index.append(t_index)
        if select_far:
            t_data = data[(data['2nd_peak_wave']<target-thres) & (data['2nd_peak_wave']>target+thres)]
            t_index = random.sample(t_data.index.to_list(), min(select_far, len(t_data)))
            p_index.append(t_index)
        for p in p_index:
            data.loc[p, 'is_open'] = 1
            data.loc[p, 'si'] = self.default_value
            data.loc[p, 'si_ucb'] = self.default_value
        return data

    def find_path(self, data: pd.DataFrame, col='si', topk=1) -> pd.DataFrame:
        """get_max_si_from_open"""
        open_set = data[data['is_open']==1]
        ch_index = open_set.sort_values([col], ascending=False).head(topk).index.values
        data_ch = data.loc[ch_index, :]
        return data_ch

    def update_zi(self, data: pd.DataFrame, data_ch: pd.DataFrame, step, target):
        """build_experiment"""
        data.loc[data_ch.index.values, 'is_open'] = 0
        data.loc[data_ch.index.values, 'is_close'] = 1
        data.loc[data_ch.index.values, 'step'] = int(step)
        for idx in data_ch.index.values:
            data.loc[idx, 'zi'] = self.heuristic(data.loc[idx, '2nd_peak_wave'], target)
        return data, data_ch['2nd_peak_wave'].values

    def update_si(self, data: pd.DataFrame, data_ch: pd.DataFrame, 
                  delta_cnt=5, HCL_thres=0.06, AgNO3_thres=0.06):
        cnt = 0
        for ch_idx in data_ch.index.values:
            idx = ch_idx - 1
            ch_zi = data.loc[ch_idx, 'zi']
            while cnt < delta_cnt and idx >= 0 \
                and abs(data.loc[idx, '3.6542M 盐酸/mL']-data_ch.loc[ch_idx,'3.6542M 盐酸/mL']) < HCL_thres \
                and abs(data.loc[idx, '4mM AgNO3/mL']-data_ch.loc[ch_idx,'4mM AgNO3/mL']) < AgNO3_thres:
                if data.loc[idx, 'is_close'] == 0:
                    self.update_si_sn(data, idx, ch_zi)
                    cnt += 1
                idx -= 1
            cnt = 0
            idx = ch_idx + 1
            while cnt < delta_cnt and idx < len(data)\
                and abs(data.loc[idx, '3.6542M 盐酸/mL']-data_ch.loc[ch_idx,'3.6542M 盐酸/mL']) < HCL_thres \
                and abs(data.loc[idx, '4mM AgNO3/mL']-data_ch.loc[ch_idx,'4mM AgNO3/mL']) < AgNO3_thres:
                if data.loc[idx, 'is_close'] == 0:
                    self.update_si_sn(data, idx, ch_zi)
                    cnt += 1
                idx += 1
        return data

    def heuristic(self, x, target, MAX_VALUE=1000):
        """get_zi_func"""
        z = 1.0 - (1.0 * abs(x - target) / MAX_VALUE)
        return z

    def update_si_sn(self, data: pd.DataFrame, idx, ch_zi):
        if not self.check_is_init(data, idx):
            data.loc[idx, 'is_open'] = 1
            data.loc[idx, 'si'] = data.loc[idx, 'si'] * data.loc[idx, 'sn'] + ch_zi
            data.loc[idx, 'sn'] += 1 
            data.loc[idx, 'si'] /= data.loc[idx, 'sn']

    def update_ucb(self, data: pd.DataFrame, step, alpha=0.1):
        for idx in data[data['is_open']==1].index.values:
            if not self.check_is_init(data, idx):
                data.loc[idx, 'si_ucb'] = data.loc[idx, 'si'] + alpha * math.sqrt(2.0 * math.log(step) / data.loc[idx, 'sn'])
        return data

    def check_is_init(self, data: pd.DataFrame, idx):
        return data.loc[idx, 'si_ucb'] == self.default_value

    def vis_hist(self, col_name: str):
        print(self.raw_data)
        self.raw_data.plot.hist(y=col_name, bins=100)
        plt.show()


class Astar_FWHM(Astar_LSPR):
    def run(self, target_list: list=[700], epoch: int=500):
        self.default_value = 20
        self.max_step_in_all_run = 35
        self.max_step_early_stopping = 100
        self.max_thres = 10  # stopping threshold
        for target in target_list:  # [600, 650, 700, 750, 800, 850, 900]
            for _ in tqdm(range(epoch)):
                self.run_once(target)

    def run_once(self, target, delta_cnt=5, HCL_thres=0.06, AgNO3_thres=0.06):
        cur_step = 1
        data = self.init_astar(select_near=3, select_far=0, target=target, thres=20)
        while cur_step < self.max_step_early_stopping and data['is_open'].sum() > 0:
            # print(f"[STEP {cur_step}] ---------------")
            data_ch = self.find_path(data, col='si_ucb', topk=1)
            data, _ = self.update_zi(data, data_ch, cur_step, target)
            data = self.update_si(data, data_ch, delta_cnt, HCL_thres, AgNO3_thres)
            data = self.update_ucb(data, cur_step)
            cur_step += 1
        if cur_step < self.max_step_in_all_run:
            Path('./output').mkdir(parents=True, exist_ok=True)
            root_save = Path(f'./output/{self.__class__.__name__}_{target}.xlsx')
            data[data['is_close']==1].sort_values(by='step', ascending=True).to_excel(root_save, header=True, index=False)
            self.max_step_in_all_run = cur_step

    def heuristic(self, x, target, MAX_VALUE=1000):
        z = 1.0 - (x / MAX_VALUE)
        return z


class Astar_RATIO(Astar_LSPR):
    def preprocess(self, data):
        data = data[(data['wave_name']<'20230501') & (data['2nd_peak_wave'] >0)]
        data = data.sort_values(by='2nd_peak_wave', ascending=True)
        data = data.reset_index(drop=True)
        return data

    def run(self, target_list: list=[700], epoch: int=200):
        self.default_value = 20
        self.max_step_in_all_run = 15
        self.max_step_early_stopping = 100
        self.max_thres = 10  # stopping threshold
        for target in target_list:  # [600, 650, 700, 750, 800, 850, 900]
            for _ in tqdm(range(epoch)):
                self.run_once(target)

    def run_once(self, target, delta_cnt=5, HCL_thres=0.06, AgNO3_thres=0.06):
        cur_step = 1
        data = self.init_astar(select_near=3, select_far=0, target=target, thres=20)
        while cur_step < self.max_step_early_stopping and data['is_open'].sum() > 0:
            # print(f"[STEP {cur_step}] ---------------")
            data_ch = self.find_path(data, col='si_ucb', topk=1)
            data, _ = self.update_zi(data, data_ch, cur_step, target)
            data = self.update_si(data, data_ch, delta_cnt, HCL_thres, AgNO3_thres)
            data = self.update_ucb(data, cur_step)
            cur_step += 1
        if cur_step < self.max_step_in_all_run:
            Path('./output').mkdir(parents=True, exist_ok=True)
            root_save = Path(f'./output/{self.__class__.__name__}_{target}.xlsx')
            data[data['is_close']==1].sort_values(by='step', ascending=True).to_excel(root_save, header=True, index=False)
            self.max_step_in_all_run = cur_step

    def heuristic(self, x, target, MAX_VALUE=4.0):
        z = x / MAX_VALUE
        return z


class Astar_LSPR_new(Astar_LSPR):
    def __init__(self, root_txt: Path, seed: int=123):
        self.seed_everything(seed)
        root_txt = Path(root_txt)
        self.raw_data = self.preprocess(root_txt)

    def preprocess(self, root_txt: Path) -> pd.DataFrame:
        data = txt_to_df(root_txt)
        data = data.sort_values(by='2nd_peak_wave', ascending=True)
        data = data.reset_index(drop=True)
        return data

    def run_once(self, target, delta_cnt=5, HCL_thres=0.06, AgNO3_thres=0.06, Crystal_thres=0.06):
        cur_step = 1
        data = self.init_astar(select_near=1, select_far=5, target=target, thres=70)
        while cur_step < self.max_step_early_stopping and data['is_open'].sum() > 0:
            # print(f"[STEP {cur_step}] ---------------")
            data_ch = self.find_path(data, col='si_ucb', topk=1)
            data, real_peak_wave = self.update_zi(data, data_ch, cur_step, target)
            flag = False
            for wave in real_peak_wave:
                if abs(wave - target) < self.max_thres:
                    flag = True
                    break
            if flag:
                break
            data = self.update_si(data, data_ch, delta_cnt, HCL_thres, AgNO3_thres)
            data = self.update_ucb(data, cur_step)
            cur_step += 1
        if flag and cur_step < self.max_step_in_all_run:
            Path('./output').mkdir(parents=True, exist_ok=True)
            root_save = Path(f'./output/{self.__class__.__name__}_{target}.xlsx')
            data[data['is_close']==1].sort_values(by='step', ascending=True).to_excel(root_save, header=True, index=False)
            self.max_step_in_all_run = cur_step

    def update_si(self, data: pd.DataFrame, data_ch: pd.DataFrame, 
                  delta_cnt=5, HCL_thres=0.06, AgNO3_thres=0.06, Crystal_thres=0.06):
        cnt = 0
        for ch_idx in data_ch.index.values:
            idx = ch_idx - 1
            ch_zi = data.loc[ch_idx, 'zi']
            while cnt < delta_cnt and idx >= 0 \
                and abs(data.loc[idx, '3.6542M 盐酸/mL']-data_ch.loc[ch_idx,'3.6542M 盐酸/mL']) < HCL_thres \
                and abs(data.loc[idx, '4mM AgNO3/mL']-data_ch.loc[ch_idx,'4mM AgNO3/mL']) < AgNO3_thres \
                and abs(data.loc[idx, '晶种/mL']-data_ch.loc[ch_idx,'晶种/mL']) < Crystal_thres:
                if data.loc[idx, 'is_close'] == 0:
                    self.update_si_sn(data, idx, ch_zi)
                    cnt += 1
                idx -= 1
            cnt = 0
            idx = ch_idx + 1
            while cnt < delta_cnt and idx < len(data)\
                and abs(data.loc[idx, '3.6542M 盐酸/mL']-data_ch.loc[ch_idx,'3.6542M 盐酸/mL']) < HCL_thres \
                and abs(data.loc[idx, '4mM AgNO3/mL']-data_ch.loc[ch_idx,'4mM AgNO3/mL']) < AgNO3_thres\
                and abs(data.loc[idx, '晶种/mL']-data_ch.loc[ch_idx,'晶种/mL']) < Crystal_thres:
                if data.loc[idx, 'is_close'] == 0:
                    self.update_si_sn(data, idx, ch_zi)
                    cnt += 1
                idx += 1
        return data


if __name__ == '__main__':
    # root_txt = './初始数据/目标：LSPR = 670 nm/20230331-2.txt'
    # astar = Astar_LSPR_new(root_txt)
    root_txt = './初始数据/目标：LSPR = 780 nm/20230423-2.txt'
    astar = Astar_LSPR_new(root_txt)
    # root_txt = './初始数据/目标：LSPR = 820 nm/20230505-1.txt'
    # astar = Astar_LSPR_new(root_txt)
    astar.run([670])
    astar.vis_hist('2nd_peak_wave')

