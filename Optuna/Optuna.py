import optuna
import pandas as pd
import numpy as np
from pathlib import Path
import random


class Optuna_LSPR:
    def __init__(self, root_data: Path, seed: int = 14):
        self.seed = seed
        self.seed_everything()
        root_data = Path(root_data)
        data = pd.read_excel(root_data, header=0)
        self.raw_data = self.preprocess(data)
        
    def seed_everything(self):
        random.seed(self.seed)
        np.random.seed(self.seed)

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data[(data['wave_name']<'20230501') & (data['2nd_peak_wave'] >0) & (data['label']=='H')]
        data = data.sort_values(by='2nd_peak_wave', ascending=True)
        data = data.reset_index(drop=True)
        return data

    def objective(self, trial, target):
        # Define parameter search space
        AgNO3 = trial.suggest_float('AgNO3', 
                                  self.raw_data['4mM AgNO3/mL'].min(),
                                  self.raw_data['4mM AgNO3/mL'].max())
        HCL = trial.suggest_float('HCL', 
                                self.raw_data['3.6542M 盐酸/mL'].min(),
                                self.raw_data['3.6542M 盐酸/mL'].max())

        # Find the closest experimental data point
        distances = np.sqrt(
            (self.raw_data['4mM AgNO3/mL'] - AgNO3)**2 + 
            (self.raw_data['3.6542M 盐酸/mL'] - HCL)**2
        )
        nearest_idx = distances.argmin()
        nearest_peak = self.raw_data.iloc[nearest_idx]['2nd_peak_wave']
        
        # Calculate the target value (minimize the difference from the target wavelength)
        # score = abs(nearest_peak - target)
        score = -(1.0 - (abs(nearest_peak - target) / 1000))
        
        # Record the best parameters found
        trial.set_user_attr('nearst', distances.min())
        trial.set_user_attr('peak_wave', nearest_peak)
        trial.set_user_attr('AgNO3_real', self.raw_data.iloc[nearest_idx]['4mM AgNO3/mL'])
        trial.set_user_attr('HCL_real', self.raw_data.iloc[nearest_idx]['3.6542M 盐酸/mL'])
        
        return score

    def run(self, target: int = 700, n_trials: int = 1000):
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.seed)
        )
        
        study.optimize(
            lambda trial: self.objective(trial, target),
            n_trials=n_trials
        )
        
        # Output the best results
        print("Best trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        print("  User attrs: ")
        for key, value in trial.user_attrs.items():
            print(f"    {key}: {value}")
            
        # Save results
        self.save_results(study, target)
        
        return study

    def save_results(self, study, target):
        # Result DataFrame
        trials_df = study.trials_dataframe()
        trials_df['peak_wave'] = [t.user_attrs.get('peak_wave') for t in study.trials]
        trials_df['AgNO3_real'] = [t.user_attrs.get('AgNO3_real') for t in study.trials]
        trials_df['HCL_real'] = [t.user_attrs.get('HCL_real') for t in study.trials]
        trials_df['nearst'] = [t.user_attrs.get('nearst') for t in study.trials]
        
        # Save results
        Path('./output').mkdir(parents=True, exist_ok=True)
        trials_df.to_excel(f'./output/{self.__class__.__name__}_{target}.xlsx', index=False)


class Optuna_FWHM(Optuna_LSPR):
    def objective(self, trial, target):
        AgNO3 = trial.suggest_float('AgNO3', 
                                  self.raw_data['4mM AgNO3/mL'].min(),
                                  self.raw_data['4mM AgNO3/mL'].max())
        HCL = trial.suggest_float('HCL', 
                                self.raw_data['3.6542M 盐酸/mL'].min(),
                                self.raw_data['3.6542M 盐酸/mL'].max())

        distances = np.sqrt(
            (self.raw_data['4mM AgNO3/mL'] - AgNO3)**2 + 
            (self.raw_data['3.6542M 盐酸/mL'] - HCL)**2
        )
        nearest_idx = distances.argmin()
        nearest_peak = self.raw_data.iloc[nearest_idx]['2nd_peak_wave']
        
        # The optimization objective of FWHM is to minimize the wavelength value
        score = nearest_peak
        
        trial.set_user_attr('peak_wave', nearest_peak)
        trial.set_user_attr('AgNO3_real', self.raw_data.iloc[nearest_idx]['4mM AgNO3/mL'])
        trial.set_user_attr('HCL_real', self.raw_data.iloc[nearest_idx]['3.6542M 盐酸/mL'])
        
        return score


class Optuna_RATIO(Optuna_LSPR):
    def objective(self, trial, target):
        AgNO3 = trial.suggest_float('AgNO3', 
                                  self.raw_data['4mM AgNO3/mL'].min(),
                                  self.raw_data['4mM AgNO3/mL'].max())
        HCL = trial.suggest_float('HCL', 
                                self.raw_data['3.6542M 盐酸/mL'].min(),
                                self.raw_data['3.6542M 盐酸/mL'].max())

        distances = np.sqrt(
            (self.raw_data['4mM AgNO3/mL'] - AgNO3)**2 + 
            (self.raw_data['3.6542M 盐酸/mL'] - HCL)**2
        )
        nearest_idx = distances.argmin()
        nearest_peak = self.raw_data.iloc[nearest_idx]['2nd_peak_wave']
        
        # The optimization goal of RATIO is to maximize the wavelength value
        score = -nearest_peak  
        trial.set_user_attr('peak_wave', nearest_peak)
        trial.set_user_attr('AgNO3_real', self.raw_data.iloc[nearest_idx]['4mM AgNO3/mL'])
        trial.set_user_attr('HCL_real', self.raw_data.iloc[nearest_idx]['3.6542M 盐酸/mL'])
        
        return score


if __name__ == '__main__':
    root_data = './data/20230320_20230628_result.xlsx'
    
    optimizer = Optuna_FWHM(root_data)
    study = optimizer.run(target=650, n_trials=100)
