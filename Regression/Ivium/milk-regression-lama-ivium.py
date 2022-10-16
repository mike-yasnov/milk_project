import os
import time
import requests
import pickle

# Essential DS libraries
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch

# LightAutoML presets, task and report generation
from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
from lightautoml.tasks import Task
from lightautoml.report.report_deco import ReportDeco

import warnings
warnings.filterwarnings('ignore')

N_THREADS = 42
N_FOLDS = 5
RANDOM_STATE = 42
TEST_SIZE = 0.2
TIMEOUT = 10*3600
TARGET_NAME = 'concentration'

df = pd.read_csv('../../all_data_ivium_new_data.csv')
df = df.drop([df.columns[0], 'antibiotic', 'path'], axis=1)

np.random.seed(RANDOM_STATE)
torch.set_num_threads(N_THREADS)

task = Task('reg')

roles = {
    'target': TARGET_NAME,
}

automl = TabularUtilizedAutoML(
    task = task, 
    timeout = TIMEOUT,
    cpu_limit = N_THREADS,
    tuning_params = {'max_tuning_time': 900},
    reader_params = {'n_jobs': N_THREADS}
)

oof_pred = automl.fit_predict(df, roles = roles, verbose = 1)

pickle.dump(automl, open('lama_ivium_regression_v1.sav', 'wb'))
