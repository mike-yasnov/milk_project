import os
import time
import requests

# Essential DS libraries
import numpy as np
import pandas as pd
from sklearn.metrics import *
import matplotlib.pyplot as plt
import torch

# LightAutoML presets, task and report generation
from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
from lightautoml.tasks import Task
from lightautoml.report.report_deco import ReportDeco

import pickle
import warnings
warnings.filterwarnings('ignore')


N_THREADS = 4
N_FOLDS = 5
RANDOM_STATE = 42
TEST_SIZE = 0.2
TIMEOUT = 10*3600
TARGET_NAME = 'target'

np.random.seed(RANDOM_STATE)
torch.set_num_threads(N_THREADS)


df = pd.read_csv("galacticum_12_plus_3_cycles_2022-07-21_all_cycles.csv")
df = df.drop([df.columns[0], 'substance'], axis=1)

task = Task('reg')
roles = {
    'target': TARGET_NAME,
}

automl = TabularUtilizedAutoML(
    task = task, 
    timeout = TIMEOUT,
    cpu_limit = N_THREADS,
    tuning_params = {'max_tuning_time': 120},
    reader_params = {'n_jobs': N_THREADS}
)

oof_pred = automl.fit_predict(df, roles = roles, verbose = 1)
pickle.dump(model, open('lama_galacticum_regression_v1.sav ', 'wb'))

