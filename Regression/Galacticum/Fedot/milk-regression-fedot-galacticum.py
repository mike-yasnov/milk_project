import numpy as np
import pandas as pd

from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task
from sklearn.metrics import *

# load data with 5 antibiotics and milk created on ivium device
data_5_antibiotics = pd.read_csv('../../galacticum_12_plus_3_cycles_2022-07-21_all_cycles.csv')
data_5_antibiotics.drop([data_5_antibiotics.columns[0]], axis=1, inplace=True)


test_data_5_antibiotics = pd.read_csv('../../test_data_df_galacticum_12.csv')
test_data_5_antibiotics.drop([test_data_5_antibiotics.columns[0]], axis=1, inplace=True)
y_test_true = test_data_5_antibiotics[test_data_5_antibiotics.columns[-2]].values

# create func for input data

def prepare_data(dataframe: pd.DataFrame) -> InputData:
    features_names = list(dataframe.columns[:1])
    features_names.append(dataframe.columns[-1])

    reg_task = Task(TaskTypesEnum.regression)
    reg_input = InputData(idx=np.arange(0, len(dataframe)),
                            features=np.array(dataframe[features_names]),
                            target=np.array(dataframe[dataframe.columns[-2]]),
                            task=reg_task, data_type=DataTypesEnum.table)

    return reg_input

def prepare_test_data(dataframe: pd.DataFrame) -> InputData:
    features_names = list(dataframe.columns[:1])
    features_names.append(dataframe.columns[-1])

    reg_task = Task(TaskTypesEnum.regression)
    reg_input = InputData(idx=np.arange(0, len(dataframe)),
                          features=np.array(dataframe[features_names]),
                          target=None, task=reg_task, data_type=DataTypesEnum.table)

    return reg_input

prepared_train_data = prepare_data(data_5_antibiotics)
prepare_test_data = prepare_test_data(test_data_5_antibiotics)

model = Fedot(problem='regression', seed=42, timeout=36000, preset='auto')

pipeline = model.fit(prepared_train_data)

pred = pipeline.predict(prepare_test_data, output_mode='full_probs')
pred = pred.predict

MAE = mean_absolute_error(y_test_true, pred)
MSE = mean_squared_error(y_test_true, pred)
R2 = r2_score(y_test_true, pred)

print(f'MAE = {MAE}')
print(f'MSE = {MSE}')
print(f'R2 = {R2}')

pipeline.save('milk_regression_fedot_galacticum')