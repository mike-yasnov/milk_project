import pandas as pd
import numpy as np

class Galacticum:
    def __init__(self, fname):
        self.fname = fname
        self.data, self.voltage, self.current = self._read_data()
        self.data_for_prediction = self._get_dataframe_for_model()


    def _read_data(self):
        data = pd.read_csv(self.fname, header=None, prefix='column_')
        voltage = data[data.columns[0]].values
        current = data[data.columns[1]].values
        if len(voltage) != 15600:
            return None, None, None
        else:
            return data, voltage, current

    def _get_dataframe_for_model(self):
        model_dataframe = pd.DataFrame(data=[self.current], columns=self.voltage)
        return model_dataframe
