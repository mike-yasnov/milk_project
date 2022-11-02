import numpy as np
import pandas as pd
from app.paths.paths import *

# TODO
class Pipeline_init:
    def __init__(self, path_to_pipeline):
        self.path_to_pipeline = path_to_pipeline
        self.predicted_class = None
        self.predicted_conc = None
        pass
