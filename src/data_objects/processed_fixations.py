import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from lifelines import CoxPHFitter

from utils.post_processing import run_post_processing

class ProcessedFixations(pd.DataFrame):
    """
    Wrapper for processed data, to make it more convenient to work with.

    This way you get the freedom of a dataframe and some additional functionality through object methods.
    """
    def __init__(self):
        result = run_post_processing()
        return super().__init__(result)
    
    def fit_cox(self, formula = "Age_Group_Cluster", duration_col = "duration"):
        coxph = CoxPHFitter()
        coxph.fit(self, duration_col = duration_col, formula = formula)
        coxph.print_summary()
    
    def fit_mixed_effects(self):
        pass