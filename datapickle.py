import sys
import pandas as pd
import os
import numpy as np
import pickle


path = sys.argv[1]
serial_path = os.path.splitext(path)[0] + '.pickle'
arr = pd.read_csv(path, sep=",", header=None, index_col=None, dtype="float32")
print(arr.shape)
pickle.dump(arr, open(serial_path, 'wb'))
