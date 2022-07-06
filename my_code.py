import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

import pandas as pd
import joblib

path = sys.argv[1]
Xa = pd.read_table("Xa.csv", sep='\t')
data = pd.read_table(path, sep='\t')
names = data[data.columns[1:4]]
features = data[data.columns[5:len(data.columns)]]

classifier = joblib.load("best_cls_wout_weights.pkl")

std_data = lambda x: (x - Xa.min()) / (Xa.max() - Xa.min())
features = std_data(features)

result = classifier.predict_proba(features)
res_df = names.assign(resultProbability=[i[1] for i in result])

res_df.to_csv("result.csv", sep='\t')
