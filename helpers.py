import pandas as pd
import csv

dataset_path = 'data/stance_detection/'
test_results_fn = "predictions/test_results.tsv"
test_results_processed_fn = "predictions/test_results_processed.tsv"
STANCES = ["AGAINST", "NONE", "FAVOR"]

def train_data():
    data, columns = read('SemEval2016-Task6-subtaskA-traindata-gold.csv', quotechar='"')
    return pd.DataFrame(data, columns=columns)

def test_data():
    data, columns = read('SemEval2016-Task6-subtaskA-testdata-gold.txt', delimiter='\t')
    return pd.DataFrame(data, columns=columns)

def read(fn, **reader_kw):
    with open(dataset_path + fn, 'r',  encoding="iso-8859-1") as fin:
        reader = csv.reader(fin, **reader_kw)
        columns = next(reader)
        return [l for l in reader], columns

def stance_data():
    return pd.read_csv(dataset_path + "stance.csv")

def argmax(l):
    max_val, max_i = 0, None
    for i, p in enumerate(map(float, l)):
        if p > max_val:
            max_val = p
            max_i = i
    return max_i

def predictions():
    with open(test_results_fn, "r") as f:
        for row in csv.reader(f, delimiter='\t'):
            yield STANCES[argmax(row)]
