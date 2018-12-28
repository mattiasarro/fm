import pandas as pd
import csv

dataset_path = 'stance_detection/'

def train_data():
    data = []
    with open(dataset_path + 'SemEval2016-Task6-subtaskA-traindata-gold.csv', 'r',  encoding="iso-8859-1") as fin:
        reader = csv.reader(fin, quotechar='"')
        columns = next(reader)
        for line in reader:
            data.append(line)

    return pd.DataFrame(data, columns=columns)

def test_data():
    data = []

    with open(dataset_path + 'SemEval2016-Task6-subtaskA-testdata-gold.txt', 'r',  encoding="iso-8859-1") as fin:
        reader = csv.reader(fin, delimiter='\t')
        columns = next(reader)
        for line in reader:
            data.append(line)

    return pd.DataFrame(data, columns=columns)

def stance_data():
    return pd.read_csv(dataset_path + "stance.csv")
