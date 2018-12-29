import pandas as pd
import csv

dataset_path = 'data/stance_detection/'
test_results_fn = "predictions/test_results.tsv"
test_results_processed_fn = "predictions/test_results_processed.tsv"
STANCES = ["AGAINST", "NONE", "FAVOR"]
TARGETS = [
    "Hillary Clinton",
    "Feminist Movement",
    "Legalization of Abortion",
    "Atheism",
    "Climate Change is a Real Concern",
]

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

def print_metrics(target=None):
    num_gold = {c: 0 for c in STANCES}
    num_guess = {c: 0 for c in STANCES}
    num_tp = {c: 0 for c in STANCES}

    data, columns = read('SemEval2016-Task6-subtaskA-testdata-gold.txt', delimiter='\t')
    for l, pred in zip(data, predictions()):
        if target and l[1] != target: continue
        label = l[-1]
        num_gold[label] += 1
        num_guess[pred] += 1
        if label == pred:
            num_tp[pred] += 1

    if target: print(target)

    p, r, f = {}, {}, {}
    for c in STANCES:
        p[c] = num_tp[c] / num_guess[c] if num_guess[c] != 0 else 0
        r[c] = num_tp[c] / num_gold[c] if num_gold[c] != 0 else 0
        f[c] = (2 * p[c] * r[c] / (p[c] + r[c])) if p[c] + r[c] != 0 else 0

        print("%s precision = %.4f  recall = %.4f  F = %.4f  predicted = %i  labelled = %i" %
              (c, p[c], r[c], f[c], num_guess[c], num_gold[c]))

    print("macro F = %.4f" % (f["FAVOR"] + f["AGAINST"] / 2))

def print_metrics_per_target():
    for target in TARGETS + [None]:
        print_metrics(target)
        print("")
