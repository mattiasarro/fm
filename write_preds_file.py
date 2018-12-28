import helpers as h

data, columns = h.read('SemEval2016-Task6-subtaskA-testdata.txt', delimiter='\t')
with open(h.test_results_processed_fn, "w") as f:
    f.write("\t".join(columns) + "\n")
    for l, pred_class in zip(data, h.predictions()):
        if l[-1] != "UNKNOWN": continue
        l[-1] = pred_class
        f.write("\t".join(l) + "\n")
