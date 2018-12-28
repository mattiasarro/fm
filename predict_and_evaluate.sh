bert/predict.sh
python3 write_preds_file.py
perl eval.pl data/stance_detection/SemEval2016-Task6-subtaskA-testdata.txt predictions/test_results_processed.tsv
