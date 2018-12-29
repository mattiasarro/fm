export BERT_BASE_DIR=data/uncased_L-12_H-768_A-12
export GLUE_DIR=data/glue_data
export SEME_DIR=data/stance_detection

$PYTHON bert/custom_run.py \
    --task=predict \
    --data_dir=$SEME_DIR \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --max_seq_length=128 \
    --output_dir=predictions
