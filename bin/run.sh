export BERT_BASE_DIR=data/uncased_L-12_H-768_A-12
export GLUE_DIR=data/glue_data
export SEME_DIR=data/stance_detection

$PYTHON bert/custom_run.py \
    --task=$1 \
    --data_dir=$SEME_DIR \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --max_seq_length=128 \
    --train_batch_size=32 \
    --learning_rate=2e-5 \
    --num_train_epochs=5.0 \
    --output_dir=/tmp/seme_output/
