export BERT_BASE_DIR=data/uncased_L-12_H-768_A-12
export GLUE_DIR=data/glue_data
export SEME_DIR=data/stance_detection

export TASK=$1
export HPARAMS_ID=$2
export STEP=$3

$PYTHON bert/custom_run.py \
    --task_name=articles \
    --task=$TASK \
    --data_dir=$SEME_DIR \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=/tmp/tune/$HPARAMS_ID/model.ckpt-$STEP \
    --max_seq_length=128 \
    --output_dir=article_preds/
