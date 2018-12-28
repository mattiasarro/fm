export BERT_BASE_DIR=data/uncased_L-12_H-768_A-12
export GLUE_DIR=data/glue_data
export SEME_DIR=data/stance_detection

# python bert/run_classifier.py \
#   --task_name=MRPC \
#   --do_train=true \
#   --do_eval=true \
#   --data_dir=$GLUE_DIR/MRPC \
#   --vocab_file=$BERT_BASE_DIR/vocab.txt \
#   --bert_config_file=$BERT_BASE_DIR/bert_config.json \
#   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
#   --max_seq_length=128 \
#   --train_batch_size=32 \
#   --learning_rate=2e-5 \
#   --num_train_epochs=3.0 \
#   --output_dir=/tmp/mrpc_output/

python bert/run_classifier.py \
    --task_name=seme \
    --do_train=true \
    --do_eval=true \
    --data_dir=$SEME_DIR \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --max_seq_length=128 \
    --train_batch_size=32 \
    --learning_rate=2e-5 \
    --num_train_epochs=3.0 \
    --output_dir=/tmp/seme_output/
