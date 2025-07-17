#!/bin/bash
TEST_CSV_PATH="C:/Users/Asus/Desktop/TCT_data/csvfiles/test.csv"
MODEL_NAME='vit_small_patch16_224'
NUM_EPOCHS=40

echo "========Fold1========"
LOG="C:/Users/Asus/Desktop/TCT_data/detvit/fold1/logs"
SAVE_MODEL_PATH="C:/Users/Asus/Desktop/TCT_data/detvit/fold1/f1.pth"
TRAIN_CSV_PATH="C:/Users/Asus/Desktop/TCT_data/csvfiles/fold1/train.csv"
VAL_CSV_PATH="C:/Users/Asus/Desktop/TCT_data/csvfiles/fold1/val.csv"
FOLD=1
python train.py \
    --model_name $MODEL_NAME \
    --logdir $LOG \
    --num_epochs $NUM_EPOCHS \
    --save_model_path $SAVE_MODEL_PATH \
    --train_csv_path $TRAIN_CSV_PATH \
    --val_csv_path $VAL_CSV_PATH \
    --test_csv_path $TEST_CSV_PATH \
    --fold $FOLD\
    AdamW


# echo "========Fold2========"
# LOG="C:/Users/Asus/Desktop/TCT_data/fasterrcnn/fold2/logs"
# SAVE_MODEL_PATH="C:/Users/Asus/Desktop/TCT_data/fasterrcnn/fold2/f2.pth"
# TRAIN_CSV_PATH="C:/Users/Asus/Desktop/TCT_data/csvfiles/fold2/train.csv"
# VAL_CSV_PATH="C:/Users/Asus/Desktop/TCT_data/csvfiles/fold2/val.csv"
# FOLD=2
# python train.py \
#     --model_name $MODEL_NAME \
#     --logdir $LOG \
#     --num_epochs $NUM_EPOCHS \
#     --save_model_path $SAVE_MODEL_PATH \
#     --train_csv_path $TRAIN_CSV_PATH \
#     --val_csv_path $VAL_CSV_PATH \
#     --test_csv_path $TEST_CSV_PATH \
#     --fold $FOLD\
#     AdamW


# echo "========Fold3========"
# LOG="C:/Users/Asus/Desktop/TCT_data/fasterrcnn/fold3/logs"
# SAVE_MODEL_PATH="C:/Users/Asus/Desktop/TCT_data/fasterrcnn/fold3/f3.pth"
# TRAIN_CSV_PATH="C:/Users/Asus/Desktop/TCT_data/csvfiles/fold3/train.csv"
# VAL_CSV_PATH="C:/Users/Asus/Desktop/TCT_data/csvfiles/fold3/val.csv"
# FOLD=3
# python train.py \
#     --model_name $MODEL_NAME \
#     --logdir $LOG \
#     --num_epochs $NUM_EPOCHS \
#     --save_model_path $SAVE_MODEL_PATH \
#     --train_csv_path $TRAIN_CSV_PATH \
#     --val_csv_path $VAL_CSV_PATH \
#     --test_csv_path $TEST_CSV_PATH \
#     --fold $FOLD\
#     AdamW


# echo "========Fold4========"
# LOG="C:/Users/Asus/Desktop/TCT_data/fasterrcnn/fold4/logs"
# SAVE_MODEL_PATH="C:/Users/Asus/Desktop/TCT_data/fasterrcnn/fold4/f4.pth"
# TRAIN_CSV_PATH="C:/Users/Asus/Desktop/TCT_data/csvfiles/fold4/train.csv"
# VAL_CSV_PATH="C:/Users/Asus/Desktop/TCT_data/csvfiles/fold4/val.csv"
# FOLD=4
# python train.py \
#     --model_name $MODEL_NAME \
#     --logdir $LOG \
#     --num_epochs $NUM_EPOCHS \
#     --save_model_path $SAVE_MODEL_PATH \
#     --train_csv_path $TRAIN_CSV_PATH \
#     --val_csv_path $VAL_CSV_PATH \
#     --test_csv_path $TEST_CSV_PATH \
#     --fold $FOLD\
#     AdamW


# echo "========Fold5========"
# LOG="C:/Users/Asus/Desktop/TCT_data/fasterrcnn/fold5/logs"
# SAVE_MODEL_PATH="C:/Users/Asus/Desktop/TCT_data/fasterrcnn/fold5/f5.pth"
# TRAIN_CSV_PATH="C:/Users/Asus/Desktop/TCT_data/csvfiles/fold5/train.csv"
# VAL_CSV_PATH="C:/Users/Asus/Desktop/TCT_data/csvfiles/fold5/val.csv"
# FOLD=5
# python train.py \
#     --model_name $MODEL_NAME \
#     --logdir $LOG \
#     --num_epochs $NUM_EPOCHS \
#     --save_model_path $SAVE_MODEL_PATH \
#     --train_csv_path $TRAIN_CSV_PATH \
#     --val_csv_path $VAL_CSV_PATH \
#     --test_csv_path $TEST_CSV_PATH \
#     --fold $FOLD \
#     AdamW

echo "Done!"
