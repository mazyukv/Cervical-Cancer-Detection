TEST_CSV_PATH="C:/Users/Administrator/Desktop/Cervical-Cancer-Detection/csvfiles/test.csv"
MODEL_NAME='resnet50'
NUM_EPOCHS=40

echo "========Fold1========"
LOG="C:/Users/Administrator/Desktop/Cervical-Cancer-Detection/cnn2/fold1/logs"
SAVE_MODEL_PATH="C:/Users/Administrator/Desktop/Cervical-Cancer-Detection/cnn2/fold1/f1.pth"
TRAIN_CSV_PATH="C:/Users/Administrator/Desktop/Cervical-Cancer-Detection/csvfiles/fold1/train.csv"
VAL_CSV_PATH="C:/Users/Administrator/Desktop/Cervical-Cancer-Detection/csvfiles/fold1/val.csv"
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

echo "Done!"
