export CUDA_VISIBLE_DEVICES=0
for k in 50
do
  for r in 2
  do
    python dwgf.py \
        --dataset stackoverflow \
        --labeled_ratio 0.1 \
        --known_cls_ratio 0.75 \
        --seed 0 \
        --k $k \
        --g_k 30 \
        --alpha 0.01 \
        --tau 5 \
        --r $r \
        --num_train_epochs 15 \
        --lr 5e-6 \
        --save_model
  done
done
