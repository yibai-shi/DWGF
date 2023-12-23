export CUDA_VISIBLE_DEVICES=0
for k in 15
do
  for r in 2
  do
    python dwgf.py \
        --dataset banking \
        --labeled_ratio 0.1 \
        --known_cls_ratio 0.75 \
        --seed 0 \
        --k $k \
        --g_k 25 \
        --alpha 0.3 \
        --tau 5 \
        --r $r \
        --num_train_epochs 50 \
        --save_model
  done
done
