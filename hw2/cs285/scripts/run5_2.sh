batch_sizes=(10000 5000 1000 500)
learning_rates=(1e-3 2e-3 5e-3 8e-3)

for bs in ${batch_sizes[*]}
do
  for lr in ${learning_rates[*]}
  do
    python cs285/scripts/run_hw2.py \
      --env_name InvertedPendulum-v4 \
      --exp_name q2_b${bs}_r${lr} \
      --ep_len 1000 \
      --discount 0.9 \
      -n 100 \
      -l 2 \
      -s 64 \
      -b $bs \
      -lr $lr \
      -rtg
  done
done