#!/bin/bash

models=("inception_v3" "resnet50" "vgg16" "maxvit_t")
attr_methods=("fast_ig" "deeplift" "guided_ig" "ig" "sg" "big" "sm" "mfaba" "eg" "agi" "attexplore" "la")

spatial_ranges=(10 20 30)
max_iters=(10 20 30)
sampling_times=(10 20 30)

for model in "${models[@]}"; do
  for method in "${attr_methods[@]}"; do

    if [ "$method" = "la" ]; then
      for spatial in "${spatial_ranges[@]}"; do
        for iter in "${max_iters[@]}"; do
          for sample in "${sampling_times[@]}"; do

            echo "Running $model with $method (spatial=$spatial, iter=$iter, sample=$sample)"

            CUDA_VISIBLE_DEVICES=0 python generate_attributions.py \
              --model $model \
              --attr_method $method \
              --spatial_range $spatial \
              --max_iter $iter \
              --sampling_times $sample \
              --prefix attributions

            CUDA_VISIBLE_DEVICES=0 python eval.py \
              --model $model \
              --attr_method $method \
              --attr_method $method \
              --spatial_range $spatial \
              --max_iter $iter \
              --prefix scores \
              --csv_path results_la.csv \
              --attr_prefix attributions

          done
        done
      done

    else
      echo "Running $model with $method (no extra params)"
      
      CUDA_VISIBLE_DEVICES=0 python generate_attributions.py \
        --model $model \
        --attr_method $method \
        --prefix attributions

      CUDA_VISIBLE_DEVICES=0 python eval.py \
        --model $model \
        --attr_method $method \
        --prefix scores \
        --csv_path results_baseline.csv \
        --attr_prefix attributions

    fi

  done
done
