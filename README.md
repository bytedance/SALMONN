This is the PyTorch implementation of OmniMem

Model file:
qwenvl/model/modeling_qwen3_vl.py

To run test with the best setting in the paper (lambda=0.02, T=0.2)

```
bash test.sh \
    --interval 1 \
    --run_name your_own_run_name \
    --dataset path_to_dataset \
    --max_frames 100000 \
    --model path_to_vs2_checkpoint \
    --model_base path_to_model_base \
    --model_type dense \
    --fixed_memory_size 20480 \
    --workingmemsize 20480 \
    --memgroupsize 16384 \
    --model_type dense \
    --search_type sim_modmask \
    --num_worker 5 \
    --total_timestamps 256 \
    --shrink_config scripts/per_layer_budget_norm_audiovisual_temperature_0.2_audiolarge_8k_avratio5.json \
    --div_factor 0.02
```