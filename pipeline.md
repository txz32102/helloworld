We can run the full pipeline as follows:

```bash
python3 scripts/pipeline_benchmark.py
```

You can view the config file `configs/config.yaml` to config the models and paths.

We can enable both single stage and multi stage for the generation pipeline.

Note that for the tools configs, we may use vllm to call the tools which is much faster (though this is optional, but highly recommended)

```bash
CUDA_VISIBLE_DEVICES=4 vllm serve "/home/data1/musong/.cache/huggingface/hub/models--google--medgemma-1.5-4b-it/snapshots/e9792da5fb8ee651083d345ec4bce07c3c9f1641" \
  --gpu-memory-utilization 0.8 \
  --dtype bfloat16 \
  --port 8009
```