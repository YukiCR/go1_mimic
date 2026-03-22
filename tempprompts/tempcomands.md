run box checkpoint
```bash
python scripts/robomimic/play.py --task ILBL-Go1-Mimic-Rough-v0 --num_rollouts 20 --horizon 100 --enable_cameras  --checkpoint logs/robomimic/ILBL-Go1-Mimic-Rough-v0/bc_rnn_go1_nav_latent_lader/20260316022213/models/model_epoch_520.pth
```

record demos
```bash
python scripts/tools/record_demos.py --task ILBL-Go1-Mimic-XXX-v0 --num_demos 200 --dataset_file XXXXX --enable_cameras --rendering_mode quality
```

train
```bash
python scripts/robomimic/train.py --task ILBL-Go1-Mimic-XXX-v0  --normalize_training_actions --algo bc --dataset XXXXX
```