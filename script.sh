# python3 scripts/train.py --max_epochs 10 --root_dir . --structured_prune --structured_prune_iteration 5 --state_dict_path ../EFFDL/lab/checkpoints/ckpt_sgd_0.1_0.9_0.9_0.999_0.0001
python3 scripts/train.py --max_epochs 200 --root_dir . --lr 0.01 --experiment_name custom_densenet
