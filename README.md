# Towards Empirical Interpretation of Internal Circuits and Properties in Grokked Transformers on Modular Polynomials

Published in **Transactions on Machine Learning Research (TMLR)** [[arxiv](https://arxiv.org/abs/2402.16726)] [[OpenReview](https://openreview.net/forum?id=MzSf70uXJO)]

## Config
Example:
- `config/config_add_transformer_emb.py` : Modualr Addition
- `config/config_2x_3y_short_transformer_emb.py` : Modualr Polynomials (2x+3y)

## Training Base Model
### Modular Arithmetic Pre-Training

Examples:
```bash
# four arithmetic operations
CUDA_VISIBLE_DEVICES=0 python train_emb.py --config config.config_add_transformer_emb --seed 0
CUDA_VISIBLE_DEVICES=0 python train_emb.py --config config.config_subtract_transformer_emb --seed 0
CUDA_VISIBLE_DEVICES=0 python train_emb.py --config config.config_multiply_transformer_emb --seed 0

# polynomial
CUDA_VISIBLE_DEVICES=0 python train_emb.py --config config.config_x2_y2_short_transformer_emb --seed 0 -f 0.3
CUDA_VISIBLE_DEVICES=0 python train_emb.py --config config.config_x2_xy_y2_short_transformer_emb --seed 0 -f 0.3
CUDA_VISIBLE_DEVICES=0 python train_emb.py --config config.config_x_x_y_short_transformer_emb --seed 0 -f 0.3
CUDA_VISIBLE_DEVICES=0 python train_emb.py --config config.config_xy_y_short_transformer_emb --seed 0 -f 0.3
CUDA_VISIBLE_DEVICES=0 python train_emb.py --config config.config_x2_xy_y2_x_short_transformer_emb --seed 0 -f 0.3
CUDA_VISIBLE_DEVICES=0 python train_emb.py --config config.config_x3_xy_short_transformer_emb --seed 0 -f 0.3
CUDA_VISIBLE_DEVICES=0 python train_emb.py --config config.config_x3_xy2_y_short_transformer_emb --seed 0 -f 0.3

# multi-task mixture
CUDA_VISIBLE_DEVICES=0 python train_emb.py --config config.config_add_mul_transformer_emb --seed 0
CUDA_VISIBLE_DEVICES=0 python train_emb.py --config config.config_add_sub_transformer_emb --seed 0
CUDA_VISIBLE_DEVICES=0 python train_emb.py --config config.config_add_sub_mul_transformer_emb --seed 0
```

### Modular Arithmetic Finetuning
You need to register the path to your own pre-grokked weights with `ckpt_path_dict` in your config (e.g. `config/config_add_transformer_emb_ft_emb.py`)

Examples:
```bash
# four arithmetic operations
## Pre-Grokked Embedding
CUDA_VISIBLE_DEVICES=0 python train_emb_ft.py --config config.config_add_transformer_emb_ft_emb --seed 0 -f 0.3 --ckpt_name pt_add_p97_03
CUDA_VISIBLE_DEVICES=0 python train_emb_ft.py --config config.config_subtract_transformer_emb_ft_emb --seed 0 -f 0.3 --ckpt_name pt_sub_p97_03
CUDA_VISIBLE_DEVICES=0 python train_emb_ft.py --config config.config_multiply_transformer_emb_ft_emb --seed 0 -f 0.3 --ckpt_name pt_mul_p97_03

## Pre-Grokked Models
CUDA_VISIBLE_DEVICES=0 python train_emb_ft.py --config config.config_add_transformer_emb_ft_model --seed 0 -f 0.3 --ckpt_name pt_add_p97_03
CUDA_VISIBLE_DEVICES=0 python train_emb_ft.py --config config.config_subtract_transformer_emb_ft_model --seed 0 -f 0.3 --ckpt_name pt_sub_p97_03
CUDA_VISIBLE_DEVICES=0 python train_emb_ft.py --config config.config_multiply_transformer_emb_ft_model --seed 0 -f 0.3 --ckpt_name pt_mul_p97_03

# Polynomials
## Pre-Grokked Embedding
CUDA_VISIBLE_DEVICES=0 python train_emb_ft.py --config config.config_x2_y2_short_transformer_emb_ft_emb --seed 0 -f 0.3 --ckpt_name pt_add_p97_03
CUDA_VISIBLE_DEVICES=0 python train_emb_ft.py --config config.config_x2_xy_y2_short_transformer_emb_ft_emb --seed 0 -f 0.3 --ckpt_name pt_add_p97_03
CUDA_VISIBLE_DEVICES=0 python train_emb_ft.py --config config.config_x_x_y_short_transformer_emb_ft_emb --seed 0 -f 0.3 --ckpt_name pt_add_p97_03
CUDA_VISIBLE_DEVICES=0 python train_emb_ft.py --config config.config_xy_y_short_transformer_emb_ft_emb --seed 0 -f 0.3 --ckpt_name pt_add_p97_03
CUDA_VISIBLE_DEVICES=0 python train_emb_ft.py --config config.config_x2_xy_y2_x_short_transformer_emb_ft_emb --seed 0 -f 0.3 --ckpt_name pt_add_p97_03
CUDA_VISIBLE_DEVICES=0 python train_emb_ft.py --config config.config_x3_xy_short_transformer_emb_ft_emb --seed 0 -f 0.3 --ckpt_name pt_add_p97_03
CUDA_VISIBLE_DEVICES=0 python train_emb_ft.py --config config.config_x3_xy2_y_short_transformer_emb_ft_emb --seed 0 -f 0.3 --ckpt_name pt_add_p97_03

## Pre-Grokked Models
CUDA_VISIBLE_DEVICES=0 python train_emb_ft.py --config config.config_x2_y2_short_transformer_emb_ft_model --seed 0 -f 0.3 --ckpt_name pt_add_p97_03
CUDA_VISIBLE_DEVICES=0 python train_emb_ft.py --config config.config_x2_xy_y2_short_transformer_emb_ft_model --seed 0 -f 0.3 --ckpt_name pt_add_p97_03
CUDA_VISIBLE_DEVICES=0 python train_emb_ft.py --config config.config_x_x_y_short_transformer_emb_ft_model --seed 0 -f 0.3 --ckpt_name pt_add_p97_03
CUDA_VISIBLE_DEVICES=0 python train_emb_ft.py --config config.config_xy_y_short_transformer_emb_ft_model --seed 0 -f 0.3 --ckpt_name pt_add_p97_03
CUDA_VISIBLE_DEVICES=0 python train_emb_ft.py --config config.config_x2_xy_y2_x_short_transformer_emb_ft_model --seed 0 -f 0.3 --ckpt_name pt_add_p97_03
CUDA_VISIBLE_DEVICES=0 python train_emb_ft.py --config config.config_x3_xy_short_transformer_emb_ft_model --seed 0 -f 0.3 --ckpt_name pt_add_p97_03
CUDA_VISIBLE_DEVICES=0 python train_emb_ft.py --config config.config_x3_xy2_y_short_transformer_emb_ft_model --seed 0 -f 0.3 --ckpt_name pt_add_p97_03
```

## Results
You can check the experimental results from wandb.

## Reference
- https://github.com/neelnanda-io/Grokking
- https://github.com/kindxiaoming/omnigrok
- https://github.com/openai/grok

## Citation
```
@article{furuta2024interpreting,
    title={Towards Empirical Interpretation of Internal Circuits and Properties in Grokked Transformers on Modular Polynomials},
    author={Hiroki Furuta and Gouki Minegishi and Yusuke Iwasawa and Yutaka Matsuo},
    year={2024},
    journal = {arXiv preprint arXiv:2402.16726}
}
```
