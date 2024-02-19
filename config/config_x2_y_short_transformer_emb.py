import os
from pathlib import Path

import numpy as np


class Exp(object):
    def __init__(self) -> None:
        # Modular Arithmetic
        # self.p = 113
        self.p = 97
        # Learning Parameter
        self.lr = 1e-3 
        self.weight_decay = 1.0
        self.seed = 0
        self.batch_style = 'full'  # ['full', 'random']
        # self.lp = 1
        self.lp = 2
        self.lp_alpha = 0.0

        # model
        self.model = 'transformer'  # ['mlp', 'transformer']
        self.d_emb = 500
        self.d_model = 128
        self.num_layers = 1
        self.use_ln = False
        self.d_vocab = self.p + 5  # 5 special tokens (=,+,-,*,/)
        self.n_ctx = 4 - 1
        self.d_mlp = 4 * self.d_model
        self.num_heads = 4
        assert self.d_model % self.num_heads == 0
        self.d_head = self.d_model // self.num_heads
        self.act_type = 'ReLU'  # ['ReLU', 'GELU']
        self.weight_scale = 1

        # training 
        self.num_epochs = 100000
        self.save_models = True 
        self.save_every = 2000
        self.stopping_thresh = -1  # Stop training when test loss is < stopping_thresh

        # SLTH
        self.prune_rate = 0.4

        # dataset
        self.fn_name = 'x2_y_short'
        self.frac_train = 0.3
        self.is_symmetric_input = False
        self.is_div = True if "only" in self.fn_name else False
        self.n_mask = 0
        self.multivariate = 2
        # functions
        self.random_answers = np.random.randint(low=0, high=self.p, size=(self.p, self.p))
        self.fns_dict = {
            'add': lambda x, y: (x + y) % self.p,
            'subtract': lambda x, y: (x - y) % self.p,
            'multiply': lambda x, y: (x * y) % self.p,
            'x2_xy_y2': lambda x, y: (x**2 + x * y + y**2) % self.p,
            'x2_y2': lambda x, y: (x**2 + y**2) % self.p,
            'x2_xy_y2_x': lambda x, y: (x**2 + x * y + y**2 + x) % self.p,
            'x3_xy': lambda x, y: (x**3 + x * y) % self.p,
            'x3_xy2_y': lambda x, y: (x**3 + x * y**2 + y) % self.p,
            'add_z': lambda x, y, z: (x + y + z) % self.p,
            'subtract_z': lambda x, y, z: (x - y - z) % self.p,
            'multiply_z': lambda x, y, z: (x * y * z) % self.p,
            'x2_y2_z': lambda x, y, z: (x**2 + y**2 + z) % self.p,
            'x2_y2_z2': lambda x, y, z: (x**2 + y**2 + z**2) % self.p,
            'x_x_y': lambda x, y: (x + x + y) % self.p,
            'xy_y': lambda x, y: (x * y + y) % self.p,
            'x2_y': lambda x, y: (x * x + y) % self.p,
            '2x_3y': lambda x, y: (2 * x + 3 * y) % self.p,
            'x3_2y': lambda x, y: (x**3 + 2 * y) % self.p,
            'x2_2xy_y2': lambda x, y: (x**2 + 2 * x * y + y**2) % self.p,
            'x2_2xy_y2_x_y': lambda x, y: (x**2 + 2 * x * y + y**2 + x + y) % self.p,
            'x_x_m_y': lambda x, y: (2 * x - y) % self.p,
            'xy_m_y': lambda x, y: (x * y - y) % self.p,
            'x2_m_y': lambda x, y: (x * x - y) % self.p,
            '2x_m_3y': lambda x, y: (2 * x - 3 * y) % self.p,
            'x3_m_2y': lambda x, y: (x**3 - 2 * y) % self.p,
            'x2_m_y2': lambda x, y: (x**2 - y**2) % self.p,
            'x2_m_2xy_y2': lambda x, y: (x**2 - 2 * x * y + y**2) % self.p,
            'rand':lambda x, y:self.random_answers[x][y],
            'only_add': lambda x, y: (x + y),
        }
        self.fn = self.fns_dict[self.fn_name.replace('_short', '')]

        # save directory
        self.root = Path(f"log/{self.model}_{self.fn_name}_emb_231230")
        os.makedirs(self.root, exist_ok=True)
        self.exp_name = f"{self.model}_{self.num_layers}L_{self.d_model}D_{self.p}P_\
            {self.frac_train}F_{self.lr}LR_{self.weight_decay}WD_{self.is_symmetric_input}S_\
            {self.weight_scale}WS_{self.fn_name}task_{self.lp}LP_{self.lp_alpha}LPA_emb_seed_{self.seed}"
