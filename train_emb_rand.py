import argparse
import importlib
import os
from pathlib import Path
from typing import OrderedDict
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torcheval.metrics.functional import multiclass_accuracy
from tqdm import tqdm
import wandb

from model import MLP
from model import Transformer
import dataset
from utils import get_weight_norm
from utils import cross_entropy_high_precision
from utils import full_loss_mlp
from utils import lp_reg
from utils import visualize_embedding
from utils import visualize_weight
from utils import visualize_weight_distribution

warnings.filterwarnings('ignore')

transformer_keys = [
    'blocks.0.attn.W_K',
    'blocks.0.attn.W_Q',
    'blocks.0.attn.W_V',
    'blocks.0.attn.W_O',
    'blocks.0.attn.mask',
    # 'blocks.0.attn.weight_maskK',
    # 'blocks.0.attn.weight_maskQ',
    # 'blocks.0.attn.weight_maskV',
    # 'blocks.0.attn.weight_maskO',
    'blocks.0.mlp.W_in',
    'blocks.0.mlp.W_out',
    # 'blocks.0.mlp.weight_mask_in',
    # 'blocks.0.mlp.weight_mask_out',
]
unembed_keys = [
    'unembed.W_U',
    # 'unembed.weight_mask',
]
embed_keys = [
    'embed.W_E',
    # 'embed.weight_mask',
]


def main(config):
    # project = 'ssl_grok_transfer'
    # project = 'poly_grok_explore'
    project = 'poly_grok_pretrain'
    wandb.init(project=project, name=config.exp_name, config=config)
    if config.model == 'transformer':
        model = Transformer(
            num_layers=config.num_layers,
            d_vocab=config.d_vocab,
            d_model=config.d_model,
            d_mlp=config.d_mlp,
            d_head=config.d_head,
            num_heads=config.num_heads,
            n_ctx=config.n_ctx,
            act_type=config.act_type,
            use_cache=False,
            use_ln=config.use_ln,
        )
        # freeze embedding randomly
        for name, param in model.named_parameters():
            if name in embed_keys:
                param.requires_grad = False
        print('='*20 + 'Trainable Parameters' + '='*20)
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)
    elif config.model == 'mlp':
        model = MLP(
            num_layers=config.num_layers,
            d_vocab=config.d_vocab,
            d_model=config.d_model,
            d_emb=config.d_emb,
            act_type=config.act_type,
            use_ln=config.use_ln,
            weight_scale=config.weight_scale,
        )
    model.to('cuda')
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.98),
    )
    run_name = f'{config.exp_name}'
    train, test = ssl_dataset.gen_train_test(
        config.frac_train,
        config.d_vocab,
        seed=config.seed,
        is_symmetric_input=config.is_symmetric_input,
        division=(config.fn_name=='division'),
        non_zero=('non_zero' in config.fn_name),
        n=config.multivariate,
    )
    fn_names = config.fn_name.split(',')
    dataset_fns = {fn_name: ssl_dataset.MODULAR_ARITHMETIC_DATASET_CONFIG[fn_name] for fn_name in fn_names}
    # dataset_fn = ssl_dataset.MODULAR_ARITHMETIC_DATASET_CONFIG[config.fn_name]

    if config.save_models:
        os.makedirs(
            os.path.join(config.root, run_name),
            exist_ok=True
        )
        save_dict = {
            'model': model.state_dict(),
            'train_data': train,
            'test_data': test,
        }
        torch.save(save_dict, os.path.join(config.root, run_name, 'init.pth'))
    train_metrics = {}
    test_metrics = {}
    with tqdm(range(config.num_epochs)) as pbar:
        pbar.set_description(f'{run_name}')
        for epoch in pbar:
            train_loss = 0
            test_loss = 0
            if config.model == 'transformer':
                # train
                for fn_name, dataset_fn in dataset_fns.items():
                    data, labels, _ = dataset_fn(
                        train,
                        mod=config.p,
                        n_mask=config.n_mask,
                        fn=config.fns_dict[fn_name.replace('_short', '')]
                    )
                    data, labels = data.to('cuda'), labels.to('cuda')
                    # logits = model(data[:,:-1])[:, -1]
                    logits = model(data[:,:-2])[:, -1]  # remove equal
                    prob = F.softmax(logits, dim=1)
                    train_acc = multiclass_accuracy(
                        input=logits,
                        target=labels,
                        num_classes=config.p + 5 + config.n_mask,
                        average='micro'
                    ).item()
                    _train_loss = cross_entropy_high_precision(logits, labels)
                    train_prob = torch.mean(
                        torch.gather(prob, index=labels[:, None], dim=-1)
                    )
                    _train_loss += config.lp_alpha * lp_reg(model, config.lp)
                    train_loss += _train_loss
                    train_metrics[fn_name] = {
                        'train_acc': train_acc,
                        'train_loss': _train_loss.item(),
                        'train_prob': train_prob.item()
                    }
                # test
                for fn_name, dataset_fn in dataset_fns.items():
                    data, labels, _ = dataset_fn(
                        test,
                        mod=config.p,
                        n_mask=config.n_mask,
                        fn=config.fns_dict[fn_name.replace('_short', '')]
                    )
                    data, labels = data.to('cuda'), labels.to('cuda')
                    # logits = model(data[:,:-1])[:, -1]
                    logits = model(data[:,:-2])[:, -1]  # remove equal
                    prob = F.softmax(logits, dim=1)
                    test_acc = multiclass_accuracy(
                        input=logits,
                        target=labels,
                        num_classes=config.p + 5 + config.n_mask,
                        average="micro"
                    ).item()
                    _test_loss = cross_entropy_high_precision(logits, labels)
                    test_prob = torch.mean(
                        torch.gather(prob, index=labels[:, None], dim=-1)
                    )
                    test_loss += _test_loss
                    test_metrics[fn_name] = {
                        'test_acc': test_acc,
                        'test_loss': _test_loss.item(),
                        'test_prob': test_prob.item()
                    }
            elif config.model == 'mlp':
                # train
                for fn_name, dataset_fn in dataset_fns.items():
                    data, labels, _ = dataset_fn(
                        train,
                        mod=config.p,
                        n_mask=config.n_mask,
                        fn=config.fns_dict[fn_name.replace('_short', '')]
                    )
                    data, labels = data.to('cuda'), labels.to('cuda')
                    # logits = model(data[:,:-1])
                    logits = model(data[:,:-2])  # remove equal
                    prob = F.softmax(logits, dim=1)
                    train_acc = multiclass_accuracy(
                        input=logits,
                        target=labels,
                        num_classes=config.p + 5 + config.n_mask,
                        average='micro'
                    ).item()
                    _train_loss = cross_entropy_high_precision(logits, labels)
                    train_prob = torch.mean(
                        torch.gather(prob, index=labels[:, None], dim=-1)
                    )
                    _train_loss += config.lp_alpha * lp_reg(model, config.lp)
                    train_loss += _train_loss
                    train_metrics[fn_name] = {
                        'train_acc': train_acc,
                        'train_loss': _train_loss.item(),
                        'train_prob': train_prob.item()
                    }
                # test
                for fn_name, dataset_fn in dataset_fns.items():
                    data, labels, _ = dataset_fn(
                        test,
                        mod=config.p,
                        n_mask=config.n_mask,
                        fn=config.fns_dict[fn_name.replace('_short', '')]
                    )
                    data, labels = data.to('cuda'), labels.to('cuda')
                    # logits = model(data[:,:-1])
                    logits = model(data[:,:-2])  # remove equal
                    prob = F.softmax(logits, dim=1)
                    test_acc = multiclass_accuracy(
                        input=logits,
                        target=labels,
                        num_classes=config.p + 5 + config.n_mask,
                        average="micro"
                    ).item()
                    _test_loss = cross_entropy_high_precision(logits, labels)
                    test_prob = torch.mean(
                        torch.gather(prob, index=labels[:, None], dim=-1)
                    )
                    test_loss += _test_loss
                    test_metrics[fn_name] = {
                        'test_acc': test_acc,
                        'test_loss': _test_loss.item(),
                        'test_prob': test_prob.item()
                    }
            train_acc = np.mean(
                [
                    train_metrics[fn_name]['train_acc'] for fn_name in fn_names
                ]
            )
            test_acc = np.mean(
                [
                    test_metrics[fn_name]['test_acc'] for fn_name in fn_names
                ]
            )
            train_prob = np.mean(
                [
                    train_metrics[fn_name]['train_prob'] for fn_name in fn_names
                ]
            )
            test_prob = np.mean(
                [
                    test_metrics[fn_name]['test_prob'] for fn_name in fn_names
                ]
            )
            pbar.set_postfix(
                OrderedDict(
                    Train_Loss=train_loss.item(),
                    Test_Loss=test_loss.item(),
                    Train_Acc=train_acc,
                    Test_Acc=test_acc,
                )
            )
            l1norm, l2norm, l1_dict, l2_dict = get_weight_norm(model)
            model_l1 = 0
            model_l2 = 0
            if config.model == 'transformer':
                for name in transformer_keys:
                    if name == 'blocks.0.attn.mask':
                        pass
                    else:
                        model_l1 += l1_dict[name]
                        model_l2 += l2_dict[name]
                model_l1 /= len(transformer_keys)
                model_l2 /= len(transformer_keys)
            log = {
                'epoch': epoch,
                'train_loss': train_loss,
                'test_loss': test_loss,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'train_prob': train_prob,
                'test_prob': test_prob,
                'l1norm': l1norm,
                'l2norm': l2norm,
                'l1norm_model': model_l1,
                'l2norm_model': model_l2,
            }
            for name in l1_dict.keys():
                log[f'l1norm_{name}'] = l1_dict[name]
                log[f'l2norm_{name}'] = l2_dict[name]
            if len(fn_names) > 1:
                for fn_name in fn_names:
                    for name in ('train_acc', 'train_prob', 'train_loss'):
                        log[f'{name}_{fn_name}'] = train_metrics[fn_name][name]
                    for name in ('test_acc', 'test_prob', 'test_loss'):
                        log[f'{name}_{fn_name}'] = test_metrics[fn_name][name]
            wandb.log(log)
            train_loss.backward()
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()
            if test_loss.item() < config.stopping_thresh:
                break
            if (config.save_models) and (epoch % config.save_every == 0):
                if config.model != 'transformer':
                    fig = visualize_weight_distribution(model)
                    wandb.log({'weight_distribution': fig})
                    plt.close()
                ims = visualize_weight(model)
                wandb.log({'weight': ims})
                plt.close()
                emb_img = visualize_embedding(model, p=config.p)
                wandb.log({'embedding': emb_img})
                plt.close()
                embedding_symbol = visualize_embedding(model, p=config.d_vocab)
                wandb.log({'embedding_symbol': embedding_symbol})
                plt.close()

                if test_loss.item() < config.stopping_thresh:
                    break
                save_dict = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    # 'scheduler': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                    'epoch': epoch,
                }
                torch.save(
                    save_dict,
                    os.path.join(config.root, run_name, f'{epoch}.pth')
                )
        if not config.save_models:
            os.mkdir(os.path.join(config.root, run_name))
        save_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            # 'scheduler': scheduler.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'epoch': epoch,
        }
        torch.save(
            save_dict,
            os.path.join(config.root, run_name, 'final.pth')
        )


if __name__ == '__main__':
    # from config.config_add_transformer_emb import Exp

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=0, help='seed')
    parser.add_argument('-f', '--frac_train', type=float, default=0.3, help='fraction of train dataset')
    parser.add_argument('--config', type=str, default='config.config_multiply_transformer_emb')
    args = parser.parse_args()

    module = importlib.import_module(args.config)

    config = module.Exp()
    config.seed = args.seed
    config.frac_train = args.frac_train
    config.exp_name = f'{config.fn_name}_{config.model}_{config.num_layers}L_{config.d_model}D_{config.p}P_{config.frac_train}F_{config.lr}LR_{config.weight_decay}WD_{config.is_symmetric_input}S_{config.weight_scale}WS_{config.lp}LP_{config.lp_alpha}LPA_emb_rand_rm_eq_seed_{config.seed}'

    main(config)
