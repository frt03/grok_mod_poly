import functools
import random

import torch


def gen_train_test(frac_train, num, seed=0, is_symmetric_input=False, division=False, non_zero=False, n=2):
    # Generate train and test split
    if is_symmetric_input and not division:
        pairs = [(i, j) for i in range(num) for j in range(num) if i <= j]
    elif is_symmetric_input and division:
        pairs = [(i, j) for i in range(num) for j in range(num) if (i <= j and j != 0)]
    elif not is_symmetric_input and not division:
        pairs = [(i, j) for i in range(num) for j in range(num)]
    elif n == 3:
        pairs = [(i, j, k) for i in range(num) for j in range(num) for k in range(num)]
    else:
        pairs = [(i, j) for i in range(num) for j in range(num) if (j != 0)]
    
    if non_zero:
        _pairs = []
        for pair in pairs:
            if (0 not in pair):
                _pairs.append(pair)
        pairs = _pairs

    random.seed(seed)
    random.shuffle(pairs)
    div = int(frac_train * len(pairs))
    return pairs[:div], pairs[div:]


def get_special_tokens(mod=113, n_mask=0):
    special_tokens = {
        '=': mod,
        '+': mod + 1,
        '-': mod + 2,
        '*': mod + 3,
        '/': mod + 4,
    }
    if n_mask > 0:
        for i in range(n_mask):
            special_tokens[f'<mask_{i}>'] = mod + 5 + i
    return special_tokens 


def addition(data, mod, n_mask, fn):
    """data: train/test."""
    st = get_special_tokens(mod=mod, n_mask=n_mask)
    equation = []
    for d in data:
        equation.append([d[0], st['+'], d[1], st['='], fn(d[0], d[1])])
    data = torch.tensor(equation)
    labels = data[:, -1]
    return data, labels, st


def subtract(data, mod, n_mask, fn):
    """data: train/test."""
    st = get_special_tokens(mod=mod, n_mask=n_mask)
    equation = []
    for d in data:
        equation.append([d[0], st['-'], d[1], st['='], fn(d[0], d[1])])
    data = torch.tensor(equation)
    labels = data[:, -1]
    return data, labels, st


def multiply(data, mod, n_mask, fn):
    """data: train/test."""
    st = get_special_tokens(mod=mod, n_mask=n_mask)
    equation = []
    for d in data:
        equation.append([d[0], st['*'], d[1], st['='], fn(d[0], d[1])])
    data = torch.tensor(equation)
    labels = data[:, -1]
    return data, labels, st


def division(data, mod, n_mask, fn):
    """data: train/test."""
    st = get_special_tokens(mod=mod, n_mask=n_mask)
    equation = []
    for d in data:
        equation.append([d[0], st['/'], d[1], st['='], fn(d[0], d[1])])
    data = torch.tensor(equation)
    labels = data[:, -1]
    return data, labels, st


def x2_xy_y2(data, mod, n_mask, fn, short=False):
    """data: train/test."""
    st = get_special_tokens(mod=mod, n_mask=n_mask)
    equation = []
    for d in data:
        if short:
            equation.append(
                [
                    d[0], st['='], d[1],
                    st['='], fn(d[0], d[1])
                ]
            )
        else:
            equation.append(
                [
                    d[0], st['*'], d[0],
                    st['+'],
                    d[0], st['*'], d[1],
                    st['+'],
                    d[1], st['*'], d[1],
                    st['='], fn(d[0], d[1])
                ]
            )
    data = torch.tensor(equation)
    labels = data[:, -1]
    return data, labels, st


def x2_y2(data, mod, n_mask, fn, short=False):
    """data: train/test."""
    st = get_special_tokens(mod=mod, n_mask=n_mask)
    equation = []
    for d in data:
        if short:
            equation.append(
                [
                    d[0], st['='], d[1],
                    st['='], fn(d[0], d[1])
                ]
            )
        else:
            equation.append(
                [
                    d[0], st['*'], d[0],
                    st['+'],
                    d[1], st['*'], d[1],
                    st['='], fn(d[0], d[1])
                ]
            )
    data = torch.tensor(equation)
    labels = data[:, -1]
    return data, labels, st


def x2_xy_y2_x(data, mod, n_mask, fn, short=False):
    """data: train/test."""
    st = get_special_tokens(mod=mod, n_mask=n_mask)
    equation = []
    for d in data:
        if short:
            equation.append(
                [
                    d[0], st['='], d[1],
                    st['='], fn(d[0], d[1])
                ]
            )
        else:
            equation.append(
                [
                    d[0], st['*'], d[0],
                    st['+'],
                    d[0], st['*'], d[1],
                    st['+'],
                    d[1], st['*'], d[1],
                    st['+'],
                    d[0],
                    st['='], fn(d[0], d[1])
                ]
            )
    data = torch.tensor(equation)
    labels = data[:, -1]
    return data, labels, st


def x3_xy(data, mod, n_mask, fn, short=False):
    """data: train/test."""
    st = get_special_tokens(mod=mod, n_mask=n_mask)
    equation = []
    for d in data:
        if short:
            equation.append(
                [
                    d[0], st['='], d[1],
                    st['='], fn(d[0], d[1])
                ]
            )
        else:
            equation.append(
                [
                    d[0], st['*'], d[0], st['*'], d[0],
                    st['+'],
                    d[0], st['*'], d[1],
                    st['='], fn(d[0], d[1])
                ]
            )
    data = torch.tensor(equation)
    labels = data[:, -1]
    return data, labels, st


def x3_xy2_y(data, mod, n_mask, fn, short=False):
    """data: train/test."""
    st = get_special_tokens(mod=mod, n_mask=n_mask)
    equation = []
    for d in data:
        if short:
            equation.append(
                [
                    d[0], st['='], d[1],
                    st['='], fn(d[0], d[1])
                ]
            )
        else:
            equation.append(
                [
                    d[0], st['*'], d[0], st['*'], d[0],
                    st['+'],
                    d[0], st['*'], d[1], st['*'], d[1],
                    st['+'],
                    d[1],
                    st['='], fn(d[0], d[1])
                ]
            )
    data = torch.tensor(equation)
    labels = data[:, -1]
    return data, labels, st


def addition_z(data, mod, n_mask, fn, short=False):
    """data: train/test."""
    st = get_special_tokens(mod=mod, n_mask=n_mask)
    # st = get_special_tokens(mod=mod, n_mask=n_mask+1)        
    equation = []
    for d in data:
        if short:
            equation.append(
                [
                    # d[0], st['='], d[1], st['<mask_0>'], d[2],
                    d[0], st['='], d[1], st['='], d[2],
                    st['='], fn(d[0], d[1], d[2])
                ]
            )
        else:
            equation.append(
                [
                    d[0], st['+'], d[1], st['+'], d[2],
                    st['='], fn(d[0], d[1], d[2])
                ]
            )
    data = torch.tensor(equation)
    labels = data[:, -1]
    return data, labels, st


def subtract_z(data, mod, n_mask, fn, short=False):
    """data: train/test."""
    st = get_special_tokens(mod=mod, n_mask=n_mask)
    equation = []
    for d in data:
        if short:
            equation.append(
                [
                    d[0], st['='], d[1], st['='], d[2],
                    st['='], fn(d[0], d[1], d[2])
                ]
            )
        else:
            equation.append(
                [
                    d[0], st['-'], d[1], st['-'], d[2],
                    st['='], fn(d[0], d[1], d[2])
                ]
            )
    data = torch.tensor(equation)
    labels = data[:, -1]
    return data, labels, st


def multiply_z(data, mod, n_mask, fn, short=False):
    """data: train/test."""
    st = get_special_tokens(mod=mod, n_mask=n_mask)
    equation = []
    for d in data:
        if short:
            equation.append(
                [
                    d[0], st['='], d[1], st['='], d[2],
                    st['='], fn(d[0], d[1], d[2])
                ]
            )
        else:
            equation.append(
                [
                    d[0], st['*'], d[1], st['*'], d[2],
                    st['='], fn(d[0], d[1], d[2])
                ]
            )
    data = torch.tensor(equation)
    labels = data[:, -1]
    return data, labels, st


def x2_y2_z(data, mod, n_mask, fn, short=False):
    """data: train/test."""
    st = get_special_tokens(mod=mod, n_mask=n_mask)
    equation = []
    for d in data:
        if short:
            equation.append(
                [
                    d[0], st['='], d[1], st['='], d[2],
                    st['='], fn(d[0], d[1], d[2])
                ]
            )
        else:
            equation.append(
                [
                    d[0], st['*'], d[0],
                    st['+'],
                    d[1], st['*'], d[1],
                    st['+'],
                    d[2],
                    st['='], fn(d[0], d[1], d[2])
                ]
            )
    data = torch.tensor(equation)
    labels = data[:, -1]
    return data, labels, st


def x2_y2_z2(data, mod, n_mask, fn, short=False):
    """data: train/test."""
    st = get_special_tokens(mod=mod, n_mask=n_mask)
    equation = []
    for d in data:
        if short:
            equation.append(
                [
                    d[0], st['='], d[1], st['='], d[2],
                    st['='], fn(d[0], d[1], d[2])
                ]
            )
        else:
            equation.append(
                [
                    d[0], st['*'], d[0],
                    st['+'],
                    d[1], st['*'], d[1],
                    st['+'],
                    d[2], st['*'], d[2],
                    st['='], fn(d[0], d[1], d[2])
                ]
            )
    data = torch.tensor(equation)
    labels = data[:, -1]
    return data, labels, st


def x_x_y(data, mod, n_mask, fn, short=False):
    """data: train/test."""
    st = get_special_tokens(mod=mod, n_mask=n_mask)
    equation = []
    for d in data:
        if short:
            equation.append(
                [
                    d[0], st['='], d[1],
                    st['='], fn(d[0], d[1])
                ]
            )
        else:
            equation.append(
                [
                    d[0], st['+'], d[0], st['+'], d[1],
                    st['='], fn(d[0], d[1])
                ]
            )
    data = torch.tensor(equation)
    labels = data[:, -1]
    return data, labels, st


def xy_y(data, mod, n_mask, fn, short=False):
    """data: train/test."""
    st = get_special_tokens(mod=mod, n_mask=n_mask)
    equation = []
    for d in data:
        if short:
            equation.append(
                [
                    d[0], st['='], d[1],
                    st['='], fn(d[0], d[1])
                ]
            )
        else:
            equation.append(
                [
                    d[0], st['*'], d[1], st['+'], d[1],
                    st['='], fn(d[0], d[1])
                ]
            )
    data = torch.tensor(equation)
    labels = data[:, -1]
    return data, labels, st


def x2_y(data, mod, n_mask, fn, short=False):
    """data: train/test."""
    st = get_special_tokens(mod=mod, n_mask=n_mask)
    equation = []
    for d in data:
        if short:
            equation.append(
                [
                    d[0], st['='], d[1],
                    st['='], fn(d[0], d[1])
                ]
            )
        else:
            equation.append(
                [
                    d[0], st['*'], d[0], st['+'], d[1],
                    st['='], fn(d[0], d[1])
                ]
            )
    data = torch.tensor(equation)
    labels = data[:, -1]
    return data, labels, st


def _2x_3y(data, mod, n_mask, fn, short=False):
    """data: train/test."""
    st = get_special_tokens(mod=mod, n_mask=n_mask)
    equation = []
    for d in data:
        if short:
            equation.append(
                [
                    d[0], st['='], d[1],
                    st['='], fn(d[0], d[1])
                ]
            )
        else:
            equation.append(
                [
                    d[0], st['+'], d[0],
                    st['+'],
                    d[1], st['+'], d[1], st['+'], d[1],
                    st['='], fn(d[0], d[1])
                ]
            )
    data = torch.tensor(equation)
    labels = data[:, -1]
    return data, labels, st


def x3_2y(data, mod, n_mask, fn, short=False):
    """data: train/test."""
    st = get_special_tokens(mod=mod, n_mask=n_mask)
    equation = []
    for d in data:
        if short:
            equation.append(
                [
                    d[0], st['='], d[1],
                    st['='], fn(d[0], d[1])
                ]
            )
        else:
            equation.append(
                [
                    d[0], st['*'], d[0], st['*'], d[0],
                    st['+'], d[1], st['+'], d[1],
                    st['='], fn(d[0], d[1])
                ]
            )
    data = torch.tensor(equation)
    labels = data[:, -1]
    return data, labels, st


def x2_2xy_y2(data, mod, n_mask, fn, short=False):
    """data: train/test."""
    st = get_special_tokens(mod=mod, n_mask=n_mask)
    equation = []
    for d in data:
        if short:
            equation.append(
                [
                    d[0], st['='], d[1],
                    st['='], fn(d[0], d[1])
                ]
            )
        else:
            equation.append(
                [
                    d[0], st['*'], d[0],
                    st['+'],
                    d[0], st['*'], d[1],
                    st['+'],
                    d[0], st['*'], d[1],
                    st['+'],
                    d[1], st['*'], d[1],
                    st['='], fn(d[0], d[1])
                ]
            )
    data = torch.tensor(equation)
    labels = data[:, -1]
    return data, labels, st


def x2_2xy_y2_x_y(data, mod, n_mask, fn, short=False):
    """data: train/test."""
    st = get_special_tokens(mod=mod, n_mask=n_mask)
    equation = []
    for d in data:
        if short:
            equation.append(
                [
                    d[0], st['='], d[1],
                    st['='], fn(d[0], d[1])
                ]
            )
        else:
            equation.append(
                [
                    d[0], st['*'], d[0],
                    st['+'],
                    d[0], st['*'], d[1],
                    st['+'],
                    d[0], st['*'], d[1],
                    st['+'],
                    d[1], st['*'], d[1],
                    t['+'],
                    d[0], st['+'], d[1],
                    st['='], fn(d[0], d[1])
                ]
            )
    data = torch.tensor(equation)
    labels = data[:, -1]
    return data, labels, st


def x_x_m_y(data, mod, n_mask, fn, short=False):
    """data: train/test."""
    st = get_special_tokens(mod=mod, n_mask=n_mask)
    equation = []
    for d in data:
        if short:
            equation.append(
                [
                    d[0], st['='], d[1],
                    st['='], fn(d[0], d[1])
                ]
            )
        else:
            equation.append(
                [
                    d[0], st['+'], d[0], st['-'], d[1],
                    st['='], fn(d[0], d[1])
                ]
            )
    data = torch.tensor(equation)
    labels = data[:, -1]
    return data, labels, st


def xy_m_y(data, mod, n_mask, fn, short=False):
    """data: train/test."""
    st = get_special_tokens(mod=mod, n_mask=n_mask)
    equation = []
    for d in data:
        if short:
            equation.append(
                [
                    d[0], st['='], d[1],
                    st['='], fn(d[0], d[1])
                ]
            )
        else:
            equation.append(
                [
                    d[0], st['*'], d[1], st['-'], d[1],
                    st['='], fn(d[0], d[1])
                ]
            )
    data = torch.tensor(equation)
    labels = data[:, -1]
    return data, labels, st


def x2_m_y(data, mod, n_mask, fn, short=False):
    """data: train/test."""
    st = get_special_tokens(mod=mod, n_mask=n_mask)
    equation = []
    for d in data:
        if short:
            equation.append(
                [
                    d[0], st['='], d[1],
                    st['='], fn(d[0], d[1])
                ]
            )
        else:
            equation.append(
                [
                    d[0], st['*'], d[0], st['-'], d[1],
                    st['='], fn(d[0], d[1])
                ]
            )
    data = torch.tensor(equation)
    labels = data[:, -1]
    return data, labels, st


def _2x_m_3y(data, mod, n_mask, fn, short=False):
    """data: train/test."""
    st = get_special_tokens(mod=mod, n_mask=n_mask)
    equation = []
    for d in data:
        if short:
            equation.append(
                [
                    d[0], st['='], d[1],
                    st['='], fn(d[0], d[1])
                ]
            )
        else:
            equation.append(
                [
                    d[0], st['+'], d[0],
                    st['-'],
                    d[1], st['-'], d[1], st['-'], d[1],
                    st['='], fn(d[0], d[1])
                ]
            )
    data = torch.tensor(equation)
    labels = data[:, -1]
    return data, labels, st


def x3_m_2y(data, mod, n_mask, fn, short=False):
    """data: train/test."""
    st = get_special_tokens(mod=mod, n_mask=n_mask)
    equation = []
    for d in data:
        if short:
            equation.append(
                [
                    d[0], st['='], d[1],
                    st['='], fn(d[0], d[1])
                ]
            )
        else:
            equation.append(
                [
                    d[0], st['*'], d[0], st['*'], d[0],
                    st['-'], d[1], st['-'], d[1],
                    st['='], fn(d[0], d[1])
                ]
            )
    data = torch.tensor(equation)
    labels = data[:, -1]
    return data, labels, st


def x2_m_2xy_y2(data, mod, n_mask, fn, short=False):
    """data: train/test."""
    st = get_special_tokens(mod=mod, n_mask=n_mask)
    equation = []
    for d in data:
        if short:
            equation.append(
                [
                    d[0], st['='], d[1],
                    st['='], fn(d[0], d[1])
                ]
            )
        else:
            equation.append(
                [
                    d[0], st['*'], d[0],
                    st['-'],
                    d[0], st['*'], d[1],
                    st['-'],
                    d[0], st['*'], d[1],
                    st['+'],
                    d[1], st['*'], d[1],
                    st['='], fn(d[0], d[1])
                ]
            )
    data = torch.tensor(equation)
    labels = data[:, -1]
    return data, labels, st


def x2_m_y2(data, mod, n_mask, fn, short=False):
    """data: train/test."""
    st = get_special_tokens(mod=mod, n_mask=n_mask)
    equation = []
    for d in data:
        if short:
            equation.append(
                [
                    d[0], st['='], d[1],
                    st['='], fn(d[0], d[1])
                ]
            )
        else:
            equation.append(
                [
                    d[0], st['*'], d[0],
                    st['-'],
                    d[1], st['*'], d[1],
                    st['='], fn(d[0], d[1])
                ]
            )
    data = torch.tensor(equation)
    labels = data[:, -1]
    return data, labels, st


def polinomials(data, mod, n_mask, fn, short=False):
    """data: train/test."""
    st = get_special_tokens(mod=mod, n_mask=n_mask)
    equation = []
    for d in data:
        if short:
            equation.append(
                [
                    d[0], st['='], d[1],
                    st['='], fn(d[0], d[1])
                ]
            )
        else:
            raise Exception
    data = torch.tensor(equation)
    labels = data[:, -1]
    return data, labels, st


MODULAR_ARITHMETIC_DATASET_CONFIG = {
    'add': addition,
    'subtract': subtract,
    'multiply': multiply,
    'division': division,
    'x2_xy_y2': x2_xy_y2,
    'x2_y2': x2_y2,
    'x2_xy_y2_x': x2_xy_y2_x,
    'x3_xy': x3_xy,
    'x3_xy2_y': x3_xy2_y,
    'add_z': addition_z,
    'subtract_z': subtract_z,
    'multiply_z': multiply_z,
    'x2_y2_z': x2_y2_z,
    'x2_y2_z2': x2_y2_z2,
    'x_x_y': x_x_y,
    'xy_y': xy_y,
    'x2_y': x2_y,
    '2x_3y': _2x_3y,
    'x3_2y': x3_2y,
    'x2_2xy_y2': x2_2xy_y2,
    'x2_2xy_y2_x_y': x2_2xy_y2_x_y,
    'x_x_m_y': x_x_m_y,
    'xy_m_y': xy_m_y,
    'x2_m_y': x2_m_y,
    '2x_m_3y': _2x_m_3y,
    'x3_m_2y': x3_m_2y,
    'x2_m_y2': x2_m_y2,
    'x2_m_2xy_y2': x2_m_2xy_y2,
    'multiply_non_zero': multiply,
    'x2_xy_y2_short': functools.partial(x2_xy_y2, short=True),
    'x2_y2_short': functools.partial(x2_y2, short=True),
    'x2_xy_y2_x_short': functools.partial(x2_xy_y2_x, short=True),
    'x3_xy_short': functools.partial(x3_xy, short=True),
    'x3_xy2_y_short': functools.partial(x3_xy2_y, short=True),
    'add_z_short': functools.partial(addition_z, short=True),
    'subtract_z_short': functools.partial(subtract_z, short=True),
    'multiply_z_short': functools.partial(multiply_z, short=True),
    'x2_y2_z_short': functools.partial(x2_y2_z, short=True),
    'x2_y2_z2_short': functools.partial(x2_y2_z2, short=True),
    'x_x_y_short': functools.partial(x_x_y, short=True),
    'xy_y_short': functools.partial(xy_y, short=True),
    'x2_y_short': functools.partial(x2_y, short=True),
    '2x_3y_short': functools.partial(_2x_3y, short=True),
    'x3_2y_short': functools.partial(x3_2y, short=True),
    'x2_2xy_y2_short': functools.partial(x2_2xy_y2, short=True),
    'x2_2xy_y2_x_y_short': functools.partial(x2_2xy_y2_x_y, short=True),
    '_x_y__3_short': functools.partial(polinomials, short=True),
    '_x_y__4_short': functools.partial(polinomials, short=True),
    'xy_x_y_short': functools.partial(polinomials, short=True),
    'x_x_m_y_short': functools.partial(x_x_m_y, short=True),
    'xy_m_y_short': functools.partial(xy_m_y, short=True),
    'x2_m_y_short': functools.partial(x2_m_y, short=True),
    '2x_m_3y_short': functools.partial(_2x_m_3y, short=True),
    'x3_m_2y_short': functools.partial(x3_m_2y, short=True),
    'x2_m_y2_short': functools.partial(x2_m_y2, short=True),
    'x2_m_2xy_y2_short': functools.partial(x2_m_2xy_y2, short=True),
    '_x_m_y__3_short': functools.partial(polinomials, short=True),
    '_x_m_y__4_short': functools.partial(polinomials, short=True),
    'xy_x_m_y_short': functools.partial(polinomials, short=True),
}
