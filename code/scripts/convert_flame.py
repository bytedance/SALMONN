"""Convert a locally downloaded FLAME 2020 pickle into the metric tensor asset."""
import argparse
import inspect
import pickle
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='FLAME 2020 generic_model.pkl')
    parser.add_argument('--output', required=True, help='Destination flame_2020.pt')
    args = parser.parse_args()
    # Chumpy 0.70 pickles require aliases removed in Python 3.11 / NumPy 1.24.
    inspect.getargspec = inspect.getfullargspec
    for name, value in dict(bool=bool, int=int, float=float, complex=complex,
                            object=object, str=str, unicode=str).items():
        if name not in np.__dict__:
            setattr(np, name, value)
    with open(args.input, 'rb') as handle:
        model = pickle.load(handle, encoding='latin1')
    tensors = {}
    for name in ('f', 'v_template', 'shapedirs', 'posedirs', 'J_regressor', 'kintree_table', 'weights'):
        value = model[name]
        if hasattr(value, 'r'):
            value = value.r
        if hasattr(value, 'toarray'):
            value = value.toarray()
        tensors[name] = torch.tensor(np.asarray(value), dtype=torch.long
                                    if name in ('f', 'kintree_table') else torch.float32)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensors, output)


if __name__ == '__main__':
    main()
