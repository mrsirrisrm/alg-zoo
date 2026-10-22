"""Zoo 2nd-argmax models over plain HTTPS, cached in zoo_cache/.

alg_zoo's loader reads gs:// through blobfile, which hangs in this environment; the same
files are public at storage.googleapis.com (fetched with curl: this Python build has no CA bundle). Returns weights in the weights.npz layout.
"""
import os, subprocess
import numpy as np, torch as th
from alg_zoo.zoo import best_of_5_seed_2nd_argmax

URL = 'https://storage.googleapis.com/arc-ml-public/alg/zoo'
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'zoo_cache')


def zoo_weights(hidden_size, seq_len=10, *, n_seqs=2 ** 30, seed=None):
    if seed is None:
        seed = best_of_5_seed_2nd_argmax(hidden_size=hidden_size, seq_len=seq_len)
    name = f'2nd_argmax_{hidden_size}_{seq_len}_{n_seqs}_{seed}.pth'
    path = os.path.join(DIR, name)
    if not os.path.exists(path):
        os.makedirs(DIR, exist_ok=True)
        subprocess.run(['curl', '-sfL', '-o', path + '.part', f'{URL}/{name}'], check=True); os.replace(path + '.part', path)
    sd = th.load(path, weights_only=True, map_location='cpu')
    return [sd['rnn.weight_ih_l0'][:, 0].double().numpy(), sd['rnn.weight_hh_l0'].double().numpy(),
            sd['linear.weight'].double().numpy()]
