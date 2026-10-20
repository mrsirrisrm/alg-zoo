"""Metrics for fine-tuned M(16,10) variants (PLAN G1): MC accuracy, hard per-neuron gate
switching within tie skeletons, region count at 99% mass (as region_count.py), and the
last-step kink statistics of decision_kinks.py. All take weights as a tuple of tensors."""
import numpy as np, torch as th
from scipy.stats import norm
from boundary_layer import sample_ties

T = 10


def pre_acts(x, W_ih, W_hh):
    """x [n,10] -> pre-activations [n,10,16] and final state [n,16]."""
    h = th.zeros(x.shape[0], W_ih.shape[0], dtype=x.dtype); out = []
    for t in range(T):
        pre = x[:, t:t + 1] * W_ih + h @ W_hh.T
        out.append(pre); h = th.relu(pre)
    return th.stack(out, 1), h


def skeleton_batch(k, B, m, gen):
    """B exact ties of kind k (1 M/S, 2 S/R), each with m background redraws: positions
    below the tied value v are redrawn iid N(0,1) truncated below v, the tied pair and
    (for k=2) the max stay fixed, as in region_count.py. Returns x [B*m,10], a0 [B]."""
    sk, a, b = sample_ties(k, B, gen)
    v = sk.gather(1, a[:, None])[:, 0]
    a0 = th.minimum(a, b)
    bgmask = sk < v[:, None] - 1e-6
    Fv = norm.cdf(v.double().numpy())
    u = th.rand(B, m, T, generator=gen, dtype=th.float64).numpy()
    bg = th.tensor(norm.ppf(np.clip(u * Fv[:, None, None], 1e-300, 1))).float()
    x = th.where(bgmask[:, None, :], bg, sk[:, None, :]).reshape(B * m, T)
    return x, a0


def switch_soft(pre, a0, m, tau):
    """Per-neuron soft switching g_j = mean over skeletons of sum_{t > a0} p(1-p),
    p = mean sigmoid(pre/tau) over the m backgrounds. pre [B*m,10,16]."""
    B = a0.shape[0]
    p = th.sigmoid(pre / tau).view(B, m, T, -1).mean(1)
    tmask = (th.arange(T)[None, :] > a0[:, None]).to(p.dtype)
    return (p * (1 - p) * tmask[:, :, None]).sum(1).mean(0)


@th.no_grad()
def accuracy(W, n_log2=26, chunk_log2=22, seed=4242):
    W_ih, W_hh, W_out = W; g = th.Generator().manual_seed(seed); c = 0
    for _ in range(1 << (n_log2 - chunk_log2)):
        x = th.randn(1 << chunk_log2, T, generator=g)
        _, h = pre_acts(x, W_ih, W_hh)
        c += ((h @ W_out.T).argmax(-1) == x.topk(2, -1).indices[:, 1]).sum().item()
    return c / (1 << n_log2)


@th.no_grad()
def switch_hard(W, B=256, m=512, seed=7):
    """Hard version of the penalty per tie kind: g_j with p = P(gate on | skeleton)."""
    W_ih, W_hh, _ = W; gen = th.Generator().manual_seed(seed); out = {}
    for k in (1, 2):
        x, a0 = skeleton_batch(k, B, m, gen)
        pre, _ = pre_acts(x, W_ih, W_hh)
        p = (pre > 0).float().view(B, m, T, -1).mean(1)
        tmask = (th.arange(T)[None, :] > a0[:, None]).float()
        out[k] = (p * (1 - p) * tmask[:, :, None]).sum(1).mean(0).numpy()
    return out


@th.no_grad()
def region_count(W, n_skel=48, n_draw=1 << 14, seed=21, sub=None):
    """Patterns (bits of neurons `sub` at steps > a0) covering 90/99/99.9% of the mass,
    median over skeletons, per tie kind; as probes/region_count.py."""
    W_ih, W_hh, _ = W; gen = th.Generator().manual_seed(seed); out = {}
    sub = list(range(W_ih.shape[0])) if sub is None else list(sub)
    for k in (1, 2):
        x, a0 = skeleton_batch(k, n_skel, n_draw, gen)
        pre, _ = pre_acts(x, W_ih, W_hh)
        bits = (pre[:, :, sub] > 0).view(n_skel, n_draw, T, len(sub))
        covers = []
        for i in range(n_skel):
            Bm = bits[i, :, int(a0[i]) + 1:, :].reshape(n_draw, -1).numpy()
            if Bm.shape[1] == 0:
                continue
            key = np.packbits(Bm, axis=1)
            _, cnt = np.unique(key, axis=0, return_counts=True)
            c = np.sort(cnt)[::-1].cumsum() / n_draw
            covers.append([int(np.searchsorted(c, q) + 1) for q in (.9, .99, .999)])
        out[k] = np.median(np.array(covers), 0).tolist()
    return out


@th.no_grad()
def kink_stats(W, nc=128, K=4096, w=4, seed=0):
    """Last-step kinks per great circle, error flips, share of flips within +-w points of a
    last-step kink, and its enrichment over chance (decision_kinks.py)."""
    W_ih, W_hh, W_out = W; gen = th.Generator().manual_seed(seed)
    g1, g2 = th.randn(nc, 1, T, generator=gen), th.randn(nc, 1, T, generator=gen)
    t = th.linspace(0, 2 * np.pi, K + 1)[:-1].view(1, K, 1)
    x = (th.cos(t) * g1 + th.sin(t) * g2).reshape(-1, T)
    pre, h = pre_acts(x, W_ih, W_hh)
    ok = ((h @ W_out.T).argmax(-1) == x.topk(2, -1).indices[:, 1]).float().view(nc, K)
    last = (pre[:, -1] > 0).view(nc, K, -1)
    lkink = (last != last.roll(1, 1)).any(-1)
    flip = ok != ok.roll(1, 1)
    near = th.zeros_like(lkink)
    for d in range(-w, w + 1):
        near |= lkink.roll(d, 1)
    dens = lkink.float().mean().item()
    local = near[flip].float().mean().item() if flip.any() else float('nan')
    chance = 1 - (1 - dens) ** (2 * w + 1)
    return dict(last=lkink.float().sum(1).mean().item(), flips=flip.float().sum(1).mean().item(),
                local=local, enrich=local / chance if chance > 0 else float('nan'))


def all_metrics(W):
    sh = switch_hard(W); rc = region_count(W)
    return dict(acc=accuracy(W), switch={k: v.tolist() for k, v in sh.items()},
                n_switching={k: int((v > 0.05).sum()) for k, v in sh.items()},
                regions={k: v for k, v in rc.items()}, kinks=kink_stats(W))
