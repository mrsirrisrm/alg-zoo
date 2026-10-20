"""Pareto table for the fine-tuned M(16,10) variants (PLAN G1-G3) from probes/finetune/*_metrics.json,
plus the exact-region polyhedral counts from probes/polyhedral_ft.out where present.
Run from estimate/: python probes/finetune_summary.py"""
import glob, json, os, re
rows = [('trained', 'probes/finetune/trained_metrics.json')]
rows += sorted((os.path.basename(f)[:-13], f) for f in glob.glob('probes/finetune/*_lam*_metrics.json'))
rows += sorted(('long/' + os.path.basename(f)[:-13], f) for f in glob.glob('probes/finetune/long/*_lam*_metrics.json'))
rows += sorted(('hard/' + os.path.basename(f)[:-13], f) for f in glob.glob('probes/finetune/lastkink_hard/*_lam*_metrics.json'))
poly = {}   # exact-region runs (KSET = all 16, 2^14 discovery draws, 99% cover), keyed by the weights path
if os.path.exists('probes/polyhedral_ft_fit14.out'):
    txt = open('probes/polyhedral_ft_fit14.out').read()
    for m in re.finditer(r'=== (\S+) KSET all16 fit14 cover99\n(?:skel.*\n)*\{(.*?)\}', txt, re.S):
        d = json.loads('{' + m.group(2) + '}')
        poly[m.group(1).replace('probes/finetune/', '').replace('.npz', '').replace('weights', 'trained')] = d
def key(k):
    return float(k.split('lam')[1]) if 'lam' in k else -1
print("acc on 2^26 (sd 2.6e-5); d_acc vs the trained model; regions99 = patterns of all 16 neurons at steps after the first tied element covering 99% of mass, median over 48 skeletons;")
print("switching = neurons with hard sum_t p(1-p) > 0.05 inside skeletons; last-kinks per great circle and the enrichment of error flips at them (decision_kinks);")
print("poly = polyhedral.py S/R with KSET = all 16 (exact regions over all 10 steps, 2^14 discovery draws, 99% cover): regions median, poly / MC-exact tie error, share of test mass in discovered regions, integration seconds.")
print(f"{'model':>22} {'acc':>7} {'d_acc':>7} {'reg M/S':>8} {'reg S/R':>8} {'sw M/S':>6} {'sw S/R':>6} {'lkinks':>7} {'enrich':>6} {'poly regions':>12} {'poly/exact':>18} {'found':>5} {'sec':>5}")
for name, f in rows:
    v = json.load(open(f)); t = json.load(open('probes/finetune/trained_metrics.json'))
    p = poly.get(name)
    ps = f"{p['regions_median']:12.0f} {p['poly']:.4f} / {p['mc_exact']:.4f} {p['cover']:5.2f} {p['poly_seconds']:5.0f}" if p else ''
    print(f"{name:>22} {v['acc']:7.4f} {v['acc'] - t['acc']:+7.4f} {v['regions']['1'][1]:8.0f} {v['regions']['2'][1]:8.0f} {v['n_switching']['1']:6d} {v['n_switching']['2']:6d} {v['kinks']['last']:7.1f} {v['kinks']['enrich']:6.2f} {ps}")
