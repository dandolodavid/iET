'''
Compare iET and baseline attribution methods against the simulated ground truth (benchmark/simulate.py).

Usage (from the repository root):  python benchmark/run.py [n_seeds] [n_coverage_reps]

Methods (each one allocates the observed revenue to channels):
  iET values   : row-level split by iET values (iET.compute_attribution, how="attribution_values")
  iET removal  : iET.estimate_removal_effect()["attributed_target"]
  last click   : last non-Direct touch of the journey
  linear       : equal split over the non-Direct touches of the journey
  Shapley      : simplified Shapley, v(S) = revenue of journeys whose set of paid channels is a subset of S

Direct is not a marketing channel (no ground truth), so every method's allocation is renormalised over the paid
channels and compared with the true share of incremental revenue.
'''
import os
import sys
from math import factorial

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iET.iET import iET  # noqa: E402
from benchmark.simulate import SCENARIOS, DIRECT, simulate  # noqa: E402


def fit_iet(df):
    m = iET('channel', 'revenue', 'user', time_colname='time')
    m.estimate_proba(df)
    m.estimate_attribution()
    return m


def iet_values(m, df):
    return m.compute_attribution(df).groupby('channel')['iET_attribution'].sum()


def last_click(paid):
    last = paid.groupby('user').tail(1)
    rev = paid.groupby('user')['revenue_journey'].first()
    return rev.reindex(last['user']).groupby(last['channel'].to_numpy()).sum()


def linear(paid):
    share = paid['revenue_journey'] / paid.groupby('user')['user'].transform('size')
    return share.groupby(paid['channel']).sum()


def shapley(paid, journey_rev, channels):
    n = len(channels)
    bit = {c: 1 << i for i, c in enumerate(channels)}
    mask = paid.drop_duplicates(['user', 'channel'])['channel'].map(bit).groupby(paid.drop_duplicates(['user', 'channel'])['user']).sum()
    mask = mask.reindex(journey_rev.index, fill_value=0)  # Direct-only journeys have the empty set
    v = np.bincount(mask.to_numpy(), weights=journey_rev.to_numpy(), minlength=1 << n)
    for i in range(n):  # subset sums: v[S] = revenue of journeys whose set is a subset of S
        for s in range(1 << n):
            if s & (1 << i):
                v[s] += v[s ^ (1 << i)]
    size = np.array([bin(s).count('1') for s in range(1 << n)])
    phi = {}
    for i, c in enumerate(channels):
        s = np.arange(1 << n)
        s = s[(s & (1 << i)) == 0]
        w = np.array([factorial(k) * factorial(n - k - 1) for k in size[s]]) / factorial(n)
        phi[c] = (w * (v[s | (1 << i)] - v[s])).sum()
    return pd.Series(phi)


def allocations(df, m, channels):
    journey_rev = df.groupby('user')['revenue'].sum()
    paid = df[df['channel'] != DIRECT].assign(revenue_journey=lambda x: x['user'].map(journey_rev))
    paid = paid[paid['revenue_journey'] > 0]
    out = pd.DataFrame({'iET values': iet_values(m, df),
                        'iET removal': m.estimate_removal_effect()['attributed_target'],
                        'last click': last_click(paid),
                        'linear': linear(paid),
                        'Shapley': shapley(paid, journey_rev, channels)}).reindex(channels).fillna(0)
    return out / out.sum()


def evaluate(scn_name, seed):
    scn = SCENARIOS[scn_name]
    channels = [c['name'] for c in scn['channels']]
    df, truth, info = simulate(scn, seed)
    m = fit_iet(df)
    shares = allocations(df, m, channels)
    true_share = truth['incremental_revenue'] / truth['incremental_revenue'].sum()
    rows = []
    for method in shares:
        rows.append(dict(scenario=scn_name, seed=seed, method=method,
                         spearman=shares[method].rank().corr(true_share.rank()),  # Spearman without scipy
                         share_mae_pp=100 * (shares[method] - true_share).abs().mean()))
    re_level = m.estimate_removal_effect()['removal_effect'].reindex(channels)
    detail = shares.assign(**{'true share': true_share, 'iET removal effect': re_level,
                              'true removal effect': truth['true_removal_effect']})
    return rows, detail, info


def coverage(scn_name, reps, n_boot=100, big_factor=10):
    '''
    Coverage of the 90% bootstrap interval of the iET values. The estimand is the population iET value,
    approximated by fitting iET on a sample big_factor times larger from the same generator (different seed).
    This checks the statistical calibration of the bootstrap, not causal validity.
    '''
    scn = SCENARIOS[scn_name]
    channels = [c['name'] for c in scn['channels']]
    big = dict(scn, n_users=scn['n_users'] * big_factor)
    target = fit_iet(simulate(big, seed=10_000)[0])._attribution_df['attribution_values'].reindex(channels)
    hits, widths = [], []
    for r in range(reps):
        ci = fit_iet(simulate(scn, seed=1000 + r)[0]).bootstrap(n_boot=n_boot, quantiles=(0.05, 0.95), random_state=r)
        ci = ci.reindex(channels)
        hits.append((ci[0.05] <= target) & (target <= ci[0.95]))
        widths.append((ci[0.95] - ci[0.05]) / target)
    return pd.DataFrame({'coverage_90': pd.concat(hits, axis=1).mean(axis=1),
                         'rel_width': pd.concat(widths, axis=1).mean(axis=1)})


def md(df, floatfmt='{:.2f}'):
    df = df.copy()
    cols = [df.index.name or ''] + [str(c) for c in df.columns]
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for idx, row in df.iterrows():
        cells = [str(idx)] + [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


if __name__ == '__main__':
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    n_cov = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    all_rows = []
    for name in SCENARIOS:
        details, infos = [], []
        for seed in range(n_seeds):
            rows, detail, info = evaluate(name, seed)
            all_rows += rows
            details.append(detail)
            infos.append(info)
        info = pd.DataFrame(infos).mean()
        print(f'\n### {name}\n')
        print(f'{info.journeys:.0f} journeys, {info.rows:.0f} rows, conversion rate {info.conversion_rate:.1%}, '
              f'{1 - info.baseline_revenue / info.total_revenue:.0%} of revenue is incremental to the channels '
              f'(mean of {n_seeds} seeds)\n')
        print('Attributed share of paid-channel revenue vs true share of incremental revenue (mean over seeds):\n')
        print(md(sum(details) / n_seeds * 100, '{:.1f}'))

    res = pd.DataFrame(all_rows)
    summ = res.groupby(['scenario', 'method'], sort=False).agg(
        spearman=('spearman', 'mean'), spearman_sd=('spearman', 'std'),
        share_mae_pp=('share_mae_pp', 'mean'), share_mae_sd=('share_mae_pp', 'std'))
    print(f'\n### Summary ({n_seeds} seeds)\n')
    print(md(summ.reset_index().set_index('scenario')))

    if n_cov:
        print(f'\n### Bootstrap coverage ({n_cov} replicates, 100 bootstrap samples, nominal 90%)\n')
        cov = pd.concat({s: coverage(s, n_cov) for s in SCENARIOS}, axis=1)
        cov.columns = [f'{s} {c}' for s, c in cov.columns]
        print(md(cov))
