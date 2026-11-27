'''
Semi-synthetic journey simulator with known causal channel effects.

Each user has a latent intent z ~ N(0, 1) and lives T days. Every day:
  1) each channel c exposes the user with probability sigmoid(logit(p_c) + kappa_c * z)
     (kappa_c > 0 = confounding: the channel is shown mostly to high-intent users);
     channels with needs_prior=True (retargeting) can only reach users already exposed to another channel;
  2) each exposure adds lift_c to a per-channel adstock that decays by decay_c per day;
  3) the user converts with daily hazard sigmoid(a0 + b * z + sum of adstocks); the journey ends at the purchase.
Revenue of a purchase ~ LogNormal(mu + rho * z, sigma), drawn per user (independent of the channels).

Observed data: one row per exposure; the purchase is recorded on the last exposure of the purchase day,
or on a "Direct" row when the user had no exposure that day.

Ground truth: incremental revenue of channel c = revenue - revenue with c switched off, re-simulated on
the same users with the same random numbers (common random numbers), so it is the exact effect on this sample.
'''
import numpy as np
import pandas as pd

DIRECT = 'Direct'

_BASE_CHANNELS = [
    # name,        p/day, lift, decay, kappa, needs_prior
    ('Search',      0.03, 1.2,  0.3,   0.0,   False),
    ('Social',      0.06, 0.4,  0.7,   0.0,   False),
    ('Display',     0.08, 0.25, 0.9,   0.0,   False),  # weak but long carry-over: an "assist" channel
    ('Email',       0.04, 0.8,  0.5,   0.0,   False),
    ('Retargeting', 0.15, 0.2,  0.5,   0.0,   True),
]


def _channels(rows, kappa=None):
    kappa = kappa or {}
    return [dict(name=n, p=p, lift=l, decay=d, kappa=kappa.get(n, k), needs_prior=r) for n, p, l, d, k, r in rows]


SCENARIOS = {
    'no_confounding': dict(n_users=20000, T=30, a0=-6.0, b=1.0, mu=4.0, rho=0.3, sigma=0.6,
                           channels=_channels(_BASE_CHANNELS)),
    # textbook confounding: Retargeting has NO causal effect and is shown mostly to high-intent users,
    # Search exposure is also driven by intent (people who want to buy search more)
    'confounding': dict(n_users=20000, T=30, a0=-6.0, b=1.0, mu=4.0, rho=0.3, sigma=0.6,
                        channels=_channels([r if r[0] != 'Retargeting' else ('Retargeting', 0.15, 0.0, 0.5, 0.0, True)
                                            for r in _BASE_CHANNELS], kappa={'Retargeting': 2.0, 'Search': 1.0})),
    # control: as no_confounding but with 3x the lifts, so most revenue is caused by the channels
    # (lowering a0 instead does not work: with logit lifts the incremental fraction barely changes)
    'strong_effects': dict(n_users=20000, T=30, a0=-6.0, b=1.0, mu=4.0, rho=0.3, sigma=0.6,
                           channels=_channels([(n, p, 3 * l, d, k, r) for n, p, l, d, k, r in _BASE_CHANNELS])),
    # fewer users and two rare channels with large effects
    'sparse': dict(n_users=5000, T=30, a0=-6.0, b=1.0, mu=4.0, rho=0.3, sigma=0.6,
                   channels=_channels(_BASE_CHANNELS + [('Affiliate', 0.002, 1.5, 0.5, 0.0, False),
                                                        ('Podcast',   0.003, 0.8, 0.8, 0.0, False)])),
}


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _draws(scn, seed):
    rng = np.random.default_rng(seed)
    N, T, C = scn['n_users'], scn['T'], len(scn['channels'])
    z = rng.normal(size=N)
    return dict(z=z,
                u_exp=rng.random((T, N, C)),
                u_conv=rng.random((T, N)),
                order=rng.random((T, N, C)),
                revenue=np.exp(scn['mu'] + scn['rho'] * z + scn['sigma'] * rng.normal(size=N)))


def _run(scn, d, active):
    '''Simulate with the given draws; active[c]=False switches channel c off. Returns exposures and purchase day.'''
    ch = scn['channels']
    N, T = len(d['z']), scn['T']
    p, lift, decay, kappa = (np.array([c[k] for c in ch]) for k in ('p', 'lift', 'decay', 'kappa'))
    needs_prior = np.array([c['needs_prior'] for c in ch])
    p_exp = _sigmoid(np.log(p / (1 - p)) + kappa * d['z'][:, None])
    base_logit = scn['a0'] + scn['b'] * d['z']

    E = np.zeros((T, N, len(ch)), dtype=bool)
    stock = np.zeros((N, len(ch)))
    seen = np.zeros(N, dtype=bool)
    alive = np.ones(N, dtype=bool)
    conv_day = np.full(N, -1)
    for t in range(T):
        e = (d['u_exp'][t] < p_exp) & active & alive[:, None]
        e[:, needs_prior] &= seen[:, None]
        stock = stock * decay + lift * e
        conv = alive & (d['u_conv'][t] < _sigmoid(base_logit + stock.sum(axis=1)))
        conv_day[conv] = t
        seen |= e[:, ~needs_prior].any(axis=1)
        alive &= ~conv
        E[t] = e
    return E, conv_day


def simulate(scn, seed):
    '''
    Returns
    -------
    df : user, time, channel, revenue (one row per exposure, plus Direct rows for purchases without exposure)
    truth : dataframe indexed by channel with incremental_revenue and true_removal_effect (= incremental / total)
    info : dict with total revenue, baseline revenue (all channels off) and conversion rate
    '''
    d = _draws(scn, seed)
    names = [c['name'] for c in scn['channels']]
    on = np.ones(len(names), dtype=bool)
    E, conv_day = _run(scn, d, on)

    def revenue(cd):
        return d['revenue'][cd >= 0].sum()

    total = revenue(conv_day)
    incremental = {}
    for i, n in enumerate(names):
        off = on.copy()
        off[i] = False
        incremental[n] = total - revenue(_run(scn, d, off)[1])
    baseline = revenue(_run(scn, d, np.zeros(len(names), dtype=bool))[1])

    t, u, c = np.nonzero(E)
    df = pd.DataFrame({'user': u, 'time': t + 0.9 * d['order'][t, u, c], 'channel': np.array(names)[c], 'revenue': 0.0})
    # purchase goes on the last exposure of the purchase day, else on a Direct row
    buyers = np.flatnonzero(conv_day >= 0)
    on_day = df[conv_day[df['user']] == df['time'].astype(int)]
    last = on_day.groupby('user')['time'].idxmax()
    df.loc[last.to_numpy(), 'revenue'] = d['revenue'][last.index]
    direct = np.setdiff1d(buyers, last.index)
    df = pd.concat([df, pd.DataFrame({'user': direct, 'time': conv_day[direct] + 0.95, 'channel': DIRECT,
                                      'revenue': d['revenue'][direct]})], ignore_index=True)
    df = df.sort_values(['user', 'time'], ignore_index=True)

    truth = pd.DataFrame({'incremental_revenue': pd.Series(incremental)})
    truth['true_removal_effect'] = truth['incremental_revenue'] / total
    info = dict(total_revenue=total, baseline_revenue=baseline, conversion_rate=len(buyers) / len(conv_day),
                rows=len(df), journeys=df['user'].nunique())
    return df, truth, info


if __name__ == '__main__':
    for name, scn in SCENARIOS.items():
        df, truth, info = simulate(scn, seed=0)
        # self-check: every simulated purchase is in the observed data exactly once
        assert np.isclose(df['revenue'].sum(), info['total_revenue'])
        print(name, {k: round(v, 3) for k, v in info.items()})
        print(truth.round(3), '\n')
