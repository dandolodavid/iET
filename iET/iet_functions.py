import pandas as pd
import numpy as np

OUTCOMES = ['target', 'quit', 'move']


def prepare_data(dataframe, client_id, session, touchpoint, time=None, top_k_name=None):
    '''
    Shared preprocessing for estimation and attribution.

    1) Build the journey key (client id, or client id + session id)
    2) Sort each journey by time (by original row order when no time column is given)
    3) Map touchpoints outside top_k_name to "Other" (dropping them would create fake transitions)

    Returns:
    ---------
    data : dataframe with a positional index (0..n-1 in the original row order)
    key : name of the journey key column
    '''
    data = dataframe.reset_index(drop=True)
    if session:
        key = 'client_session_id'
        data[key] = data[client_id].astype(str) + ' : ' + data[session].astype(str)
    else:
        key = client_id
    data = data.sort_values([key] + ([time] if time else []), kind='stable')
    if top_k_name is not None:
        data[touchpoint] = data[touchpoint].where(data[touchpoint].isin(top_k_name), 'Other')
    return data, key


def build_steps(data, key, touchpoint, target, accept_consecutive_touchpoint):
    '''
    One row per step of each journey, labelled with its outcome:
    "target" if the target is reached on that step, "move" if the journey continues, "quit" otherwise.
    Unless accept_consecutive_touchpoint, consecutive rows on the same touchpoint are merged in one step
    (target summed), so probabilities and transition matrix are estimated on exactly the same events.
    '''
    steps = data[[key, touchpoint, target]]
    if not accept_consecutive_touchpoint:
        new_step = (steps[key] != steps[key].shift()) | (steps[touchpoint] != steps[touchpoint].shift())
        steps = steps.groupby(new_step.cumsum().to_numpy(), sort=False).agg({key: 'first', touchpoint: 'first', target: 'sum'})
    steps = steps.reset_index(drop=True)
    steps['first'] = steps[key] != steps[key].shift()
    steps['next'] = steps.groupby(key, sort=False)[touchpoint].shift(-1)
    steps['outcome'] = np.where(steps[target] != 0, 'target', np.where(steps['next'].isna(), 'quit', 'move'))
    return steps


def estimate_probabilities(steps, touchpoint, target, alpha=0.0, accept_consecutive_touchpoint=False, weight=None):
    '''
    Estimate all the model quantities from the steps.

    Parameters:
    -----------
    alpha : Dirichlet (additive) smoothing added to outcome and transition counts
    weight : optional weight of each step (used by the bootstrap); default 1

    Returns:
    ---------
    final_df : target_prob, quit_prob, move_prob (they sum to 1) and xT (mean target when reached) by touchpoint
    prob_transition : row-stochastic transition matrix between touchpoints (rows with no moves are all 0)
    count_transition : (weighted) transition counts
    start : distribution of the first touchpoint of the journeys
    '''
    w = pd.Series(1.0, index=steps.index) if weight is None else weight
    tp = steps[touchpoint]

    counts = pd.crosstab(tp, steps['outcome'], values=w, aggfunc='sum').reindex(columns=OUTCOMES).fillna(0) + alpha
    final_df = counts.div(counts.sum(axis=1).replace(0, 1), axis=0) \
                     .set_axis(['target_prob', 'quit_prob', 'move_prob'], axis=1) \
                     .rename_axis(index=None, columns=None)
    tps = final_df.index

    # touchpoints that never convert get xT = 0 instead of being dropped: they still pass value through moves
    conv = steps[target] != 0
    xT = (steps[target] * w)[conv].groupby(tp[conv]).sum() / w[conv].groupby(tp[conv]).sum()
    final_df['xT'] = xT.reindex(tps).fillna(0)

    move = steps['outcome'] == 'move'
    count_transition = w[move].groupby([tp[move], steps.loc[move, 'next']]).sum().unstack() \
                              .reindex(index=tps, columns=tps).fillna(0).rename_axis(index=None, columns=None)
    smoothed = count_transition + alpha
    if not accept_consecutive_touchpoint:
        smoothed = smoothed.mask(np.eye(len(tps), dtype=bool), 0)
    prob_transition = smoothed.div(smoothed.sum(axis=1).replace(0, 1), axis=0)

    first = steps['first']
    start = w[first].groupby(tp[first]).sum().reindex(tps).fillna(0)
    start = start / start.sum()

    return final_df, prob_transition, count_transition, start


def _reward_and_q(final_df, prob_transition):
    tp = final_df.index
    q = final_df['move_prob'].to_numpy()[:, None] * prob_transition.loc[tp, tp].to_numpy()
    r = (final_df['target_prob'] * final_df['xT']).to_numpy()
    return r, q


def calculate_attribution_values(final_df, prob_transition, max_steps=None):
    '''
    Value of each touchpoint: V = target_prob * xT + move_prob * P V (Bellman equation of the absorbing chain).
    With max_steps=None the exact solution (I - diag(move_prob) P)^-1 (target_prob * xT) is returned; the system
    is non-singular because every journey ends (with target or quit) in a finite number of steps.
    With max_steps=k only the target reached within k steps is counted.
    '''
    r, q = _reward_and_q(final_df, prob_transition)
    if max_steps is None:
        v = np.linalg.solve(np.eye(len(r)) - q, r)
    else:
        v = np.zeros(len(r))
        for _ in range(max_steps):
            v = r + q @ v
    return pd.Series(v, index=final_df.index, name='attribution_values')


def removal_effects(final_df, prob_transition, start):
    '''
    Removal effect of each touchpoint (Anderl et al. 2016): relative drop of the expected target of a journey
    when the touchpoint is removed and every journey reaching it quits.
    '''
    r, q = _reward_and_q(final_df, prob_transition)
    s0 = start.reindex(final_df.index).to_numpy()
    n = len(r)
    base = s0 @ np.linalg.solve(np.eye(n) - q, r)
    effects = np.zeros(n)
    if base > 0:
        for i in range(n):
            keep = np.arange(n) != i
            v = np.linalg.solve(np.eye(n - 1) - q[np.ix_(keep, keep)], r[keep])
            effects[i] = 1 - s0[keep] @ v / base
    return pd.Series(effects, index=final_df.index, name='removal_effect')
