import numpy as np
import pandas as pd
import pytest
from iET.iET import iET


def toy():
    # "Assist" never converts directly but always precedes the conversion on "Close"
    rows = []
    for u in range(50):
        rows += [(f'u{u}', 1, 'Assist', 0), (f'u{u}', 2, 'Close', 100 if u < 25 else 0)]
    for u in range(50, 100):
        rows += [(f'u{u}', 1, 'Close', 0)]
    return pd.DataFrame(rows, columns=['user', 't', 'ch', 'rev'])


def fit(df, **kw):
    m = iET('ch', 'rev', 'user', time_colname='t', **kw)
    m.estimate_proba(df)
    return m, m.estimate_attribution()


def test_non_converting_touchpoint_is_kept():
    m, att = fit(toy())
    assert set(att.index) == {'Assist', 'Close'}
    assert att.loc['Assist', 'attribution_values'] > 0
    assert m.estimate_removal_effect().loc['Assist', 'removal_effect'] > 0


def test_time_order_not_row_order():
    df = toy()
    m1, att1 = fit(df)
    m2, att2 = fit(df.sample(frac=1, random_state=0))
    pd.testing.assert_frame_equal(m1.get_transaction_matrix(), m2.get_transaction_matrix())
    pd.testing.assert_frame_equal(att1, att2)
    assert m1.get_transaction_matrix().loc['Assist', 'Close'] == 1


def test_probabilities_and_bellman_fixed_point():
    m, att = fit(toy())
    f, P = m.get_info(), m.get_transaction_matrix()
    np.testing.assert_allclose(f[['target_prob', 'quit_prob', 'move_prob']].sum(axis=1), 1)
    assert set(P.sum(axis=1).round(10)) <= {0.0, 1.0}
    v = att['attribution_values'].reindex(f.index)
    np.testing.assert_allclose(v, f.target_prob * f.xT + f.move_prob * (P @ v))
    # horizon of 1 step = only direct target
    np.testing.assert_allclose(m.estimate_attribution(max_steps=1)['attribution_values'].reindex(f.index), f.target_prob * f.xT)


def test_old_signature_still_works():
    _, att = fit(toy())
    m, _ = fit(toy())
    pd.testing.assert_frame_equal(m.estimate_attribution(max_iter=20, epsilon=0.001), att)


@pytest.mark.parametrize('how', ['attribution_values', 'removal_effect'])
def test_row_attribution_sums_to_target(how):
    # numeric ids, duplicated index, two purchases for the same client and no session
    df = pd.DataFrame({'user': [1, 1, 1, 1, 2, 2, 3],
                       't': [1, 2, 3, 4, 1, 2, 1],
                       'ch': ['A', 'B', 'A', 'B', 'A', 'B', 'B'],
                       'rev': [0, 10, 0, 5, 0, 0, 7]}, index=[0, 0, 1, 1, 2, 2, 3])
    m, _ = fit(df)
    m.estimate_removal_effect()
    out = m.compute_attribution(df, how=how)
    assert out.index.equals(df.index)
    assert out['iET_attribution'].sum() == pytest.approx(df['rev'].sum())
    assert out.loc[out.user == 2, 'iET_attribution'].sum() == 0


def test_default_attribution_is_removal_effect():
    # notebook flow: no explicit estimate_removal_effect() call before compute_attribution
    df = toy()
    m, _ = fit(df)
    pd.testing.assert_frame_equal(m.compute_attribution(df), m.compute_attribution(df, how='removal_effect'))
    with pytest.raises(ValueError):
        m.compute_attribution(df, how='nope')


def test_removal_effect_attributes_total_target():
    m, _ = fit(toy())
    assert m.estimate_removal_effect()['attributed_target'].sum() == pytest.approx(toy()['rev'].sum())


def test_compute_session():
    df = pd.DataFrame({'user': ['a'] * 5 + ['b'] * 2,
                       't': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-20', '2024-01-21', '2024-01-01', '2024-01-02']),
                       'ch': list('ABABABA'),
                       'rev': [0, 9, 0, 0, 0, 0, 0]})
    m = iET('ch', 'rev', 'user', time_colname='t', session_colname='session')
    out = m.compute_session(df)
    # new session after the purchase and after a gap > 7 days, numbering starts at 1 for every user
    assert out['session'].tolist() == [1, 1, 2, 3, 3, 1, 1]


def test_top_k_groups_other():
    df = toy()
    df.loc[df.index[:3], 'ch'] = 'Rare'
    m, att = fit(df, use_top_k=2)
    assert set(att.index) == {'Assist', 'Close', 'Other'}


def test_smoothing_and_bootstrap():
    m, _ = fit(toy(), alpha=1.0)
    assert (m.get_transaction_matrix().values.diagonal() == 0).all()
    ci = m.bootstrap(n_boot=20, random_state=0)
    assert list(ci.columns) == [0.05, 0.5, 0.95]
    assert (ci[0.05] <= ci[0.95]).all()


def test_plots_build():
    m, _ = fit(toy())
    for fig in (m.plot(), m.plot_proba(), m.plot_transaction(), m.plot_ExT()):
        assert fig is not None
