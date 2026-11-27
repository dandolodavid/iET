import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from iET.iet_functions import *
from iET.compute_attribution_function import *

class iET():

    def __init__(self, touchpoint_colname, target_colname, client_id_colname, time_colname = None, session_colname = None, use_top_k = None, accept_consecutive_touchpoint = False, alpha = 0.0):
        '''
        Parameters:
        ------------
        time_colname : column used to order each journey (if None the row order of the dataframe is used)
        session_colname : optional session column; journeys are (client, session) pairs
        use_top_k : keep the k touchpoints with most journeys, the others are grouped as "Other"
        accept_consecutive_touchpoint : if False consecutive rows on the same touchpoint are merged in one step
        alpha : Dirichlet (additive) smoothing for outcome and transition counts, useful with sparse touchpoints
        '''
        self._touchpoint_colname = touchpoint_colname
        self._target_colname = target_colname
        self._client_id_colname = client_id_colname
        self._session_colname = session_colname
        self._time_colname = time_colname
        self._top_k = use_top_k
        self._top_k_name = None
        self._accept_consecutive_touchpoint = accept_consecutive_touchpoint
        self._alpha = alpha

    def estimate_proba(self, dataframe ):
        '''
        Prepare the dataframe and estimate all the required probabilities
        '''
        tp = self._touchpoint_colname
        data, key = prepare_data(dataframe, self._client_id_colname, self._session_colname, tp, self._time_colname)
        self._client_session_colname = key

        if self._top_k:
            self._top_k_name = data.groupby(tp)[key].nunique().sort_values().index[-self._top_k:].tolist()
            data[tp] = data[tp].where(data[tp].isin(self._top_k_name), 'Other')

        self._count_df = data.groupby(tp)[key].nunique().to_frame(key)
        self._steps = build_steps(data, key, tp, self._target_colname, self._accept_consecutive_touchpoint)
        self._final_df, self._transaction_df, self._count_transaction_df, self._start_df = \
            estimate_probabilities(self._steps, tp, self._target_colname, self._alpha, self._accept_consecutive_touchpoint)

    def estimate_attribution(self, max_steps = None, max_iter = None, epsilon = None):
        '''
        Value of each touchpoint: expected target of a journey that is currently on that touchpoint.

        Parameters:
        ------------
        max_steps : if set, only the target reached within max_steps steps is counted (lookback horizon);
                    if None the exact solution is computed
        max_iter, epsilon : ignored, kept so that calls written for previous versions keep working
        '''
        values = calculate_attribution_values(self._final_df, self._transaction_df, max_steps)
        self._start_value = float(self._start_df @ values)
        self._attribution_df = values.to_frame().sort_values(by = 'attribution_values', ascending = False)
        return self._attribution_df.copy()

    def estimate_removal_effect(self):
        '''
        Removal effect of each touchpoint (relative drop of the expected target when it is removed) and the total
        target attributed proportionally to it (attributed_target sums to the total target of the data).
        '''
        effects = removal_effects(self._final_df, self._transaction_df, self._start_df)
        out = effects.to_frame()
        out['attributed_target'] = self._steps[self._target_colname].sum() * effects / effects.sum() if effects.sum() > 0 else 0.0
        self._removal_df = out.sort_values(by = 'removal_effect', ascending = False)
        return self._removal_df.copy()

    def bootstrap(self, n_boot = 100, quantiles = (0.05, 0.5, 0.95), random_state = None):
        '''
        Bootstrap (resampling journeys with replacement) of the attribution values.

        Returns:
        -----------
        dataframe touchpoint x quantile
        '''
        rng = np.random.default_rng(random_state)
        codes, uniques = pd.factorize(self._steps[self._client_session_colname])
        samples = []
        for _ in range(n_boot):
            journey_weight = np.bincount(rng.integers(0, len(uniques), len(uniques)), minlength = len(uniques))
            weight = pd.Series(journey_weight[codes].astype(float), index = self._steps.index)
            final_df, prob_transition, _, _ = estimate_probabilities(self._steps, self._touchpoint_colname, self._target_colname,
                                                                     self._alpha, self._accept_consecutive_touchpoint, weight)
            samples.append(calculate_attribution_values(final_df, prob_transition))
        return pd.DataFrame(samples).quantile(list(quantiles)).T

    def get_transaction_matrix(self):
       return self._transaction_df

    def get_info(self):
       return self._final_df

    def plot(self):
        '''
        '''
        plot_df = self._attribution_df.merge(self._count_df, left_index=True, right_index=True)
        size = plot_df[self._client_session_colname].values

        fig = go.Figure(data=[go.Bar(
            x = plot_df.index.tolist(),
            y = plot_df.attribution_values.values,
            width = size/max(size) # customize width here
        )])

        fig.update_layout(title={'text': 'Attribution values ( bar width is proportional to the number of journeys with that touchpoint )' })

        return fig

    def plot_proba(self):
        '''
        '''
        plot_df = self._final_df[['target_prob','quit_prob','move_prob']].stack().reset_index()
        plot_df.columns = [self._touchpoint_colname, 'hue', 'value']
        fig = px.bar(plot_df, x = self._touchpoint_colname, y = 'value', color = 'hue', title= 'Buy Quit and Move probability by channels')
        return fig

    def plot_transaction(self, verbose = True):
        '''
        '''
        z = self._transaction_df.values

        x = self._transaction_df.columns.tolist()
        y = self._transaction_df.index.tolist()

        z_text = np.char.add((100*self._transaction_df.values).round(2).astype(str), '%')
        if verbose:
            z_text = np.char.add(np.char.add(z_text, '<br>('), np.char.add(self._count_transaction_df.values.astype(int).astype(str), ')'))

        fig = go.Figure(go.Heatmap(z=z, x=x, y=y, text=z_text, texttemplate='%{text}', colorscale = 'bluered'))
        fig.update_yaxes(autorange = 'reversed')

        return fig

    def plot_ExT(self):
        '''
        '''
        plot_df = self._final_df.merge(self._count_df, left_index=True, right_index=True)
        size = plot_df[self._client_session_colname].values

        fig = go.Figure(data=[go.Bar(
            x = plot_df.index.tolist(),
            y = plot_df.xT.values,
            width = size/max(size) # customize width here
        )])

        fig.update_layout(title={'text': 'Expected target values ( bar width is proportional to the number of journeys with that touchpoint )' })

        return fig

    def compute_session(self, dataframe, max_delta_days = 7, additional_sorting_columns = None ):
        '''
        Compute the session based on the target and the days time delta. A session ends when the user reaches the target
        or after more than max_delta_days without actions; the next action starts a new session.

        Parameters:
        ------------
        dataframe: pd.DataFrame
            dataframe
        max_delta_days: int
            max number of consecutive days without any actions to consider the end of the session
        additional_sorting_columns : [str]
            list of additional columns used to split the journeys (user_column + additional_sorting + time_column)

        Returns:
        -----------
        dataframe sorted by user, additional columns and time, with the session column (1, 2, ... for each user) added

        '''
        if self._session_colname is None or self._time_colname is None:
            raise AttributeError('Parameters "session_colname" and "time_colname" are required to compute session. Please set a valid string for the colnames')

        group_cols = [self._client_id_colname] + list(additional_sorting_columns or [])
        data = dataframe.copy()
        data[self._time_colname] = pd.to_datetime(data[self._time_colname])
        data = data.sort_values(group_cols + [self._time_colname], kind = 'stable')

        groups = data.groupby(group_cols, sort = False, dropna = False)
        new_session = (groups[self._target_colname].shift().fillna(0) != 0) | (groups[self._time_colname].diff().dt.days > max_delta_days)
        session_in_group = new_session.groupby([data[c] for c in group_cols], sort = False, dropna = False).cumsum()
        # number the sessions 1..n per user across all the additional groups, so (user, session) is a unique journey
        session_id = session_in_group.groupby([data[c] for c in group_cols] + [session_in_group], sort = False, dropna = False).ngroup()
        data[self._session_colname] = session_id.groupby(data[self._client_id_colname]).rank(method = 'dense').astype(int)

        return data

    def compute_attribution(self, dataframe, how = 'removal_effect'):
        '''
        Calculate the attribution for each line of the dataset: the target of each journey is split among its lines
        proportionally to the weight of their touchpoint.
        how = "removal_effect" (default) uses the removal effects (how much each touchpoint matters overall),
        how = "attribution_values" uses the iET values (estimate_attribution must be called first).
        '''
        if how == 'attribution_values':
            weights = self._attribution_df['attribution_values']
        elif how == 'removal_effect':
            # recomputed every time (n+1 small linear solves) so it always matches the last estimate_proba
            weights = self.estimate_removal_effect()['removal_effect']
        else:
            raise ValueError('how must be "attribution_values" or "removal_effect"')

        return compute_attribution_weights( dataframe = dataframe, session_colname = self._session_colname, client_id_colname = self._client_id_colname,
                                            touchpoint_colname = self._touchpoint_colname, time_colname = self._time_colname, target_colname = self._target_colname,
                                            weights = weights, top_k_name = self._top_k_name )
