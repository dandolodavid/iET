import pandas as pd
import numpy as np
from tqdm import tqdm
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from iET.iet_functions import *
from iET.compute_attribution_function import *

class iET():

    def __init__(self, touchpoint_colname, target_colname, client_id_colname, time_colname = None, session_colname = None, use_top_k = None, accept_consecutive_touchpoint = False):
        self._touchpoint_colname = touchpoint_colname
        self._target_colname = target_colname
        self._client_id_colname = client_id_colname
        self._session_colname = session_colname
        self._time_colname = time_colname
        self._top_k = use_top_k
        self._top_k_name = None
        self._accept_consecutive_touchpoint = accept_consecutive_touchpoint

    def estimate_proba(self, dataframe ):
        '''
        Prepare the dataframe and estimate all the required probabilities
        '''
        data = dataframe.copy()
        #if session is used we create a new key that is the joined of client id and session id
        if self._session_colname:
            data['client_session_id'] = data[self._client_id_colname].astype(str) +' : ' + data[self._session_colname].astype(str)
            self._client_session_colname = 'client_session_id'
        else:
            self._client_session_colname = self._client_id_colname
        
        index_name = data.index.name
        if index_name is  None:
            index_name = 'index'
        data = data.reset_index().sort_values(by =  [self._client_session_colname, index_name]).set_index(index_name)


        if self._top_k:
            top_touchpoint = count_source_medium(data, self._touchpoint_colname, self._client_session_colname)
            self._top_k_name = top_touchpoint.sort_values(self._client_session_colname).index.tolist()[-self._top_k:]
            data = data.loc[ data[self._touchpoint_colname].apply(lambda x: x in self._top_k_name) ]
        
        self._count_df = count_source_medium(data, self._touchpoint_colname, self._client_session_colname)

        #compute probability of reach the target
        target_df = target_reached_probability( data, self._client_session_colname, self._touchpoint_colname, self._target_colname)

        #compute quit probability
        quit_df = quit_probability( data, self._client_session_colname, self._touchpoint_colname, self._target_colname )

        #estimate transaction matrix
        self._transaction_df, self._count_transaction_df = estimate_transaction_matrix( data, self._client_id_colname, self._client_session_colname,  self._touchpoint_colname, self._target_colname, self._accept_consecutive_touchpoint )

        #compute move proabability
        move_df = move_probabilities(target_df, quit_df)

        #compute expected target
        xT_df = expected_target(data, self._target_colname, self._touchpoint_colname)

        #aggregate all the info and compute the attribution values
        self._final_df = merge_elements(target_df, quit_df, move_df, xT_df)
    
    def estimate_attribution(self, max_iter = 20, epsilon = 0.1):
        '''
        '''
        self._iter_solution, self._diff_df = calculate_attribution_values( self._final_df, self._transaction_df, max_iter , epsilon )
        attribution_df = pd.DataFrame( self._iter_solution.iloc[:,-1] ) 
        attribution_df.columns = ['attribution_values']
        self._attribution_df = attribution_df.sort_values(by = 'attribution_values', ascending = False)
        
        return self._attribution_df.copy()

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
        
        fig.update_layout(title={'text': 'Attribution values ( bar width is proportional to the number of users and sessions for that touchpoint )' })

        return fig
    
    def plot_proba(self):
        '''
        '''
        plot_df = self._final_df[['target_prob','quit_prob','move_prob']].stack().reset_index()
        plot_df.columns = [self._touchpoint_colname, 'hue', 'value']
        fig = px.bar(plot_df, x = self._touchpoint_colname, y = 'value', color = 'hue', title= 'Buy Quit and Move probability by channels')
        return fig
    
    def plot_transaction(self):
        '''
        '''
        z = self._transaction_df.values

        x = self._transaction_df.columns.tolist()
        y = self._transaction_df.index.tolist()

        z_text = self._transaction_df.astype(str).values
        
        fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale = 'bluered')
        fig['layout']['yaxis']['autorange'] = "reversed"
        
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
        
        fig.update_layout(title={'text': 'Expected target values ( bar width is proportional to the number of users and sessions for that touchpoint )' })

        return fig
    
    def compute_session(self, dataframe, max_delta_days = 7, additional_sorting_columns = None ):
        '''
        Compute the session based on the revenue and the days time delta. One session end when the user complete the target and a new session begin 
        from the successive action

        Parameters:
        ------------
        dataframe: pd.DataFrame
            dataframe
        max_delta_days: int
            max number of consecutive days without any actions to consider the end of the session
        additional_sorting_columns : [str]
            list of additional columns to use while sorting the dataframe (user_column + additional_sorting + time_column)
    

        Returns:
        -----------
        dataframe with session column added

        '''

        if additional_sorting_columns is None:
            additional_sorting_columns = []
        
        if self._session_colname is None:
            raise AttributeErrore('Parameter "session_colname" is required to compute session. Please set a valid string for the colname')
        
        sorting_columns = [self._client_id_colname ] + additional_sorting_columns + [self._time_colname]
        grouping_columns = [self._client_id_colname ] + additional_sorting_columns
        
        data = dataframe.sort_values(sorting_columns).copy().reset_index()
        
        for asc in additional_sorting_columns:
            data[asc] = data[asc].fillna('NA')
        
        data[self._time_colname] = pd.to_datetime(data[self._time_colname])
        
        data['time_delta'] = data.groupby(grouping_columns)[self._time_colname].diff().fillna(pd.Timedelta(seconds=0))
        data['previus_value'] = data.groupby(self._client_id_colname)[self._target_colname].shift().fillna(0)
        
        change_session = data.loc[np.logical_or(    data['previus_value']!=0, #if on the preivus line, the user has bought
                                                    data['time_delta'].dt.days > max_delta_days  #if the time from the las action
                                                )].groupby(self._client_id_colname ).cumcount() +2

        data[self._session_colname] = None
        data.loc[data.drop_duplicates(self._client_id_colname ,keep='first').index,self._session_colname] = 1
        data.loc[change_session.index,self._session_colname] = change_session
        data[self._session_colname] = data[self._session_colname].fillna(method='ffill')

        data = data.set_index('index')
        data.index.name = None
        self._session_colname = self._session_colname

        return data

    def compute_attribution(self,dataframe, how='attribution_values'):
        '''
        Calculate the attribution for each line of the dataset.
        It accepts different methods : "step_proba" and "attribution_value" (default)
        Step probability compute the probability of see each step and use them to calculate the weight for the attribution
        Attribution value assigne to each touchpoint a weight related to the 
        '''

        if how == 'attribution_values':
           out = compute_attribution_iet_values( dataframe = dataframe, session_colname = self._session_colname, client_id_colname = self._client_id_colname, 
                                                touchpoint_colname = self._touchpoint_colname, time_colname = self._time_colname, target_colname = self._target_colname, 
                                                iet_values_df = self._attribution_df, top_k = self._top_k, top_k_name = self._top_k_name )
        if how == 'step_proba':
            out = compute_attribution_step_probability( dataframe = dataframe, session_colname = self._session_colname, client_id_colname = self._client_id_colname, 
                                                touchpoint_colname = self._touchpoint_colname, time_colname = self._time_colname, target_colname = self._target_colname,
                                                transition_df = self._transaction_df, proba_df = self._final_df, top_k = self._top_k, top_k_name = self._top_k_name )
                
        return out