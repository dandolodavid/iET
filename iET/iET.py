import pandas as pd
import numpy as np
from tqdm import tqdm
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from iET.iet_functions import *

class iET():

    def __init__(self, touchpoint_colname, target_colname, client_id_colname, time_colname = None, session_colname = None, use_top_k = None, accept_consecutive_touchpoint = False):
        self._touchpoint_colname = touchpoint_colname
        self._target_colname = target_colname
        self._client_id_colname = client_id_colname
        self._session_colname = session_colname
        self._time_colname = time_colname
        self._top_k = use_top_k
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
        Compute the session based on the revenue. One session end when the user complete the target and a new session begin 
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


    def compute_attribution(self,dataframe):
        '''
        Calculate the attribution for each line of the dataset.
        For each user we compute the probability associated to his journey, with moving probs unitl we reach the target touchpoint
        The probs are then normalized and multiplied for the target value
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
        data = data.reset_index().sort_values(by = [self._client_session_colname, index_name]).set_index(index_name)

        if self._top_k:
            data = data.loc[ data[self._touchpoint_colname].apply(lambda x: x in self._top_k_name) ]
            
        # extract only custumers with transactions
        client_session_with_transaction = data.loc[data[self._target_colname]>0][self._client_session_colname].unique()
        
        # find the next touchpoint for each client_session, if it is the last touchpoint, "next" will be None
        data['next'] = data.groupby(self._client_session_colname)[self._touchpoint_colname].shift(-1)
        
        # extract the users touchpoint journey
        movements_df = data[ [self._client_session_colname, self._time_colname, self._touchpoint_colname ] + ['next'] ]
        movements_df = movements_df.loc[movements_df[self._client_session_colname].isin(client_session_with_transaction)]
        
        # extract the transition matrix and stack it in 2 columns
        stacked_trm = self._transaction_df.stack().reset_index().rename( columns={'level_0':'start','level_1':'end',0:'transition_prob'} )
        
        # merge the probability of transition 
        original_index = movements_df.index
        movements_df = movements_df.merge(stacked_trm, how = 'left', left_on=[self._touchpoint_colname,'next'], right_on=['start','end'])

        # merge the probability of moving and buy
        movements_df['action'] = 'move'
        movements_df.loc[movements_df.end.isna(),'action'] = 'buy'
        
        
        
        movements_df = movements_df.merge(self._final_df[['target_prob','move_prob']],how='left',left_on=[self._touchpoint_colname], right_index=True)
        movements_df[['target_prob','move_prob']] = movements_df[['target_prob','move_prob']] .fillna(0)
        movements_df.loc[movements_df.action=='buy','move_prob'] = 0
        movements_df.loc[movements_df.action=='buy','transition_prob'] = 0
        
        # compute the weights : if buy -> target_proba, if move -> move_proba*transition_proba
        movements_df['weight'] = movements_df['target_prob'] + movements_df['move_prob']*movements_df['transition_prob']
        
        # add for each client_session the total weight
        history_df = movements_df.merge( movements_df.groupby(self._client_session_colname)['weight'].sum().reset_index().rename(columns={'weight':'sum_weight'}), left_on=self._client_session_colname, right_on=self._client_session_colname )
        history_df['norm_weight'] = history_df['weight']/history_df['sum_weight']
        
        # add for each client_session the transaction value
        if self._session_colname is None:
            transaction_val = data.loc[data[self._target_colname]>0][[self._client_id_colname,self._target_colname]]
        else:
            transaction_val = data.loc[data[self._target_colname]>0][[self._client_id_colname,self._session_colname,self._target_colname]]
            transaction_val[self._client_session_colname] = transaction_val[self._client_id_colname] + ' : ' + transaction_val[self._session_colname].astype(str)
        
        history_df = history_df.merge(transaction_val[[self._client_session_colname, self._target_colname]], left_on=self._client_session_colname, right_on=self._client_session_colname)
        history_df['iET_attribution'] = history_df[self._target_colname]*history_df['norm_weight']
        history_df.index = original_index
        
        out = dataframe.copy()
        out = out.merge(history_df[['action','move_prob','target_prob','norm_weight','iET_attribution']], left_index=True,right_index=True, how='left')
                
        return out