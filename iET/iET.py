import pandas as pd
import numpy as np
from tqdm import tqdm
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from iET.ier_functions import *

class iET():

    def __init__(self, touchpoint_colname, target_colname, client_id_colname, session_colname = None, use_top_k = None):
        self._touchpoint_colname = touchpoint_colname
        self._target_colname = target_colname
        self._client_id_colname = client_id_colname
        self._session_colname = session_colname
        self._top_k = use_top_k

    def estimate_proba(self, dataframe ):
        '''
        Prepare the dataframe and estimate all the required probabilities
        '''
        data = dataframe.copy()

        #if session is used we create a new key that is the joined of client id and session id
        if self._session_colname:
            dataframe['client_session_id'] = dataframe[self._client_id_colname].astype(str) +' : ' + dataframe[self._session_colname].astype(str)
            self._client_session_colname = 'client_session_id'
        else:
            self._client_session_colname = self._client_id_colname

        if self._top_k:
            top_touchpoint = count_source_medium(dataframe, self._touchpoint_colname, self)
            top_k = count_source_medium.sort_values('cdp_id').tail(self._top_k).index
            data = data.loc[ data[self._touchpoint_colname].apply(lambda x: x in top_touchpoint) ]
            
        self._count_df = count_source_medium(data, self._touchpoint_colname, self._client_session_colname)

        #compute probability of reach the target
        target_df = target_reached_probability( data, self._client_session_colname, self._touchpoint_colname, self._target_colname)

        #compute quit probability
        quit_df = quit_probability( data, self._client_session_colname, self._touchpoint_colname, self._target_colname )

        #estimate transaction matrix
        self._transaction_df, self._count_transaction_df = estimate_transaction_matrix( data, self._client_session_colname, self._client_id_colname, self._touchpoint_colname, self._target_colname )

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
    
    def proba_df(self):
        return self._final_df
    


    



        
        

