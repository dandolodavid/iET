import pandas as pd
import numpy as np
from tqdm import tqdm
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from iET.iet_functions import *

class iET():

    def __init__(self, touchpoint_colname, target_colname, client_id_colname, session_colname = None, use_top_k = None, accept_consecutive_touchpoint = False):
        self._touchpoint_colname = touchpoint_colname
        self._target_colname = target_colname
        self._client_id_colname = client_id_colname
        self._session_colname = session_colname
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
        data = data.reset_index().sort_values(by =  [self._client_session_colname, index_name]).set_index(index_name)

        if self._top_k:
            data = data.loc[ data[self._touchpoint_colname].apply(lambda x: x in self._top_k_name) ]
        
        out = data.copy()
        out['iET_attribution'] = 0

        for user_id in tqdm(data[self._client_session_colname].unique()):
            
            tmp = data.loc[ data[self._client_session_colname] == user_id ].copy()

            if tmp[self._target_colname].sum() > 0:
                tmp = tmp.loc[(tmp.shift(-1)[[self._client_session_colname, self._touchpoint_colname]] != tmp[[self._client_session_colname, self._touchpoint_colname]]).any(axis=1)]
                
                t_probs = []
                m_probs = []
                target_probs = []
                
                for i in range(0,tmp.shape[0]):
                    if i == tmp.shape[0] - 1:
                        target_probs.append( self._final_df.loc[tmp.iloc[i][self._touchpoint_colname]]['target_prob'] )
                    else:
                        t_probs.append( self._transaction_df.loc[ tmp.iloc[i][self._touchpoint_colname] ][tmp.iloc[i+1][self._touchpoint_colname]] )
                        m_probs.append( self._final_df.loc[tmp.iloc[i][self._touchpoint_colname]].move_prob )

                weights = list( np.array(t_probs) * np.array(m_probs) ) + target_probs
                
                attribution_values =  weights/np.sum(weights) * tmp[self._target_colname].sum()
                out.loc[ tmp.index.tolist(), 'iET_attribution' ] = attribution_values

        return out

    def compute_session_based_on_target(self, dataframe):
        '''
        Compute the session based on the revenue. One session end when the user complete the target and a new session begin 
        from the successive action

        Parameters:
        ------------
        dataframe: pd.DataFrame

        Returns:
        -----------
        dataframe with session column added

        '''

        session_list = []
        index_list = []
        for user_id in tqdm(dataframe[self._client_id_colname].unique()):
            ses = 1
            tmp = dataframe.loc[dataframe[self._client_id_colname] == user_id ]
            idxs = tmp.loc[tmp[self._target_colname] > 0 ].index
            
            user_session = []
            
            if len(idxs)>0:
                for i in range(0,len(idxs)):
                    if i == 0:
                        new_ses = list( np.repeat(ses, len(tmp.loc[ : idxs[i] ] )) ) 
                    else:
                        new_ses = list( np.repeat(ses, len(tmp.loc[ idxs[i-1]+1 : idxs[i] ])) )
                    
                    user_session = user_session + new_ses
                    ses = ses+1
                    
                    if i == len(idxs)-1 and len(user_session) < len(tmp) :
                        new_ses = list( np.repeat(ses, len(tmp.loc[ idxs[i]+1 : ])) )
                        user_session = user_session + new_ses  
            else:
                user_session = user_session + list( np.repeat(ses, len(tmp) ) )
            
            index_list = index_list + tmp.index.tolist()
            session_list = session_list + user_session
                    
        session_df = pd.DataFrame( { 'session':session_list }, index= index_list )

        self._session_colname = 'session'

        return dataframe.merge(session_df, left_index = True, right_index = True, how='left').fillna(1)
        



            
            

        
        

