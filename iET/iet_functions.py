import pandas as pd
import numpy as np
from tqdm import tqdm

def count_source_medium( data, touchpoint, client_session_id ):
    return data.groupby(touchpoint).count()[[client_session_id]]   

def target_reached_probability( data, client_session_id, touchpoint, target):
    '''
    Compute the probability to be last touchpoint before the target

    1) Find unique client_session + touchpoint combinations
    2) Groups by touchpoint and count the session with target reached
    3) Compute the target probability
    
    Parameters:
    -----------
    data: dataframe
    client_session_id : unique ids for client session colname
    touchpoint: colname with the touchpoint label
    target: target colname
    
    Returns:
    ---------
    touchpoint_target_prob : dataframe with target probabilities
    '''

    #all occurences of all the combinations client/session + touchpoint 
    combinations = data.drop_duplicates( subset =  [ client_session_id, touchpoint ] )[[ client_session_id, touchpoint ]]
    target_combinations = data.drop_duplicates( subset =  [ client_session_id, touchpoint ], keep = 'last' ).loc[ data[target] != 0 ][[ client_session_id, touchpoint ]]

    base = combinations.groupby(touchpoint).count()[[client_session_id]]
    has_target = target_combinations.groupby(touchpoint).count()[[client_session_id]]

    touchpoint_target_prob = base.merge(has_target, left_index=True, right_index=True, how='left').fillna(0)
    touchpoint_target_prob.columns = ['base','has_target']
    touchpoint_target_prob['prob'] = touchpoint_target_prob['has_target'] / touchpoint_target_prob['base']

    return touchpoint_target_prob

def quit_probability( data, client_session_id, touchpoint, target):
    '''
    Compute the probability to be last touchpoint before quit 

    1) Find unique client_session + touchpoint combinations
    2) Groups by touchpoint and count the session with target not reached
    3) Compute the target probability
    
    Parameters:
    -----------
    data: dataframe
    client_session_id : unique ids for client session colname
    touchpoint: colname with the touchpoint label
    target: target colname
    
    Returns:
    ---------
    touchpoint_quit_prob : dataframe with quit probabilities
    '''

    #all occurences of all the combinations client/session + touchpoint for all the data and only for zero data
    combinations = data.drop_duplicates( subset =  [ client_session_id, touchpoint ] )[[ client_session_id, touchpoint ]]
    quit_combinations = data.drop_duplicates( subset =  [ client_session_id ], keep = 'last' ).loc[ data[target] == 0 ][[ client_session_id, touchpoint ]]

    base = combinations.groupby(touchpoint).count()[[client_session_id]]
    has_quit = quit_combinations.groupby(touchpoint).count()[[client_session_id]]

    touchpoint_quit_prob = base.merge(has_quit, left_index=True, right_index=True, how='left').fillna(0)
    touchpoint_quit_prob.columns = ['base','has_quit']
    touchpoint_quit_prob['prob'] = touchpoint_quit_prob['has_quit'] / touchpoint_quit_prob['base']

    return touchpoint_quit_prob

def estimate_transaction_matrix( data, client_id, client_session_id, touchpoint, target ):
    '''
    Compute transaction matrix probabilities

    Parameters:
    -----------
    data: dataframe
    client_session_id : unique ids for client session colname
    touchpoint: colname with the touchpoint label
    target: target colname
    
    Returns:
    ---------
    prob_transiction : dataframe with transaction probabilities
    count_transiction : dataframe with transaction count
    '''

    #remove consecuitve duplicates of the same session in the same touchpoint
    shifted = data.shift(1)
    client_touchpoint_path = data.loc[ np.logical_or( shifted[client_session_id] != data[client_session_id] , shifted[touchpoint] != data[touchpoint] ) ].reset_index(drop=True)
    #detect change in client_id 
    #TO READ: here we use only the client id not the client session id. In this way when there are session there is the possiblity to cambe back with the same touchpoint
    # when tere are no session the same user could not move to the same touchpoint 

    client_touchpoint_path['is_client_changed'] = client_touchpoint_path[client_id].shift(1, fill_value = client_touchpoint_path[client_id].head(1))  !=  \
                                                client_touchpoint_path[client_id]

    change_index = client_touchpoint_path.loc[client_touchpoint_path['is_client_changed']].index

    #add none where the client change, to easily remove that line from the count
    partial = pd.DataFrame(None, index= change_index - 0.5)
    partial[touchpoint] = None

    touchpoint_state = client_touchpoint_path[[touchpoint]].copy()
    touchpoint_state = touchpoint_state.append(partial).sort_index().reset_index(drop=True)

    #create the shift and drop the line with None: they are last status of the user 
    touchpoint_state['shift'] = touchpoint_state[touchpoint].shift(-1)
    touchpoint_state['count'] = 1
    touchpoint_state = touchpoint_state.dropna()

    #groupby and count the combinations touchpoint-shift
    count_transiction = touchpoint_state.groupby([touchpoint, 'shift']).count().unstack().fillna(0)
    count_transiction.columns = count_transiction.columns.get_level_values(1)
    count_transiction.index.name=None
    count_transiction.columns.name=None

    for tp in data[touchpoint].unique().tolist():
        if tp not in count_transiction.columns.tolist():
            count_transiction[tp] = 0

        if tp not in count_transiction.index.values:
            count_transiction.loc[tp] = np.zeros(count_transiction.shape[1])

    
    prob_transiction = pd.DataFrame( np.round(count_transiction.values / (count_transiction.sum(axis=1).values[:,None] + 0.000000001) ,4) ,\
                                            index = count_transiction.index, columns=count_transiction.columns )
    

    return prob_transiction, count_transiction

def move_probabilities(target_probabilities, quit_probabilitis):
    '''
    Find the probability to move to another touchpoint

    Returns:
    -------
    dataframe with the move probabilities
    '''
    return pd.DataFrame( np.round(1 - target_probabilities.prob - quit_probabilitis.prob, 4 ) )

def expected_target(data, target, touchpoint):
    '''
    Compute the expected target value
    
    Returns:
    --------
    dataframe with the expcted target value conditioned to target > 0
    '''
    return data.loc[ data[target] > 0 ].groupby( touchpoint ).mean()[ [target] ]

def merge_elements(target_df, quit_df, move_df, xT_df):
    '''
    Create unique df merging the result of target_reached_probability, quit_probability, move_probabilities, expected_target

    Returns:
    ----------
    final_df : dataframe
    '''

    final_df = target_df[['prob']].merge(quit_df[['prob']], left_index=True, right_index=True)
    final_df = final_df.merge(move_df, left_index=True, right_index=True)
    final_df = final_df.merge(xT_df, left_index=True, right_index=True)
    final_df.columns=['target_prob','quit_prob','move_prob','xT']

    return final_df

def calculate_attribution_values( final_df, prob_transiction, max_iter = 20, epsilon = 0.1 ):
    '''
    Calculate the attribution values and evaluate the convergences of the algorithm.
    The cicle end when max iter is reached or the differences between consecutive state sum is less than epsilon

    Parameters:
    ----------
    final_df : dataframe form merge_elments function
    prob_transiction : probabilities transaction matrix from estimate_transaction_matrix
    max_iter : int, max number of cycle
    epsilon: float, minimium differences sum to reach to stop the iteractions

    Returns:
    --------
    values_df: dataframe with all the attribution values state
    diff_df: dataframe with the story of differences

    '''
    tp_list = final_df.index.tolist()
    values_df = pd.DataFrame( np.zeros(len(tp_list)), index = tp_list, columns = ['v0'])
    diff_df = pd.DataFrame( np.zeros(len(tp_list)), index = tp_list, columns = ['dv0'])
    
    for itr in range(0,max_iter):
        
        values_df['v' + str( itr+1 )] = None
        
        for tp in tp_list:
            values_df.loc[tp, 'v' + str( itr+1 )] = final_df.loc[ tp,'target_prob' ]*final_df.loc[ tp,'xT' ] + \
            final_df.loc[ tp,'move_prob' ] * np.sum( values_df[ 'v'+ str(itr) ] * prob_transiction.loc[tp][tp_list] )
        
        diff_df['dv'+str(itr)] = values_df[ 'v' + str(itr+1) ] - values_df[ 'v' + str(itr) ]   

        if diff_df['dv'+str(itr)].sum() < epsilon:
            print('Convergence reached in ' +str(itr)+ ' passes')
            break

    return values_df, diff_df


    '''
    Calculate the attribution values and evaluate the convergences of the algorithm.
    The cicle end when max iter is reached or the differences between consecutive state sum is less than epsilon

    Parameters:
    ----------
    final_df : dataframe form merge_elments function
    prob_transiction : probabilities transaction matrix from estimate_transaction_matrix
    max_iter : int, max number of cycle
    epsilon: float, minimium differences sum to reach to stop the iteractions

    Returns:
    --------
    values_df: dataframe with all the attribution values state
    diff_df: dataframe with the story of differences

    '''
    tp_list = final_df.index.tolist()
    values_df = pd.DataFrame( np.zeros(len(tp_list)), index = tp_list, columns = ['v0'])
    diff_df = pd.DataFrame( np.zeros(len(tp_list)), index = tp_list, columns = ['dv0'])
    
    for itr in range(0,max_iter):
        
        values_df['v' + str( itr+1 )] = None
        
        for tp in tp_list:
            values_df.loc[tp, 'v' + str( itr+1 )] = final_df.loc[ tp,'target_prob' ]*final_df.loc[ tp,'xT' ] + \
            final_df.loc[ tp,'move_prob' ] * np.sum( values_df[ 'v'+ str(itr) ] * prob_transiction.loc[tp][tp_list] )
        
        diff_df['dv'+str(itr)] = values_df[ 'v' + str(itr+1) ] - values_df[ 'v' + str(itr) ]   

        if diff_df['dv'+str(itr)].sum() < epsilon:
            print('Convergence reached in ' +str(itr)+ ' passes')
            break

    return values_df, diff_df