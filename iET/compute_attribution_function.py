def compute_attribution_step_probability(dataframe, session_colname, client_id_colname, touchpoint_colname, time_colname, target_colname, transition_df, proba_df, top_k, top_k_name ):
    '''
    Calculate the attribution for each line of the dataset.
    For each user we compute the probability associated to his journey, with moving probs unitl we reach the target touchpoint
    The probs are then normalized and multiplied for the target value
    '''
    
    data = dataframe.copy()
    #if session is used we create a new key that is the joined of client id and session id
    if session_colname:
        data['client_session_id'] = data[client_id_colname].astype(str) +' : ' + data[session_colname].astype(str)
        client_session_colname = 'client_session_id'
    else:
        client_session_colname = client_id_colname
    
    index_name = data.index.name
    if index_name is  None:
        index_name = 'index'
    data = data.reset_index().sort_values(by = [client_session_colname, index_name]).set_index(index_name)

    if top_k:
        data = data.loc[ data[touchpoint_colname].apply(lambda x: x in top_k_name) ]
        
    # extract only custumers with transactions
    client_session_with_transaction = data.loc[data[target_colname]>0][client_session_colname].unique()
    
    # find the next touchpoint for each client_session, if it is the last touchpoint, "next" will be None
    data['next'] = data.groupby(client_session_colname)[touchpoint_colname].shift(-1)
    
    # extract the users touchpoint journey
    movements_df = data[ [client_session_colname, time_colname, touchpoint_colname ] + ['next'] ]
    movements_df = movements_df.loc[movements_df[client_session_colname].isin(client_session_with_transaction)]
    
    # extract the transition matrix and stack it in 2 columns
    stacked_trm = transition_df.stack().reset_index().rename( columns={'level_0':'start','level_1':'end',0:'transition_prob'} )
    
    # merge the probability of transition 
    original_index = movements_df.index
    movements_df = movements_df.merge(stacked_trm, how = 'left', left_on=[touchpoint_colname,'next'], right_on=['start','end'])
    import pdb;pdb.set_trace()
    # merge the probability of moving and buy
    movements_df['action'] = 'move'
    movements_df.loc[movements_df.end.isna(),'action'] = 'buy'
    
    movements_df = movements_df.merge(proba_df[['target_prob','move_prob']],how='left',left_on=[touchpoint_colname], right_index=True)
    movements_df[['target_prob','move_prob']] = movements_df[['target_prob','move_prob']].fillna(0)
    movements_df.loc[movements_df.action=='buy','move_prob'] = 0
    movements_df.loc[movements_df.action=='buy','transition_prob'] = 0
    
    # compute the weights : if buy -> target_proba, if move -> move_proba*transition_proba
    movements_df['weight'] = movements_df['target_prob'] + movements_df['move_prob']*movements_df['transition_prob']
    
    # add for each client_session the total weight
    history_df = movements_df.merge( movements_df.groupby(client_session_colname)['weight'].sum().reset_index().rename(columns={'weight':'sum_weight'}), left_on=client_session_colname, right_on=client_session_colname )
    history_df['norm_weight'] = history_df['weight']/history_df['sum_weight']
    
    # add for each client_session the transaction value
    if session_colname is None:
        transaction_val = data.loc[data[target_colname]>0][[client_id_colname,target_colname]]
    else:
        transaction_val = data.loc[data[target_colname]>0][[client_id_colname,session_colname,target_colname]]
        transaction_val[client_session_colname] = transaction_val[client_id_colname] + ' : ' + transaction_val[session_colname].astype(str)
    
    history_df = history_df.merge(transaction_val[[client_session_colname, target_colname]], left_on=client_session_colname, right_on=client_session_colname)
    history_df['iET_attribution'] = history_df[target_colname]*history_df['norm_weight']
    history_df.index = original_index
    
    out = dataframe.copy()
    out = out.merge(history_df[['action','move_prob','target_prob','transition_prob','norm_weight','iET_attribution']], left_index=True,right_index=True, how='left')
    out['action'] = out['action'].fillna('no target')
    out[['move_prob','target_prob','transition_prob','norm_weight','iET_attribution']] = out[['move_prob','target_prob','transition_prob','norm_weight','iET_attribution']].fillna(0)
    
    return out

def compute_attribution_iet_values(dataframe, session_colname, client_id_colname, touchpoint_colname, time_colname, target_colname, iet_values_df, top_k, top_k_name ):
    '''
    Calculate the attribution for each line of the dataset.
    To distribuate the value of the transaction we use the iet values calculated normalized.
    '''
    
    data = dataframe.copy()
    #if session is used we create a new key that is the joined of client id and session id
    if session_colname:
        data['client_session_id'] = data[client_id_colname].astype(str) +' : ' + data[session_colname].astype(str)
        client_session_colname = 'client_session_id'
    else:
        client_session_colname = client_id_colname
    
    index_name = data.index.name
    if index_name is  None:
        index_name = 'index'
    data = data.reset_index().sort_values(by = [client_session_colname, index_name]).set_index(index_name)

    if top_k:
        data = data.loc[ data[touchpoint_colname].apply(lambda x: x in top_k_name) ]
        
    # extract only custumers with transactions
    client_session_with_transaction = data.loc[data[target_colname]>0][client_session_colname].unique()
    data = data.loc[data[client_session_colname].isin(client_session_with_transaction)]
    
    #merge the attibution values
    history_df = data.merge(iet_values_df, left_on=touchpoint_colname,right_index=True, how='left').drop(columns=target_colname)
    total_weight = history_df.groupby([client_session_colname])[['attribution_values']].sum().reset_index().rename(columns={'attribution_values':'weight'})
    history_df = history_df.merge(total_weight, left_on = client_session_colname, right_on = client_session_colname, how='left')
    history_df['norm_weight'] = history_df['attribution_values']/history_df['weight']
    
    
    # add for each client_session the transaction value
    if session_colname is None:
        transaction_val = data.loc[data[target_colname]>0][[client_id_colname,target_colname]]
    else:
        transaction_val = data.loc[data[target_colname]>0][[client_id_colname,session_colname,target_colname]]
        transaction_val[client_session_colname] = transaction_val[client_id_colname] + ' : ' + transaction_val[session_colname].astype(str)


    history_df = history_df.merge(transaction_val[[client_session_colname, target_colname]], left_on=client_session_colname, right_on=client_session_colname, how='left')
    history_df['iET_attribution'] = history_df[target_colname]*history_df['norm_weight']
    history_df.index = data.index
    
    out = dataframe.copy()
    out = out.merge(history_df[['norm_weight','iET_attribution']], left_index=True,right_index=True, how='left')
    out[['norm_weight','iET_attribution']] = out[['norm_weight','iET_attribution']].fillna(0)

    return out