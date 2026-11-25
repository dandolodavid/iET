from iET.iet_functions import prepare_data


def compute_attribution_weights(dataframe, session_colname, client_id_colname, touchpoint_colname, time_colname, target_colname, weights, top_k_name=None):
    '''
    Calculate the attribution for each line of the dataset.
    The total target of each journey is split among its lines proportionally to the weight of their touchpoint
    (iET values or removal effects). Journeys without target get 0; journeys whose weights are all 0 split evenly.

    Returns:
    ---------
    copy of dataframe (same index and order) with norm_weight and iET_attribution columns
    '''
    data, key = prepare_data(dataframe, client_id_colname, session_colname, touchpoint_colname, time_colname, top_k_name)
    groups = data.groupby(key)
    total = groups[target_colname].transform('sum')  # sum handles multiple targets in the same journey
    weight = data[touchpoint_colname].map(weights).fillna(0)
    weight_sum = weight.groupby(data[key]).transform('sum')
    norm_weight = (weight / weight_sum).where(weight_sum != 0, 1 / groups[key].transform('size')).where(total != 0, 0)

    out = dataframe.copy()
    out['norm_weight'] = norm_weight.sort_index().to_numpy()
    out['iET_attribution'] = (norm_weight * total).sort_index().to_numpy()
    return out
