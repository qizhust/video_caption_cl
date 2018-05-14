import numpy as np

def generate_neg_samples(data, phase, neg_K=None):
    '''
    prepare mismatched pairs <VideoID, Description> with provided data,
    no need for repeat
    '''

    if phase == 'reference':
        index = list(data.index)
        np.random.shuffle(index)
        data = data.ix[index]

        current_data = data.groupby('video_feat_path').apply(lambda x: x.iloc[np.random.choice(len(x))])
        # current_data = data_n.groupby('video_feat_path').apply(lambda x: x.iloc[0])
        current_data = current_data.reset_index(drop=True)

        # generate single negative sample according to each positive sample
        for idx in range(len(current_data)):
            # find out all rows without current video_id
            current_feat_path = current_data.iloc[idx].video_feat_path
            items = current_data[current_data['VideoID']!=current_data.iloc[idx].VideoID]
            # use row instead of video_feat_path as index
            # current_data.loc[current_data.video_feat_path==current_feat_path, 'Description'] = items.iloc[np.random.choice(len(items))].Description
            current_data.loc[idx, 'Description'] = items.iloc[np.random.choice(len(items))].Description
    elif phase == 'CL':
        # generate neg_K negative samples according to each positive sample
        current_data = data.iloc[np.repeat(np.arange(len(data)), neg_K)]
        current_data = current_data.reset_index(drop=True)
        for idx in range(len(data)):
            current_feat_path = data.iloc[idx].video_feat_path
            items = data[data['VideoID']!=data.iloc[idx].VideoID]
            items = items.reset_index(drop=True)
            mis_desc = items.loc[np.random.choice(len(items), neg_K), 'Description']
            current_data.loc[idx*neg_K:(idx+1)*neg_K-1, 'Description'] = mis_desc.tolist()
    else:
        print('Phase Error...')
        return
    return current_data

def generate_cl_samples(data, neg_K):
    '''
    prepare positive samples, i.e. original data repeated for neg_K times;
    prepare negative samples, i.e. neg_K mismatched pairs for each positive sample.
    '''
    
    index = list(data.index)
    np.random.shuffle(index)
    data = data.ix[index]
    current_data = data.groupby('video_feat_path').apply(lambda x: x.iloc[np.random.choice(len(x))])

    # positive data: repeat original data for neg_K times
    pos_data = current_data.iloc[np.repeat(np.arange(len(current_data)), neg_K)]
    pos_data = pos_data.reset_index(drop=True)
    # negative data: call generate_neg_samples(positive data)
    neg_data = generate_neg_samples(data=current_data, phase='CL', neg_K=neg_K)
    return pos_data, neg_data
