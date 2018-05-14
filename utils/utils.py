import numpy as np
import pandas as pd
import os
from keras.preprocessing import sequence

def preProBuildWordVocab(sentence_iterator, word_count_threshold=0):
    print('preprocessing word counts and creating vocab based on word count threshold %d' % word_count_threshold)
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        for w in sent.lower().split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print('filtered words from %d to %d' % (len(word_counts), len(vocab)))

    ixtoword = {}
    ixtoword[0] = '<pad>'
    ixtoword[1] = '<bos>'
    ixtoword[2] = '<eos>'
    ixtoword[3] = '<unk>'

    wordtoix = {}
    wordtoix['<pad>'] = 0
    wordtoix['<bos>'] = 1
    wordtoix['<eos>'] = 2
    wordtoix['<unk>'] = 3

    for idx, w in enumerate(vocab):
        wordtoix[w] = idx + 4
        ixtoword[idx+4] = w

    word_counts['<pad>'] = nsents
    word_counts['<bos>'] = nsents
    word_counts['<eos>'] = nsents
    word_counts['<unk>'] = nsents

    bias_init_vector = np.array([1.0 * word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range

    return wordtoix, ixtoword, bias_init_vector

def prepare_data(video_data_path, video_path, video_feat_path, split, dict_path):
    data = get_video_data(video_data_path, video_path, video_feat_path, split)
    captions = data['Description'].values
    captions_list = list(captions)
    captions = np.array(captions_list, dtype=np.object)
    captions = preprocess_caption(captions)

    if split == 'train':
        if os.path.isfile(os.path.join(dict_path, 'wordtoix.npy')):
            wordtoix = pd.Series(np.load(os.path.join(dict_path, 'wordtoix.npy')).tolist())
            ixtoword = pd.Series(np.load(os.path.join(dict_path, 'ixtoword.npy')).tolist())
            bias_init_vector = np.load(os.path.join(dict_path, 'bias_init_vector.npy'))
        else:
            wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions)
            np.save(os.path.join(dict_path, 'wordtoix.npy'), wordtoix)
            np.save(os.path.join(dict_path, 'ixtoword.npy'), ixtoword)
            np.save(os.path.join(dict_path, 'bias_init_vector.npy'), bias_init_vector)
        return data, captions, wordtoix, ixtoword, bias_init_vector
    else:
        data = data.groupby('video_feat_path').apply(lambda x: x.iloc[0])
        data = data.reset_index(drop=True)
        videos = data['video_feat_path'].unique()
        captions = data['Description'].values
        captions_list = list(captions)
        captions = np.array(captions_list, dtype=np.object)
        captions = preprocess_caption(captions)
        try:
            wordtoix = pd.Series(np.load(os.path.join(dict_path, 'wordtoix.npy')).tolist())
            ixtoword = pd.Series(np.load(os.path.join(dict_path, 'ixtoword.npy')).tolist())
            bias_init_vector = np.load(os.path.join(dict_path, 'bias_init_vector.npy'))
        except EOFError:
            print('Error: files not found...')
        return data, videos, captions, wordtoix, ixtoword, bias_init_vector

def get_video_data(video_data_path, video_path, video_feat_path, split):
    video_path = os.path.join(video_path, split)

    video_data = pd.read_csv(video_data_path, sep=',')
    video_data = video_data[video_data['Language'] == 'English']
    video_data['video_id'] = video_data.apply(lambda row: row['VideoID']+'_'+str(row['Start'])+'_'+str(row['End'])+'.avi', axis=1)
    video_data['video_path'] = video_data['video_id'].map(lambda x: os.path.join(video_path, x))
    video_data['video_feat_path'] = video_data['video_id'].map(lambda x: os.path.join(video_feat_path, x))

    video_data = video_data[video_data['video_path'].map(lambda x: os.path.exists( x ))]
    video_data = video_data[video_data['Description'].map(lambda x: isinstance(x, str))]

    video_data['video_feat_path'] = video_data['video_feat_path'].apply(lambda x: x + '.npy')

    video_data = video_data[video_data['video_feat_path'].map(lambda x: os.path.exists( x ))]

    video_filenames = video_data['video_path'].unique()

    data = video_data[video_data['video_path'].map(lambda x: x in video_filenames)]

    return data

def convert2sen(ixtoword, index):
    generated_words = [ixtoword[x] for x in index]
    punctuation = np.argmax(np.array(generated_words) == '<eos>') + 1
    generated_words = generated_words[:punctuation]

    generated_sentence = ' '.join(generated_words)
    generated_sentence = generated_sentence.replace('<bos> ', '')
    generated_sentence = generated_sentence.replace(' <eos>', '')

    return generated_sentence

def preprocess_caption(captions):
    captions = list(map(lambda x: x.replace('.', ''), captions))
    captions = list(map(lambda x: x.replace(',', ''), captions))
    captions = list(map(lambda x: x.replace('"', ''), captions))
    captions = list(map(lambda x: x.replace('\n', ''), captions))
    captions = list(map(lambda x: x.replace('?', ''), captions))
    captions = list(map(lambda x: x.replace('!', ''), captions))
    captions = list(map(lambda x: x.replace('\\', ''), captions))
    captions = list(map(lambda x: x.replace('/', ''), captions))
    return captions

def fetch_batch_data(current_train_data, start, end, train_opt, wordtoix, phase):
    '''organize batch data during different training phases'''
    if phase == 'reference' or phase == 'target':
        batch_size = train_opt.batch_size
    elif phase == 'CL':
        batch_size = train_opt.batch_size*train_opt.neg_K
    else:
        print('Phase Error...')
        return

    current_batch = current_train_data[start:end]
    current_videos = current_batch['video_feat_path'].values
    current_captions = current_batch['Description'].values

    current_feats = np.zeros((batch_size, train_opt.n_video_step, train_opt.ctx_shape[1]))
    current_feats_vals = list(map(lambda vid: np.load(vid), current_videos))
    current_video_masks = np.zeros((batch_size, train_opt.n_video_step))

    for ind, feats in enumerate(current_feats_vals):
        current_feats[ind][:len(current_feats_vals[ind])] = feats
        current_video_masks[ind][:len(current_feats_vals[ind])] = 1

    current_captions = preprocess_caption(current_captions)

    for idx, each_cap in enumerate(current_captions):
        word = each_cap.lower().split(' ')
        if len(word) < train_opt.n_caption_lstm_step:
            current_captions[idx] = current_captions[idx] + ' <eos>'
        else:
            new_word = ''
            for i in range(train_opt.n_caption_lstm_step-1):
                new_word = new_word + word[i] + ' '
            current_captions[idx] = new_word + '<eos>'

    current_caption_ind = []
    for cap in current_captions:
        current_word_ind = []
        for word in cap.lower().split(' '):
            if word in wordtoix:
                current_word_ind.append(wordtoix[word])
            else:
                current_word_ind.append(wordtoix['<unk>'])
        current_caption_ind.append(current_word_ind)

    current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=train_opt.n_caption_lstm_step)
    current_caption_matrix = np.hstack([current_caption_matrix, np.zeros([len(current_caption_matrix), 1])]).astype(int)
    current_caption_masks = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
    nonzeros = np.array(list(map(lambda x: (x!=0).sum() + 1, current_caption_matrix)))

    for ind, row in enumerate(current_caption_masks):
        row[:nonzeros[ind]] = 1

    return current_caption_matrix, current_caption_masks, current_feats
