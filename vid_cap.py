import argparse
from opts import opts
from train import train
from test import test

train_opt = opts(epochs_m=900, epochs_n=900, epochs_cl=100, batch_size=50, learning_rate=0.0001,
                dim_embed=1000, dim_ctx=4096, dim_hidden=1000, n_video_step=40, ctx_shape=4096,
                n_caption_lstm_step=20, beam_size=3, hypo_size=5, neg_K=5, KN=3, PN=10,
                pretrained_graph_m='model-900.meta', pretrained_graph_n='model-1800.meta',
                pretrained_graph_cl='model-1900.meta',
                test_graph='model-900', video_path='./data/msvd/videos',
                video_feat_path='./data/msvd/feats/fc7',
                video_data_path='./data/msvd/video_corpus.csv',
                model_path='./models/msvd/fc7/', dict_path='./data/msvd')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for the model')
    parser.add_argument('phase', type=str, help='choice of the phase, train or test')
    args = parser.parse_args()
    if args.phase == 'train':
        train(train_opt)
    elif args.phase == 'test':
        test(train_opt)
