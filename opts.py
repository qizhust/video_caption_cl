class opts(object):
    def __init__(self, epochs_m, epochs_n, epochs_cl, batch_size, learning_rate,
                dim_embed, dim_ctx, dim_hidden, n_video_step, ctx_shape, n_caption_lstm_step,
                beam_size, hypo_size, neg_K, KN, PN, pretrained_graph_m,
                pretrained_graph_n, pretrained_graph_cl, test_graph,
                video_path, video_feat_path, video_data_path, model_path, dict_path):
        super(opts, self).__init__()
        self.epochs_m = epochs_m
        self.epochs_n = epochs_n
        self.epochs_cl = epochs_cl
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dim_embed = dim_embed
        self.dim_ctx = dim_ctx
        self.dim_hidden = dim_hidden
        self.n_video_step = n_video_step
        self.ctx_shape = [n_video_step, ctx_shape]
        self.n_caption_lstm_step = n_caption_lstm_step
        self.beam_size = beam_size
        self.hypo_size = hypo_size
        self.neg_K = neg_K
        self.KN = KN
        self.PN = PN
        self.pretrained_graph_m = pretrained_graph_m
        self.pretrained_graph_n = pretrained_graph_n
        self.pretrained_graph_cl = pretrained_graph_cl
        self.test_graph = test_graph
        self.video_path = video_path
        self.video_feat_path = video_feat_path
        self.video_data_path = video_data_path
        self.model_path = model_path
        self.dict_path = dict_path
