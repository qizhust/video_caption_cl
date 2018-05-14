import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import copy
from utils.utils import prepare_data, convert2sen

def test(opt=train_opt):
    test_data, test_videos, captions, wordtoix, ixtoword, bias_init_vector = prepare_data(
            train_opt.video_data_path, train_opt.video_path, train_opt.video_feat_path,
            'val', train_opt.dict_path)

    n_words = len(ixtoword)
    print('vocabulary size is %d' % n_words)

    sess = tf.InteractiveSession()
    model = Video_Caption_Generator(
            n_words=n_words,
            dim_embed=train_opt.dim_embed,
            dim_ctx=train_opt.dim_ctx,
            dim_hidden=train_opt.dim_hidden,
            n_caption_lstm_step=train_opt.n_caption_lstm_step,
            batch_size=train_opt.batch_size,
            ctx_shape=train_opt.ctx_shape)

    saver = tf.train.Saver()
    saver.restore(sess, os.path.join(train_opt.model_path, train_opt.test_graph))

    def beam_step(logprobsf, beam_size, hypo_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, h_rec, c_rec):
        ys = np.sort(logprobsf)
        ys[0] = ys[0,::-1]
        ix = np.argsort(logprobsf)
        ix[0] = ix[0,::-1]
        candidates = []
        cols = ys.shape[1]
        rows = beam_size if t > 0 else 1
        for col in range(cols):
            for row in range(rows):
                local_logprob = ys[row, col]
                candidate_logprob = beam_logprobs_sum[row] + local_logprob
                candidates.append({'c':ix[row,col], 'q':row, 'p':candidate_logprob, 'r':local_logprob})
        candidates = sorted(candidates, key=lambda x: -x['p'])

        new_h = h_rec.copy()
        new_c = c_rec.copy()
        if t >= 1:
            beam_seq_prev = beam_seq[:t].copy()
            beam_seq_logprobs_prev = beam_seq_logprobs[:t].copy()
        for vix in range(beam_size):
            v = candidates[vix]
            if t >= 1:
                beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
                beam_seq_logprobs[:t, vix] = beam_seq_logprobs_prev[:, v['q']]
            for state_ix in range(len(new_h)):
                new_h[state_ix][:, vix] = h_rec[state_ix][:, v['q']]
                new_c[state_ix][:, vix] = c_rec[state_ix][:, v['q']]
            beam_seq[t, vix] = v['c']
            beam_seq_logprobs[t, vix] = v['r']
            if beam_logprobs_sum[vix] != -1000:
                beam_logprobs_sum[vix] = v['p']
            else:
                beam_logprobs_sum[vix] = beam_logprobs_sum[vix]
        h_rec = new_h
        c_rec = new_c
        return beam_seq, beam_seq_logprobs, beam_logprobs_sum, h_rec, c_rec, candidates

    file_index = train_opt.test_graph
    file_index = file_index[file_index.find('-')+1::]
    output_txt_gs_fd = open('./result/greedy_'+file_index+'.txt', 'w')
    output_txt_bs_fd = open('./result/beam_'+file_index+'.txt', 'w')
    output_txt_bsm_fd = open('./result/beam_mul_'+file_index+'.txt', 'w')

    for idx, video_feat_path in enumerate(test_videos):
        start_time = time.time()
        print(idx, video_feat_path, captions[idx])
        video_feat = np.load(video_feat_path)[None,...]

        beam_seq = np.zeros((train_opt.n_caption_lstm_step, train_opt.beam_size)).astype(np.int32)
        beam_seq_logprobs = np.zeros((train_opt.n_caption_lstm_step, train_opt.beam_size))
        beam_logprobs_sum = np.zeros(train_opt.beam_size)
        ## to record previous h & c
        h_rec = np.zeros((train_opt.n_caption_lstm_step, train_opt.dim_hidden, train_opt.beam_size)).astype(np.float32)
        c_rec = np.zeros((train_opt.n_caption_lstm_step, train_opt.dim_hidden, train_opt.beam_size)).astype(np.float32)
        done_beams = []

        for ind in range(train_opt.n_caption_lstm_step):
            if ind == 0:
                context_tf, log_prob_tf, h_tf, c_tf = model.pred_word(ind)
                log_prob, h, c = sess.run([log_prob_tf, h_tf, c_tf], feed_dict={context_tf:video_feat})
                h_rec[ind][:, 0] = h
                c_rec[ind][:, 0] = c
                beam_seq, beam_seq_logprobs, beam_logprobs_sum, h_rec, c_rec, candidates = beam_step(
                        log_prob, train_opt.beam_size, train_opt.hypo_size, ind,
                        beam_seq, beam_seq_logprobs, beam_logprobs_sum, h_rec, c_rec)
                print('Elapsed time:', str((time.time()-start_time)))
            else:
                log_prob = np.zeros((train_opt.beam_size, n_words)).astype(np.float32)
                for vix in range(train_opt.beam_size):
                    v = candidates[vix]
                    context_tf, log_prob_tf, h_tf, c_tf = model.pred_word(
                        ind, prev_word=v['c'], h=h_rec[ind-1][:,v['q']], c=c_rec[ind-1][:,v['q']])
                    log_prob[vix,:], h_rec[ind][:,vix], c_rec[ind][:,vix] = sess.run([log_prob_tf, h_tf, c_tf],
                                                                            feed_dict={context_tf:video_feat})

                ## suppress UNK tokens in the decoding
                log_prob[:, 3] -= 1000
                beam_seq, beam_seq_logprobs, beam_logprobs_sum, h_rec, c_rec, candidates = beam_step(
                        log_prob, train_opt.beam_size, train_opt.hypo_size, ind,
                        beam_seq, beam_seq_logprobs, beam_logprobs_sum, h_rec, c_rec)

                for beam_ind in range(train_opt.beam_size):
                    if beam_seq[ind, beam_ind] == 2 or ind == train_opt.n_caption_lstm_step-1:
                        final_beam = {
                            'seq': beam_seq[:, beam_ind],
                            'logps': beam_seq_logprobs[:, beam_ind],
                            'p': beam_logprobs_sum[beam_ind]
                        }
                        done_beams.append(copy.deepcopy(final_beam))
                        beam_logprobs_sum[beam_ind] = -1000

                if len(done_beams) >= train_opt.hypo_size:
                    break
        print('Elapsed time:', str((time.time()-start_time)))

        context_tf, generated_words_tf, logit_list_tf, alpha_list_tf = model.build_generator()
        generated_word_index, alpha_list_val = sess.run([generated_words_tf,alpha_list_tf], feed_dict={context_tf:video_feat})
        generated_word_index = [x[0] for x in generated_word_index]

        generated_sentence_gs = convert2sen(ixtoword, generated_word_index)
        print('greedy search: ', generated_sentence_gs)
        output_txt_gs_fd.write(video_feat_path + '\n')
        output_txt_gs_fd.write(generated_sentence_gs + '\n\n')
        print('Elapsed time:', str((time.time()-start_time)))

        for beam_ind in range(train_opt.hypo_size):
            beam = done_beams[beam_ind]
            generated_sentence = convert2sen(ixtoword, beam['seq'])
            print('bs {}, p={}: {}'.format(beam_ind, beam['p'], generated_sentence))
            output_txt_bsm_fd.write(video_feat_path + '\n')
            output_txt_bsm_fd.write(generated_sentence + '\n\n')
            if beam_ind == 1:
                output_txt_bs_fd.write(video_feat_path + '\n')
                output_txt_bs_fd.write(generated_sentence + '\n\n')

    output_txt_gs_fd.close()
    output_txt_bs_fd.close()
    output_txt_bsm_fd.close()
