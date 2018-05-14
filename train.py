import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
from utils.utils import prepare_data, fetch_batch_data
from utils.sampling import generate_neg_samples, generate_cl_samples

def train(train_opt):
    train_data, captions, wordtoix, ixtoword, bias_init_vector = prepare_data(train_opt.video_data_path,
            train_opt.video_path, train_opt.video_feat_path,'train', train_opt.dict_path)

    n_words = len(wordtoix)
    print('vocabulary size is %d' % n_words)

    sess = tf.InteractiveSession()
    caption_generator = Video_Caption_Generator(
            n_words=n_words,
            dim_embed=train_opt.dim_embed,
            dim_ctx=train_opt.dim_ctx,
            dim_hidden=train_opt.dim_hidden,
            n_caption_lstm_step=train_opt.n_caption_lstm_step,
            batch_size=train_opt.batch_size,
            ctx_shape=train_opt.ctx_shape,
            bias_init_vector=bias_init_vector)

    loss_m, context_m, sentence_m, mask_m = caption_generator.build_model_m()
    loss_n, context_n, sentence_n, mask_n = caption_generator.build_model_n()
    loss_cl, context_cl, sentence_cl_p, sentence_cl_n, mask_cl, h_seq_pos_cl, h_seq_neg_cl = caption_generator.build_model_cl()
    saver = tf.train.Saver(max_to_keep=50)
    train_op_m = tf.train.AdamOptimizer(train_opt.learning_rate).minimize(loss_m)
    train_op_n = tf.train.AdamOptimizer(train_opt.learning_rate).minimize(loss_n, var_list=[
                                        caption_generator.lstm_W_n, caption_generator.lstm_U_n, caption_generator.lstm_b_n,
                                        caption_generator.decode_lstm_W_n, caption_generator.decode_lstm_W_n])
    train_op_cl = tf.train.AdamOptimizer(train_opt.learning_rate).minimize(loss_cl, var_list=[
                                        caption_generator.lstm_W_m, caption_generator.lstm_U_m, caption_generator.lstm_b_m])
    tf.global_variables_initializer().run()

    #---------------------------------------------------------------------------
    # first train p_m with pure positive samples for epochs_m
    pretrained_model_m = os.path.join(train_opt.model_path, train_opt.pretrained_graph_m)
    if os.path.isfile(pretrained_model_m):
        print("Target model has already been trained...")
    else:
        print("Start training target model...")
        for epoch in range(train_opt.epochs_m+1):
            index = list(train_data.index)
            np.random.shuffle(index)
            train_data_m = train_data.ix[index]
            current_train_data = train_data_m.groupby('video_feat_path').apply(lambda x: x.iloc[0])
            current_train_data = current_train_data.reset_index(drop=True)
            for start, end in zip(
                    range(0, len(current_train_data), train_opt.batch_size),
                    range(train_opt.batch_size, len(current_train_data), train_opt.batch_size)):
                start_time = time.time()
                current_caption_matrix, current_caption_masks, current_feats = fetch_batch_data(current_train_data, start, end, train_opt, wordtoix, 'target')
                _, loss_value = sess.run([train_op_m, loss_m], feed_dict={
                    context_m: current_feats,
                    sentence_m: current_caption_matrix,
                    mask_m: current_caption_masks})
                print('idx:', start, 'Epoch:', epoch, 'loss:', loss_value, 'Elapsed time:', str((time.time()-start_time)))
            if np.mod(epoch, 100) == 0 and epoch > 0:
                print("Epoch ", epoch, "is done. Saving the model...")
                saver.save(sess, os.path.join(train_opt.model_path, 'model'), global_step=epoch)

    #---------------------------------------------------------------------------
    # then train p_n with pure negative samples for epochs_n, load the pre-trained p_m model
    pretrained_model_n = os.path.join(train_opt.model_path, train_opt.pretrained_graph_n)
    if os.path.isfile(pretrained_model_n):
        print("Reference model has already been trained...")
    else:
        print("Start training reference model...")
        saver = tf.train.import_meta_graph(pretrained_model_m)
        saver.restore(sess, os.path.join(train_opt.model_path,'model-900'))
        for epoch in range(train_opt.epochs_m+1, train_opt.epochs_m+train_opt.epochs_n+1):
            current_train_data = generate_neg_samples(data=train_data, phase='reference')
            for start, end in zip(
                    range(0, len(current_train_data), train_opt.batch_size),
                    range(train_opt.batch_size, len(current_train_data), train_opt.batch_size)):
                start_time = time.time()
                current_caption_matrix, current_caption_masks, current_feats = fetch_batch_data(current_train_data, start, end, train_opt, wordtoix, 'reference')
                _, loss_value = sess.run([train_op_n, loss_n], feed_dict={
                    context_n: current_feats,
                    sentence_n: current_caption_matrix,
                    mask_n: current_caption_masks})
                print('idx:', start, 'Epoch:', epoch, 'loss:', loss_value, 'Elapsed time:', str((time.time()-start_time)))
            if np.mod(epoch, 100) == 0 and epoch > 0:
                print("Epoch ", epoch, "is done. Saving the model...")
                saver.save(sess, os.path.join(train_opt.model_path, 'model'), global_step=epoch)

    #---------------------------------------------------------------------------
    # next finetune p_m with p_n model for epochs_cl, load pre-trained p_m and p_n model
    pretrained_model_cl = os.path.join(train_opt.model_path, train_opt.pretrained_graph_cl)
    if os.path.isfile(pretrained_model_cl):
        print("CL model has already been trained...")
    else:
        print("Start training CL model...")
        saver = tf.train.import_meta_graph(os.path.join(train_opt.model_path, 'model-1800.meta'))
        saver.restore(sess, os.path.join(train_opt.model_path, 'model-1800'))
        for epoch in range(train_opt.epochs_m+train_opt.epochs_n+1, train_opt.epochs_m+train_opt.epochs_n+train_opt.epochs_cl+1):
            current_train_data_p, current_train_data_n = generate_cl_samples(train_data, train_opt.neg_K)
            for start, end in zip(
                    range(0, len(current_train_data_p), train_opt.batch_size*train_opt.neg_K),
                    range(train_opt.batch_size*train_opt.neg_K, len(current_train_data_p), train_opt.batch_size*train_opt.neg_K)):
                start_time = time.time()
                current_caption_matrix_p, current_caption_masks_p, current_feats = fetch_batch_data(current_train_data_p, start, end, train_opt, wordtoix, 'CL')
                current_caption_matrix_n, current_caption_masks_n, _ = fetch_batch_data(current_train_data_n, start, end, train_opt, wordtoix, 'CL')
                _, loss_value = sess.run([train_op_cl, loss_cl], feed_dict={
                    context_cl: current_feats,
                    sentence_cl_p: current_caption_matrix_p,
                    sentence_cl_n: current_caption_matrix_n,
                    mask_cl: current_caption_masks_p})
                print('idx:', start, 'Epoch:', epoch, 'loss:', loss_value, 'Elapsed time:', str((time.time()-start_time)))
            if np.mod(epoch, 10) == 0 and epoch > 0:
                print("Epoch ", epoch, "is done. Saving the model...")
                saver.save(sess, os.path.join(train_opt.model_path, 'model'), global_step=epoch)
