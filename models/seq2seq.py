import sys
sys.path.append('.')
import tensorflow as tf
from utils.param import FLAGS
from tensorflow.python.ops import lookup_ops
from tensorflow.python.layers import core as layers_core
from utils.xletter import XletterPreprocessor
from utils.layers import xletter_feature_extractor, term_emb_extract
if FLAGS.use_mstf_ops == 1:
    import tensorflow.contrib.microsoft as mstf
UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"
UNK_ID = 2
SOS_ID = 1
EOS_ID = 0

def default_init():
        # replica of tf.glorot_uniform_initializer(seed=seed)
    return tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode="FAN_AVG",uniform=True) #seed=seed)

class Seq2Seq():
    def __init__(self):
        #Dictionary initialization
        if FLAGS.use_mstf_ops == 1:
            self.op_dict = mstf.dssm_dict(FLAGS.xletter_dict)
        elif FLAGS.use_mstf_ops == -1:
            self.op_dict = XletterPreprocessor(FLAGS.xletter_dict, FLAGS.xletter_win_size)
        else:
            self.op_dict = None
        self.decoder_dict = lookup_ops.index_table_from_file(FLAGS.input_previous_model_path + "/" + FLAGS.decoder_vocab_file, default_value=UNK_ID)
        self.reverse_decoder_dict = lookup_ops.index_to_string_table_from_file(FLAGS.input_previous_model_path + "/" + FLAGS.decoder_vocab_file, default_value=UNK) 

    #Inference function
    def inference(self, input_fields,mode):
        if mode == tf.contrib.learn.ModeKeys.TRAIN or mode == tf.contrib.learn.ModeKeys.EVAL:
            if FLAGS.use_mstf_ops:
                query,doc = input_fields[0], input_fields[1]
            else:
                query,doc = input_fields[0][0], input_fields[1]
        else:
            if FLAGS.use_mstf_ops:
                query,doc = input_fields[0], None
            else:
                query,doc = input_fields[0][0], None
        output_state, c_state, source_sequence_length = self.encoder(query)
        logits, sample_id, final_state, target_id, target_sequence_length = self.decoder(output_state, c_state, source_sequence_length, doc, mode)
        if not mode == tf.contrib.learn.ModeKeys.INFER:
            return [logits, target_id, target_sequence_length]
        else:
            return [logits, sample_id, target_sequence_length]

    def encoder(self, query):
        with tf.variable_scope('encoder',reuse=tf.AUTO_REUSE):
            #q_vecs, sequence_length = self.xletter_feature_extract(query)
            q_vecs, q_mask, sequence_length = xletter_feature_extractor(query, 'Q', self.op_dict, FLAGS.xletter_cnt, FLAGS.xletter_win_size, FLAGS.dim_xletter_emb)
            encoder_cells = [self.build_cell('encoder_' + str(idx) , FLAGS.dim_encoder) for idx in range(0,2)]
            encoder_cell = tf.contrib.rnn.MultiRNNCell(encoder_cells)
            initial_state = encoder_cell.zero_state(tf.shape(q_vecs)[0], dtype=tf.float32)
            output,c_state = tf.nn.dynamic_rnn(encoder_cell, q_vecs, initial_state = initial_state, sequence_length=sequence_length)
        return output, c_state, sequence_length

    def decoder(self, encoder_outputs, encoder_state, encoder_sequence_length, doc, mode):
        with tf.variable_scope('decoder',reuse=tf.AUTO_REUSE) as decoder_scope:
            decoder_emb = tf.get_variable(name='decoder_emb',shape=[FLAGS.decoder_vocab_size, FLAGS.dim_decoder_emb])
            decoder_cells = [self.build_cell('decoder_' + str(idx), FLAGS.dim_decoder) for idx in range(0,2)]
            decoder_cell = tf.contrib.rnn.MultiRNNCell(decoder_cells)
            #attention_mechanism = self.attention_mechanism_fn(encoder_outputs, tf.maximum(encoder_sequence_length,tf.ones_like(encoder_sequence_length)))
            #alignment_history = False
            #cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size = FLAGS.dim_attention, alignment_history=alignment_history, output_attention = True, name='attention')
            
            #initial_state = self.build_initial_state(encoder_state)
            #initial_state = cell.zero_state(tf.shape(encoder_sequence_length)[0],tf.float32).clone(cell_state = encoder_state)
            output_layer = layers_core.Dense(FLAGS.decoder_vocab_size, name='output_projection', kernel_initializer = default_init())
            if not mode == tf.contrib.learn.ModeKeys.INFER:
                target_input = doc
                #target_emb,step_mask,sequence_length,target_id = self.term_emb_extract(target_input, decoder_emb)
                attention_mechanism = self.attention_mechanism_fn(encoder_outputs, tf.maximum(encoder_sequence_length,tf.ones_like(encoder_sequence_length)))
                alignment_history = False
                cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size = FLAGS.dim_attention, alignment_history=alignment_history, output_attention = True, name='attention')
                initial_state = cell.zero_state(tf.shape(encoder_sequence_length)[0],tf.float32).clone(cell_state = encoder_state)
                #output_layer = layers_core.Dense(FLAGS.decoder_vocab_size, name='output_projection', kernel_initializer = default_init())
                ############
                target_emb,step_mask,sequence_length,target_id = term_emb_extract(target_input, self.decoder_dict, decoder_emb, FLAGS.dim_decoder_emb,add_terminator=True)
                sos_emb = tf.tile(tf.nn.embedding_lookup(decoder_emb,SOS_ID),tf.stack([tf.shape(sequence_length)[0]]))
                sos_emb = tf.reshape(sos_emb,tf.stack([tf.shape(sequence_length)[0],1,FLAGS.dim_decoder_emb]))
                target_emb = tf.concat([sos_emb,target_emb],axis=1)
                sequence_length = sequence_length + 1
                helper = tf.contrib.seq2seq.TrainingHelper(target_emb,sequence_length,time_major=False)
                #Decoder
                #my_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, initial_state)
                my_decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, initial_state)
                outputs, c_state, _ = tf.contrib.seq2seq.dynamic_decode(my_decoder, output_time_major = False,impute_finished=True,scope=decoder_scope)
                #??
                sample_id = outputs.sample_id
                logits = output_layer(outputs.rnn_output)
            else:
                beam_width = FLAGS.beam_width
                start_tokens = tf.fill([tf.shape(encoder_outputs)[0]],SOS_ID)
                end_token = EOS_ID
                if beam_width > 0:
                    encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=beam_width)
                    encoder_sequence_length = tf.contrib.seq2seq.tile_batch(encoder_sequence_length, multiplier=beam_width)
                    encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=beam_width)
                    true_batch_size = tf.shape(encoder_sequence_length)[0]
                    attention_mechanism = self.attention_mechanism_fn(encoder_outputs, tf.maximum(encoder_sequence_length,tf.ones_like(encoder_sequence_length)))
                    cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size = FLAGS.dim_attention, alignment_history=False, output_attention = True, name='attention')
                    initial_state = cell.zero_state(batch_size=true_batch_size, dtype=tf.float32)
                    initial_state = initial_state.clone(cell_state=encoder_state) 
                    my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                            cell = cell,
                            embedding=decoder_emb,
                            start_tokens = start_tokens,
                            end_token = end_token,
                            initial_state=initial_state,
                            beam_width=beam_width,
                            output_layer = output_layer,
                            length_penalty_weight=FLAGS.length_penalty_weight
                            )
                else:
                    attention_mechanism = self.attention_mechanism_fn(encoder_outputs, tf.maximum(encoder_sequence_length,tf.ones_like(encoder_sequence_length)))
                    alignment_history = False
                    cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size = FLAGS.dim_attention, alignment_history=alignment_history, output_attention = True, name='attention')
                    initial_state = cell.zero_state(tf.shape(encoder_sequence_length)[0],tf.float32).clone(cell_state = encoder_state)
                    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_emb, start_tokens, end_token)
                    #my_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, initial_state, output_layer = output_layer)
                    my_decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, initial_state, output_layer = output_layer)
                outputs, c_state, final_sequence_length = tf.contrib.seq2seq.dynamic_decode(my_decoder,maximum_iterations=10,impute_finished=False,scope=decoder_scope)
                target_id, sequence_length = None, None
                if beam_width > 0:
                    logits = tf.no_op()
                    sample_id = outputs.predicted_ids
                    sequence_length = final_sequence_length
                else:
                    logits = outputs.rnn_output
                    sample_id = tf.cast(outputs.sample_id,tf.int64)
                    sequence_length = final_sequence_length
        return logits, sample_id, c_state, target_id, sequence_length

    def build_cell(self, prefix, units):
        with tf.variable_scope('rnn_cell_' + prefix):
            cell = tf.contrib.rnn.GRUCell(units)
        return cell

    def attention_mechanism_fn(self, encoder_outputs, encoder_sequence_length):
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(FLAGS.dim_attention, encoder_outputs, memory_sequence_length = encoder_sequence_length)
        return attention_mechanism

    def calc_loss(self, inference_res):
    #"""Compute optimization loss."""
        logits, target_output, sequence_length = inference_res
        max_time = tf.shape(target_output)[1]
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_output, logits=logits)
        #target_weights = tf.sequence_mask(sequence_length, max_time, dtype=logits.dtype)
        #loss = tf.reduce_sum(crossent * target_weights) / tf.cast(tf.shape(target_output)[0],tf.float32)
        target_weights = tf.sequence_mask(sequence_length, max_time, dtype=tf.bool)
        loss = tf.reduce_sum(tf.where(target_weights, crossent, tf.zeros_like(crossent))) #/ tf.cast(tf.shape(target_output)[0],tf.float32)
        #return tf.reduce_sum(loss), tf.reduce_sum(tf.cast(tf.reduce_sum(sequence_length),tf.float32))
        return [loss], tf.cast(tf.reduce_sum(sequence_length),tf.float32)
    def get_optimizer(self):
        return [tf.train.GradientDescentOptimizer(FLAGS.learning_rate)]
    def calc_score(self, inference_res):
        logits, target_output, sequence_length = inference_res
        if FLAGS.beam_width:
            return tf.zeros([tf.shape(sequence_length)[0]])
        max_time = tf.shape(target_output)[1]
        #crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_output, logits=logits)
        softmax_res = tf.log(tf.nn.softmax(logits))
        idx = tf.where(~tf.is_nan(tf.cast(target_output,tf.float32)))
        values = tf.reshape(target_output,[-1,1])
        idx = tf.concat([idx,values],axis=-1)
        prob = tf.reshape(tf.gather_nd(softmax_res, idx),tf.shape(target_output))
        #return prob
        target_weights = tf.sequence_mask(sequence_length, max_time, dtype=tf.bool)
        score = tf.reduce_sum(tf.where(target_weights, prob, tf.zeros_like(prob)),axis=1)/tf.cast(sequence_length,tf.float32)# / tf.cast(tf.reduce_sum(sequence_length),tf.float32)
        return tf.exp(score)#, target_output, softmax_res

    def lookup_infer(self, inference_res):
        sample_id = inference_res[1]
        if not FLAGS.beam_width:
            sample_id_padding = tf.pad(sample_id, [[0,0],[0,10-tf.shape(sample_id)[1]]])
        else:
            sample_id_padding = tf.pad(sample_id, [[0,0],[0,10-tf.shape(sample_id)[1]],[0,0]])
        reverse_id = self.reverse_decoder_dict.lookup(tf.to_int64(sample_id_padding))
        return reverse_id,inference_res[2]

#Test
if __name__ == '__main__':
    #m = Seq2Seq(tf.contrib.learn.ModeKeys.INFER)
    m = Seq2Seq()
    q = tf.placeholder(tf.string)
    infer = m.inference([q],tf.contrib.learn.ModeKeys.INFER)
    output = m.lookup_infer(infer)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        ckpt = tf.train.get_checkpoint_state(FLAGS.input_previous_model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("No initial model found.")
        fea = sess.run([infer,output],feed_dict={q:["amanda","microsoft surface","amanda is smart","minnwest corporation minnetonka","survey htc"]})#, m.doc:["google doc","what the hell is that","amanda is a beauty"]})
        #fea = sess.run([m.output], feed_dict = {m.query:["san francisco hotels union square","sallie mae private student loan qualifcations"]})
        print(fea)
