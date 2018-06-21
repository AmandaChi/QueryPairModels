import sys
sys.path.append('./')
import tensorflow as tf
from utils.param import FLAGS
import math
from utils.layers import xletter_feature_extractor, mask_maxpool
from utils.xletter import XletterPreprocessor
if FLAGS.use_mstf_ops == 1:
    import tensorflow.contrib.microsoft as mstf

def parse_dims(dims_str):
    dims = [int(dim) for dim in dims_str.split(',')]
    return dims

def default_init():
    return tf.contrib.layers.variance_scaling_nitializer(factor=1.0, mode = 'FAN_AVG', uniform = True)

class CDSSMModel():
    def __init__(self):
        if FLAGS.use_mstf_ops == 1:
            self.op_dict = mstf.dssm_dict(FLAGS.xletter_dict)
        elif FLAGS.use_mstf_ops == -1:
            self.op_dict = XletterPreprocessor(FLAGS.xletter_dict, FLAGS.xletter_win_size)
        else:
            self.op_dict = None

    def inference(self, input_fields, mode):
        if mode == tf.contrib.learn.ModeKeys.TRAIN or mode == tf.contrib.learn.ModeKeys.EVAL:
            if FLAGS.use_mstf_ops:
                query, doc = input_fields[0], input_fields[1]
            else:
                query, doc = input_fields[0][0], input_fields[1][0]
            query_vec = self.vector_generation(query, 'Q')
            doc_vec = self.vector_generation(doc, 'D')
        else:
            if FLAGS.use_mstf_ops:
                query, doc = input_fields[0], None
            else:
                query, doc = input_fields[0][0], None
            query_vec = self.vector_generation(query, 'Q')
            doc_vec = None
        return query_vec, doc_vec

    def vector_generation(self, text, model_prefix):
        dims = parse_dims(FLAGS.semantic_model_dims)
        text_vecs, step_mask, sequence_length = xletter_feature_extractor(text, model_prefix, self.op_dict, FLAGS.xletter_cnt, FLAGS.xletter_win_size, FLAGS.dim_xletter_emb)
        maxpooling_vec = mask_maxpool(tf.nn.tanh(text_vecs),step_mask)
        dim_input = FLAGS.dim_xletter_emb
        input_vec = maxpooling_vec
        for i, dim in enumerate(dims):
            dim_output = dim
            random_range = math.sqrt(6.0/(dim_input+dim_output))
            with tf.variable_scope("semantic_layer{:}".format(i)):
                weight = tf.get_variable("weight_" + model_prefix, shape = [dim_input, dim_output], initializer = tf.random_uniform_initializer(-random_range, random_range))
                output_vec = tf.matmul(input_vec, weight)
                output_vec = tf.nn.tanh(output_vec)
                input_vec = output_vec
        normalized_vec = tf.nn.l2_normalize(output_vec, dim = 1)
        return normalized_vec

    def calc_loss(self, inference_res):
        query_vec, doc_vec = inference_res
        batch_size = tf.shape(query_vec)[0]
        posCos = tf.reduce_sum(tf.multiply(query_vec, doc_vec), axis = 1)
        allCos = [posCos]
        for i in range(0, FLAGS.negative_sample):
            random_indices = (tf.range(batch_size) + tf.random_uniform([batch_size],1,batch_size,tf.int32)) % batch_size
            negCos = tf.reduce_sum(tf.multiply(query_vec, tf.gather(doc_vec, random_indices)),axis=1)
            allCos.append(tf.where(tf.equal(negCos,1),tf.zeros_like(negCos),negCos))
        allCos = tf.stack(allCos, axis=1)
        softmax = tf.nn.softmax(allCos * FLAGS.softmax_gamma, dim = 1)
        loss = tf.reduce_sum(-tf.log(softmax[:,0]))
        weight = batch_size
        tf.summary.scalar('softmax_losses',loss)
        return [loss], weight

    def calc_score(self, inference_res):
        query_vec, doc_vec = inference_res
        score = tf.reduce_sum(tf.multiply(query_vec, doc_vec), axis = 1)
        return score


if __name__ == '__main__':
    with tf.Session() as sess:
        m = CDSSMModel()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
