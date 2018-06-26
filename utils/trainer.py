import sys
sys.path.append('./')
import tensorflow as tf
import time
from datetime import datetime
import os
import numpy as np
from utils.param import FLAGS
from sklearn.metrics import roc_auc_score
import nltk

class Metrics:
    def __init__(self):
        self.best_value = 0
        self.best_step = 0
        self.bad_step = 0
        self.improved = False
        self.earlystop = False
        self._top = []
    def update(self, value, step):
        if value > self.best_value:
            self.best_value = value
            self.best_step = step
            self.improved = True
            self.bad_step = 0
        else:
            self.improved = False
            self.bad_step += 1
            if FLAGS.early_stop_steps > 0 and self.bad_step > FLAGS.early_stop_steps:
                self.earlystop = True

class SingleboxTrainer:
    def __init__(self, model, inc_step, inp, score_inp=None, infer_inp=None):
        self.model = model
        self.eval_metrics = Metrics()
        self.inc_step = inc_step
        self.devices = self.get_devices()
        
        self.score_inp = score_inp
        self.infer_inp = infer_inp
        # Training progress record
        #self.total_weight = [tf.Variable(0., trainable=False) for i in range(0, FLAGS.loss_cnt)]
        self.total_weight = [tf.Variable(0., trainable=False) for i in range(0, FLAGS.loss_cnt)]
        self.total_loss = [tf.Variable(0., trainable=False) for i in range(0, FLAGS.loss_cnt)]
        # Optimizer
        opt = self.create_optimizer()
        
        # training
        tower_grads = []
        tower_loss = [[] for i in range(0,FLAGS.loss_cnt)]

        # For Log Printer
        self.weight_record = 0
        
        #Training
        if FLAGS.mode == 'train':
            for i in range(0,len(self.devices)):
                with tf.device(self.devices[i]):
                    with tf.name_scope('device_%d' % i) as scope:
                        batch_input = inp.get_next()
                        loss,weight = self.tower_loss(scope, batch_input)
                        tf.get_variable_scope().reuse_variables()
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                        grads = opt.compute_gradients(loss[0])
                        tower_grads.append(grads)
                        for j in range(0,len(loss)):
                            tower_loss[j].append((loss[j],weight))
            self.avg_loss = [self.update_loss(tower_loss[i],i) for i in range(0, len(tower_loss))]
            grads = self.sum_gradients(tower_grads)
            self.train_op = opt.apply_gradients(grads)

        #Evaluation
        if FLAGS.mode == 'train' and FLAGS.auc_evaluation or FLAGS.mode == 'eval':
            tower_pred = []
            for i in range(0, len(self.devices)):
                with tf.device(self.devices[i]):
                    eval_test_batch = score_inp.get_next()
                    #print(eval_test_batch)
                    score = self.tower_score(eval_test_batch)
                    tower_pred.append([eval_test_batch,score])

            self.score_list = self.merge_eval_res(tower_pred)  
        
        #Inference
        #if FLAGS.mode == 'train' and FLAGS.bleu_evaluation or FLAGS.mode == 'predict':
        #    tower_infer = []
        #    for i in range(0, len(self.devices)):
        #        with tf.device(self.devices[i]):
        #            bleu_test_batch = infer_inp.get_next()
        #            rewrite, res_len, score = self.tower_inference(bleu_test_batch)
        #            tower_infer.append([bleu_test_batch,rewrite, res_len, score])
        #    self.infer_list = self.merge_bleu_res(tower_infer)
        
        #Inference --new
        if FLAGS.mode == 'train' and FLAGS.bleu_evaluation or FLAGS.mode == 'predict':
            tower_infer = []
            for i in range(0, len(self.devices)):
                with tf.device(self.devices[i]):
                    infer_batch = infer_inp.get_next()
                    infer_res = self.tower_inference(infer_batch)
                    tower_infer.append([infer_batch, infer_res])
            self.infer_list = self.merge_infer_res(tower_infer)

        #search
        if FLAGS.mode == 'search':
            tower_search_res = []
            for i in range(0, len(self.devices)):
                with tf.device(self.devices[i]):
                    search_batch = infer_inp.get_next()
                    candidates = self.tower_search(search_batch)
                    tower_search_res.append([search_batch, candidates])
            self.search_list = self.merge_search_res(tower_search_res)


    def update_loss(self, tower_loss, idx):
        loss, weight = zip(*tower_loss)
        loss_inc = tf.assign_add(self.total_loss[idx], tf.reduce_sum(loss) / 10000)
        weight_inc = tf.assign_add(self.total_weight[idx], tf.cast(tf.reduce_sum(weight) / 10000, tf.float32))
        avg_loss = loss_inc / weight_inc
        tf.summary.scalar("avg_loss" + str(idx), avg_loss)
        return avg_loss

    def tower_score(self, batch_input):
        inference_output = self.model.inference(batch_input, tf.contrib.learn.ModeKeys.EVAL)
        prediction = self.model.calc_score(inference_output)
        return prediction

    def tower_inference(self, batch_input):
        inference_output = self.model.inference(batch_input, tf.contrib.learn.ModeKeys.INFER)
        rewrite, seq_length = self.model.lookup_infer(inference_output)
        score = self.model.calc_score(inference_output)
        return rewrite, seq_length, score

    def tower_search(self, batch_input):
        search_idx = self.model.search(batch_input)
        search_res = self.model.lookup_infer(search_idx)
        return search_res

    def tower_loss(self, scope, batch_input):
        inference_output = self.model.inference(batch_input,tf.contrib.learn.ModeKeys.TRAIN)
        loss,weight = self.model.calc_loss(inference_output)
        tf.summary.scalar("losses",loss[0])
        #losses = tf.get_collection('losses',scope)
        #total_loss = tf.add_n(losses, name='total_loss')
        return loss,weight
    
    def merge_eval_res(self, tower_pred):
        test_batch, score = zip(*tower_pred)
        merge_batch = []
        for i in zip(*test_batch):
            if not isinstance(i[0],tf.Tensor):
                merge_batch.append(tf.concat([j[0] for j in i], axis = 0))
            else:
                merge_batch.append(tf.concat(i, axis=0))
        #print(merge_batch)
        merge_score = tf.concat(score, axis = 0)
        return merge_batch, merge_score
    

    def merge_infer_res(self, tower_infer):
        infer_batch, infer_res = zip(*tower_infer)
        merge_batch = []
        merge_res = []
        for i in zip(*infer_batch):
            if not isinstance(i[0],tf.Tensor):
                merge_batch.append(tf.concat([j[0] for j in i], axis = 0))
            else:
                merge_batch.append(tf.concat(i, axis=0))
        for i in zip(*infer_res):
            merge_res.append(tf.concat(i,axis=0))
        return merge_batch, merge_res

    #def merge_bleu_res(self, tower_infer):
    #    test_batch, rewrite, seq_length, score = zip(*tower_infer)
    #    #merge_batch = tf.concat(test_batch, axis = 0)
    #    merge_batch = []
    #    for i in zip(*test_batch):
    #        if not isinstance(i[0],tf.Tensor):
    #            merge_batch.append(tf.concat([j[0] for j in i], axis = 0))
    #        else:
    #            merge_batch.append(tf.concat(i, axis=0))
    #    merge_rewrite = tf.concat(rewrite, axis = 0)
    #    merge_seq_length = tf.concat(seq_length, axis = 0)
    #    merge_score = tf.concat(score, axis = 0)
    #    return merge_batch, merge_rewrite, merge_seq_length, merge_score

    def merge_search_res(self, tower_search):
        search_batch, search_res = zip(*tower_search)
        merge_batch = []
        for i in zip( *search_batch):
            if not isinstance(i[0], tf.Tensor):
                merge_batch.append(tf.concat([j[0] for j in i], axis = 0))
            else:
                merge_batch.append(tf.concat(i, axis=0))
        merge_res = tf.concat(search_res, axis=0)
        return merge_batch, merge_res
    
    def sum_gradients(self, tower_grads):
       sum_grads = []
       #print(tower_grads)
       for grad_and_vars in zip(*tower_grads):
           #print(grad_and_vars)
           if isinstance(grad_and_vars[0][0],tf.Tensor):
               #print(grad_and_vars[0][0])
               grads = []
               for g, _ in grad_and_vars:
                   expanded_g = tf.expand_dims(g,0)
                   grads.append(expanded_g)
               #print(gradsuuu)
               grad = tf.concat(grads, 0)
               grad = tf.reduce_sum(grad, 0)
               v = grad_and_vars[0][1]
               grad_and_var = (grad, v)
               sum_grads.append(grad_and_var)
           else:
               values = tf.concat([g.values for g,_ in grad_and_vars],0)
               indices = tf.concat([g.indices for g,_ in grad_and_vars],0)
               v = grad_and_vars[0][1]
               grad_and_var = (tf.IndexedSlices(values, indices),v)
               sum_grads.append(grad_and_var)
       return sum_grads
    def get_devices(self):
        devices = []
        if os.environ and 'CUDA_VISIBLE_DEVICES' in os.environ:
            for i, gpu_id in enumerate(os.environ['CUDA_VISIBLE_DEVICES'].split(',')):
                gpu_id = int(gpu_id)
                if gpu_id < 0:
                    continue
                devices.append('/gpu:'+str(gpu_id))
        if not len(devices):
            devices.append('/cpu:0')
        print("available devices", devices)
        return devices
    def create_optimizer(self):
        return tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
    def train_ops(self):
        return [self.train_op, self.avg_loss, self.total_weight, self.inc_step]
    def auc_eval_ops(self):
        return self.score_list
    def bleu_eval_ops(self):
        return self.infer_list#[self.rewrite, self.inference_res_bleu]
    def search_ops(self):
        return self.search_list
    def improved(self):
        return self.eval_metrics.improved
    def early_stop(self):
        return self.eval_metrics.earlystop
    def metrics_update(self, score, step):
        self.eval_metrics.update(score, step)
    def print_log(self, total_weight, step, avg_loss):
        examples, self.weight_record = total_weight[0] - self.weight_record, total_weight[0]
        current_time = time.time()
        duration, self.start_time = current_time - self.start_time, time.time()
        examples_per_sec = examples * 10000 / duration
        sec_per_steps = float(duration / FLAGS.log_frequency)
        format_str = "%s: step %d, %5.1f examples/sec, %.3f sec/step, %.1f samples processed,"
        #print(format_str % (datetime.now(), step, avg_loss, examples_per_sec, sec_per_steps, total_weight)) 
        avgloss_str = "avg_loss = " + ",".join([str(avg_loss[i]) for i in range(0,len(avg_loss))])
        print(format_str % (datetime.now(), step, examples_per_sec, sec_per_steps, total_weight[0]) + avgloss_str)
        
    def eval(self, step, sess,eval_type):
        #eval_pipe.reset()
        imporved_mark = ""
        if eval_type == "auc":
            sess.run([self.score_inp.iterator.initializer])
            score_list, label_list = [],[]
            while True:
                try:
                    input_batch, score = sess.run(self.auc_eval_ops())
                    #print(input_batch)
                    label_list.extend([int(i) for i in input_batch[2]])
                    score_list.extend(score)
                except tf.errors.OutOfRangeError:
                    print("auc_evaluation done.")
                    break
            auc_score = roc_auc_score(label_list, score_list)
            improved_mark = ""
            if FLAGS.metrics_early_stop == 'auc':
                self.metrics_update(auc_score, step)
                improved_mark = "?" if self.improved() else ""
            format_str = "%s: step %d, %d sample evaluated, eval_auc = %.10f" + improved_mark
            print(format_str % (datetime.now(), step, len(score_list), auc_score))
        else:
            sess.run([self.infer_inp.iterator.initializer])
            bleu_score = []
            noscore = 0
            while True:
                try:
                    input_batch, [rewrite,resLen,score] = sess.run(self.bleu_eval_ops())
                    for i in range(len(rewrite)):
                        rwt = " ".join([rewrite[i][j].decode('utf-8') for j in range(0, resLen[i]-1)])
                        try:
                        #    print(input_batch[-1][i], rwt)
                            bleu_score.append(nltk.translate.bleu_score.sentence_bleu(input_batch[1][i].decode('utf-8').split(";"), rwt))
                        except:
                            noscore += 1
                except tf.errors.OutOfRangeError:
                    print("bleu evaluation done.")
                    break
            bleu_score_avg = np.mean(bleu_score)
            improved_mark = ""
            if FLAGS.metrics_early_stop == 'bleu':
                self.metrics_update(bleu_score_avg, step)
                improved_mark = "?" if self.improved() else ""
            format_str = "%s: step %d, %d sample evaluated, %d sample without score, eval_bleu = %.10f" + improved_mark
            print(format_str % (datetime.now(), step, len(bleu_score), noscore, bleu_score_avg))

    def predict(self, sess, predict_mode, outputter):
        if predict_mode == tf.contrib.learn.ModeKeys.EVAL:
            sess.run([self.score_inp.iterator.initializer])
            xf_count = len(self.score_inp.xf)
            while True:
                try:
                    input_batch,score = sess.run(self.auc_eval_ops())
                    for i in range(len(score)):
                        output_str = ""
                        xf_ori = -xf_count
                        for j in range(0,self.score_inp.fields):
                            if j in self.score_inp.xf:
                                output_str += input_batch[xf_ori][i].decode('utf-8') + "\t"
                                xf_ori += 1
                            else:
                                output_str += input_batch[j][i].decode('utf-8') + "\t"
                        output_str += str(score[i])
                        outputter.write(output_str + "\n")
                except tf.errors.OutOfRangeError:
                    print("score predict done.")
                    break
        else:
            sess.run([self.infer_inp.iterator.initializer])
            xf_count = len(self.infer_inp.xf)
            while True:
                try:
                    input_batch, [rewrite, resLen, score] = sess.run(self.bleu_eval_ops())
                    for i in range(len(resLen)):
                        output_str = ""
                        xf_ori = -xf_count
                        for j in range(0, self.infer_inp.fields):
                            if j in self.infer_inp.xf:
                                output_str += input_batch[xf_ori][i].decode('utf-8') + "\t"
                                xf_ori += 1
                            else:
                                output_str += input_batch[j][i].decode('utf-8') + "\t"
                        if not FLAGS.beam_width:
                            output_str += " ".join([rewrite[i][j].decode('utf-8') for j in range(0, resLen[i] - 1)]) + "\t"
                        else:
                            res = []
                            for k in range(0, FLAGS.beam_width):
                                res.append(" ".join(filter(lambda x: not x=='</s>',[rewrite[i][j][k].decode('utf-8') for j in range(0, resLen[i][k] - 1)])))
                            output_str += ",".join(res) + "\t"
                        output_str += str(score[i])
                        outputter.write(output_str + "\n")
                except tf.errors.OutOfRangeError:
                    print("inference done.")
                    break
    
    def search(self, sess, outputter):
        cnt = 0
        sess.run([self.infer_inp.iterator.initializer])
        while True:
            try:
                cnt += 1
                if cnt % 10000 == 0:
                    print(cnt)
                input_batch,res = sess.run(self.search_ops())
                #print(input_batch)
                #print(res[0])
                for i in range(len(res)):
                    output_str = ""
                    for j in range(0, 1):
                        output_str += input_batch[j][i].decode('utf-8') + "\t"
                    output_str += ",".join([res[i][j].decode('utf-8') for j in range(0, len(res[i]))])
                    outputter.write(output_str + "\n")
            except tf.errors.OutOfRangeError:
                print("search done.")
                break



        

                                


