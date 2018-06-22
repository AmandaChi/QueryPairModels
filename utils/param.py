import tensorflow as tf
#Interface
#tf.app.flags.DEFINE_string('input-training-data-path','../../ACDSSM/Train_Data/','training data path')
tf.app.flags.DEFINE_string('input-training-data-path','../../S2S/Train_Data','training data path')
tf.app.flags.DEFINE_string('input-validation-data-path','../Eval_Data/label_data.txt', 'validation path')
#tf.app.flags.DEFINE_string('input-validation-data-path','../Eval_Data/', 'validation path')
#tf.app.flags.DEFINE_string('input-previous-model-path','initial_model','initial model path')
tf.app.flags.DEFINE_string('input-previous-model-path','finalmodel','initial model path')
tf.app.flags.DEFINE_string('output-model-path','finalmodel','path to save model')
tf.app.flags.DEFINE_string('log-dir','log_folder','folder to save log')

#msft cdssm operator
tf.app.flags.DEFINE_integer('use-mstf-ops',-1, 'whether to use mstf operator: 1: use, 0: not use and preprocess xletter in reader, -1: faster than 0')
tf.app.flags.DEFINE_string('xletter-dict','utils/l3g.txt','xletter dictionary name')
tf.app.flags.DEFINE_integer('xletter-win-size',3,'xletter conv win size')
tf.app.flags.DEFINE_integer('xletter-cnt',49292,'xletter feature num')


#Data Reader Setting
tf.app.flags.DEFINE_integer('read-thread',10,'threads count to read data')
tf.app.flags.DEFINE_integer('buffer-size',10000,'buffer size for data reader')

#Trainer
tf.app.flags.DEFINE_string('mode','train','train, predict or evaluation mode')
tf.app.flags.DEFINE_integer('early-stop-steps',30, 'bad checks to trigger early stop, -1 is to disable early stop')
tf.app.flags.DEFINE_integer('batch-size', 256,'training batch size')
tf.app.flags.DEFINE_integer('eval-batch-size',128,'evaluation batch size')
tf.app.flags.DEFINE_integer('num-epochs',5, 'training epochs')
tf.app.flags.DEFINE_integer('max-model-to-keep',10, 'max models to save')
tf.app.flags.DEFINE_integer('log-frequency', 1000, 'log frequency during training procedure')
tf.app.flags.DEFINE_integer('checkpoint-frequency', 100000, 'evaluation frequency during training procedure')
tf.app.flags.DEFINE_float('learning-rate',0.001, 'learning rate')
tf.app.flags.DEFINE_bool('auc-evaluation',True,'whether to do auc evaluation')
tf.app.flags.DEFINE_bool('bleu-evaluation', True, 'whether to do bleu evaluation')
tf.app.flags.DEFINE_integer('negative-sample',4,'negative sample count')
tf.app.flags.DEFINE_string('metrics-early-stop','auc','metrics to control early stop')
tf.app.flags.DEFINE_integer('loss-cnt',1,'total loss count to update')

#Test settings
tf.app.flags.DEFINE_integer('test-fields',3,'test fields count')
tf.app.flags.DEFINE_string('result-filename','predict.txt','result file name')

#CDSSM Model
tf.app.flags.DEFINE_string('semantic-model-dims','64', 'semantic model dims, split by ,')
tf.app.flags.DEFINE_integer('dim-xletter-emb', 288, 'xletter embedding dimension')
tf.app.flags.DEFINE_float('softmax-gamma',10.0,'softmax parameters')


#Seq2Seq Model: From xletter to term
tf.app.flags.DEFINE_string('decoder-vocab-file','vocab100000.txt','term vocabulary file')
tf.app.flags.DEFINE_integer('beam-width',0, 'beam search width')
tf.app.flags.DEFINE_float('length-penalty-weight', 0.0, 'length penalty weight')
tf.app.flags.DEFINE_integer('dim-decoder', 64, 'decoder dimension')
tf.app.flags.DEFINE_integer('dim-decoder-emb', 128, 'decoder embedding dimension')
tf.app.flags.DEFINE_integer('decoder-vocab-size', 100003, 'decoder vocab size')
tf.app.flags.DEFINE_integer('dim-encoder', 64, 'encoder dimension')
tf.app.flags.DEFINE_integer('dim-attention', 64, 'attention dim')

FLAGS = tf.app.flags.FLAGS
