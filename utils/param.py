import tensorflow as tf
#Interface
tf.app.flags.DEFINE_string('input-training-data-path','../../ACDSSM/Train_Data/','training data path')
tf.app.flags.DEFINE_string('input-validation-data-path','../Eval_Data/label_data.txt', 'validation path')
tf.app.flags.DEFINE_string('input-previous-model-path','initial_model_ql_tree','initial model path')
tf.app.flags.DEFINE_string('output-model-path','finalmodel','path to save model')
tf.app.flags.DEFINE_string('log-dir','log_folder','folder to save log')

#msft cdssm operator
tf.app.flags.DEFINE_bool('use-mstf-ops',False, 'whether to use mstf operator')
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
tf.app.flags.DEFINE_bool('bleu-evaluation', False, 'whether to do bleu evaluation')
tf.app.flags.DEFINE_integer('negative-sample',4,'negative sample count')
tf.app.flags.DEFINE_string('metrics-early-stop','auc','metrics to control early stop')
tf.app.flags.DEFINE_integer('loss-cnt',1,'total loss count to update')
tf.app.flags.DEFINE_string('result-filename','predict.txt','result file name')

#CDSSM Model
tf.app.flags.DEFINE_string('semantic-model-dims','64', 'semantic model dims, split by ,')
tf.app.flags.DEFINE_integer('dim-xletter-emb', 288, 'xletter embedding dimension')
tf.app.flags.DEFINE_float('softmax-gamma',10.0,'softmax parameters')

FLAGS = tf.app.flags.FLAGS
