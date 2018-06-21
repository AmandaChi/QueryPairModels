import tensorflow as tf
import sys
sys.path.append('.')
from utils.xletter import XletterPreprocessor
from utils.param import FLAGS

class InputPipe():
    def __init__(self, filenames, batch_size, num_epochs, fields, xletter_fields, append_ori):
        self.fields = fields
        self.xletter_preprocessor = XletterPreprocessor(FLAGS.xletter_dict, FLAGS.xletter_win_size)
        if len(xletter_fields):
            self.xf = [int(f) for f in xletter_fields.split(',')]
        else:
            self.xf = []
        self.append_ori = append_ori
        ds = tf.data.TextLineDataset(filenames)
        ds = ds.map(self.parse_line, num_parallel_calls = FLAGS.read_thread)
        ds = ds.repeat(num_epochs)
        ds = ds.shuffle(buffer_size=FLAGS.buffer_size)
        ds_batch = ds.batch(batch_size)
        self.iterator = ds_batch.make_initializable_iterator()
    def parse_line(self, line):
        columns = tf.decode_csv(line, [[""] for i in range(0,self.fields)],field_delim="\t",use_quote_delim=False)
        if len(self.xf):
            columns = self.convert_to_xletter(columns)
        return columns
    def convert_to_xletter(self, column):
        res = [tuple(tf.py_func(self.xletter_preprocessor.xletter_extractor,[column[i]],[tf.string])) if i in self.xf else column[i] for i in range(0,self.fields)]
        if self.append_ori:
            res += [column[i] for i in self.xf]
        return res
    def get_next(self):
        return self.iterator.get_next()

if __name__ == '__main__':
    inp = InputPipe(FLAGS.input_validation_data_path,2,1,2,"0")
    with tf.Session() as sess:
        for i in range(0,3):
            print(i)
            print(sess.run(inp.get_next())) 


