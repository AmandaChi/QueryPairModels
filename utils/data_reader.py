import tensorflow as tf
import sys
sys.path.append('../')

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

class XletterPreprocessor():
    def __init__(self, xletter_dict, xletter_win_size):
        self.load_dict(xletter_dict)
        self.win_size = xletter_win_size
    def load_dict(self, filename):
        self.xlt_dict = {}
        idx = 1
        for line in open(filename):
            self.xlt_dict[line.strip()] = idx
            idx += 1
    def extract_xletter(self, term, xlt_dict):
        return [xlt_dict[term[i:i+3]] for i in range(0, len(term)-2) if term[i:i+3] in xlt_dict]
    def xletter_extractor(self, text):
        if isinstance(text,str):
            terms = text.strip().split(" ")
        else:
            terms = text.decode('utf-8').strip().split(" ")
        terms = ['#' + term + '#' for term in terms]
        terms_fea = [self.extract_xletter(term, self.xlt_dict) for term in terms]
        band = int(self.win_size / 2)
        offset = len(self.xlt_dict)
        res = ""
        for i in range(0, len(terms_fea)):
            tmp = ""
            for idx in range(0, self.win_size):
                if i - band + idx >= 0 and i - band + idx < len(terms_fea):
                    if len(tmp) and not tmp[-1] == ",":
                        tmp += ","
                    tmp += ",".join([str(int(idx*offset)+ix) for ix in terms_fea[i-band+idx]])
            if len(tmp):
                #res += ";" + tmp.strip(",") if len(res) else tmp.strip(",")
                res += ";" + tmp if len(res) else tmp
        return res

if __name__ == '__main__':
    inp = InputPipe(FLAGS.input_validation_data_path,2,1,2,"0")
    with tf.Session() as sess:
        for i in range(0,3):
            print(i)
            print(sess.run(inp.get_next())) 


