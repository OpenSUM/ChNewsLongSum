import numpy as np

import tensorflow as tf

from data.lcsts_loader import load
from utils import load_config, Config
from constants import *


# config = load_config('./lcsts.yaml')
# data = load(config, train=False)
# print(data)

class Model:
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.param = Config()

        self.input_src = None
        self.input_dst = None
        self.dropout = None

        self.V_U_g = tf.get_variable(name='V_U_g', shape=[2 * self.config.gru_hidden_dim],
                                     dtype=tf.float32, initializer=tf.random_normal_initializer())  # todo: here.
        self.V_U_w = tf.get_variable(name='V_U_w', dtype=tf.float32,
                                     shape=[4 * self.config.gru_hidden_dim],
                                     initializer=tf.random_normal_initializer())
        self.W_Era = tf.get_variable(name='W_Era',
                                     shape=[2 * self.config.gru_hidden_dim, 2 * self.config.gru_hidden_dim],
                                     dtype=tf.float32, initializer=tf.random_normal_initializer())
        self.W_Add = tf.get_variable(name='W_Add',
                                     shape=[2 * self.config.gru_hidden_dim, 2 * self.config.gru_hidden_dim],
                                     dtype=tf.float32, initializer=tf.random_normal_initializer())
        self.W_Att = tf.get_variable(name='W_Att',
                                     shape=[2 * self.config.gru_hidden_dim, 2 * self.config.gru_hidden_dim],
                                     dtype=tf.float32, initializer=tf.random_normal_initializer())

        self.W_c = tf.get_variable(name='W_c',
                                   shape=[2*self.config.gru_hidden_dim], dtype=tf.float32, initializer=tf.random_normal_initializer())
        self.W_s = tf.get_variable(name='W_s',
                                   shape=[2*self.config.gru_hidden_dim], dtype=tf.float32, initializer=tf.random_normal_initializer())
        self.W_y = tf.get_variable(name='W_y',
                                   shape=[self.config.embedding_size], dtype=tf.float32, initializer=tf.random_normal_initializer())
        self.b_ptr = tf.get_variable(name='b_ptr',
                                     shape=[1], dtype=tf.float32, initializer=tf.random_normal_initializer())

        self.MS = tf.Variable(name='MS', dtype=tf.float32, trainable=False,
                              initial_value=tf.random_normal(shape=[self.config.sequence_length,
                                                                    self.config.batch_size,
                                                                    2 * self.config.gru_hidden_dim]))

        self.MU = None

        self.w_u = None

        self.all1 = tf.constant(np.zeros(shape=(2 * self.config.gru_hidden_dim), dtype=np.float32))
        self.build()

    # @staticmethod
    def lstm_cell(self, hidden_dim, cudnn=True):
        if cudnn:
            return tf.contrib.rnn.LSTMBlockCell(hidden_dim)
            # return tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(self.config.gru_hidden_dim)
        else:
            return tf.nn.rnn_cell.BasicLSTMCell(hidden_dim, state_is_tuple=True)

    def gru_cell(self, hidden_dim, cudnn=True):
        if cudnn:
            return tf.contrib.rnn.GRUBlockCellV2(hidden_dim)
            # return tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(self.config.gru_hidden_dim)
        else:
            return tf.nn.rnn_cell.GRUCell(hidden_dim)

    def transformer_cell(self, cudnn=False):
        return None

    def cell(self, double=False):
        hidden_dim = self.config.gru_hidden_dim if not double else (2 * self.config.gru_hidden_dim)
        assert self.config.cell in ['gru', 'lstm']
        if self.config.cell == 'lstm':
            cell = self.lstm_cell(hidden_dim)
        elif self.config.cell == 'gru':
            cell = self.gru_cell(hidden_dim)
        else:
            cell = self.transformer_cell()
        return cell

    def dropout_cell(self, cell, keep_prob=None):
        if keep_prob is None:
            keep_prob = self.config.dropout
        return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

    def encoder_basic(self, inputs):
        # encoder definitions
        fcells = [self.cell() for _ in range(self.config.gru_hidden_layers)]
        fstacked_rnn = tf.contrib.rnn.MultiRNNCell(fcells, state_is_tuple=True)
        bcells = [self.cell() for _ in range(self.config.gru_hidden_layers)]
        bstacked_rnn = tf.nn.rnn_cell.MultiRNNCell(bcells, state_is_tuple=True)
        # encode operation
        outputs, states = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(fstacked_rnn, bstacked_rnn, inputs,
                                                                         dtype=tf.float32)
        return outputs, states

    def encode(self, inputs):
        """
        encode the inputs using CUDNN LSTM
        :param inputs: shape=(batch_size, time_len, input_size(embedding_dim)]
        :return: outputs, (output_h, output_c)
        """
        # todo : CudnnLSTM的call方法参数inputs的shape=[time_len, batch_size, input_size]，因此这里需要变换一下维度
        ip = tf.transpose(inputs, perm=[1, 0, 2])
        encoder = tf.contrib.cudnn_rnn.CudnnGRU(
            num_layers=self.config.gru_hidden_layers,
            num_units=self.config.gru_hidden_dim,
            input_mode='linear_input',
            direction='bidirectional',
            dropout=0.0,
            seed=None,
            dtype=tf.float32,
            kernel_initializer=None,
            bias_initializer=None,
            name=None)
        output, state = encoder(ip)  # 这里c和h的顺序与BasicLSTMCell相反
        return output, state

    def decoder_RNN(self):
        cells = self.cell(double=True)

    def read(self, SS):
        """
        read content from MU using SS.
        MU: shape=(time_step, batch_size, hidden_dim)
        :param SS:shape=(batch_size, hidden_dim)
        :return: context c(shape=(batch_size, hidden_dim)), weight w.
        """
        if self.config.cell == 'lstm':
            SS = SS[1]
        SS_tiled = tf.tile(tf.expand_dims(SS, 0), multiples=[self.config.sequence_length, 1, 1])
        t = tf.tensordot(tf.concat([self.MU, SS_tiled], axis=2), self.V_U_w, axes=[2, 0])
        w_ = tf.nn.softmax(t, axis=1)
        g = tf.nn.sigmoid(tf.tensordot(SS, self.V_U_g, axes=[1, 0]))
        if self.w_u is None:
            self.w_u = (1 - g) * w_
        else:
            self.w_u = g * self.w_u + (1 - g) * w_
        c_u = tf.reduce_sum(self.MU * tf.expand_dims(self.w_u, -1), 0)
        return c_u, self.w_u

    def write(self, SS):
        """
        modify content of MU using SS.
        MU: shape=(time_step, batch_size, hidden_dim)
        uEra/uAdd: shape=(batch_size, hiddem_dim)
        all1: shape=(hidden_dim)
        self.w_u: shape=(batch_size,
        :param SS:shape=(batch_size, hidden_dim)
        :return:
        """
        if self.config.cell == 'lstm':
            SS = SS[1]

        uEra = tf.sigmoid(tf.tensordot(SS, self.W_Era, axes=[-1, 0]))
        uAdd = tf.sigmoid(tf.tensordot(SS, self.W_Add, axes=[-1, 0]))
        self.MU = self.MU * (1 - tf.expand_dims(self.w_u, -1) * tf.expand_dims(uEra, 0)) + \
                  tf.expand_dims(self.w_u, -1) * tf.expand_dims(uAdd, 0)
        pass

    def read_MS(self, state):
        """
        read content of MS using state of SU.
        MS: shape=(time_step, batch_size, hidden_dim)
        :param state: shape=(batch_size, hidden_dim)
        :return: context c
        """
        if self.config.cell == 'lstm':
            state = state[1]
        e = tf.tensordot(self.MS, self.W_Att, axes=[-1, 0])  # shape=(time_step, batch_size, hidden_dim)
        e = tf.reduce_sum(tf.expand_dims(state, 0) * e, axis=2)  # shape=(time_step, batch_size)
        a = tf.nn.softmax(e, axis=0)  # shape=(time_step, batch_size)
        c = tf.reduce_sum(tf.expand_dims(a, -1) * self.MS, axis=0)
        # c = self.MS * a  # shape=(time_step, batch_size, hidden_dim)
        return c, a

    def predict(self, y_, SS, cs, alpha, pointer=True):
        """
        calc prediction probabilities of every word
        :param y_: the last output word, shape=(batch_size, embedding_size)
        :param SS: shape=(batch_size, hidden_dim)
        :param cs: shape=(batch_size, hidden_dim)
        :param pointer: whether use pointer-generator network
        :param alpha:
        :return y: probabilities of every word in current batch. shape=(batch_size, predict_vocab_size)
        """
        if self.config.cell == 'lstm':
            SS = SS[1]
        if pointer:
            p_gen = tf.nn.sigmoid(tf.tensordot(cs, self.W_c, axes=[1, 0]) +
                                  tf.tensordot(SS, self.W_s, axes=[1, 0]) +
                                  tf.tensordot(y_, self.W_y, axes=[1, 0]) +
                                  self.b_ptr)
            p_vocab = tf.nn.softmax(tf.layers.dense(tf.layers.dense(tf.concat([SS, cs], axis=1), self.config.gru_hidden_dim), self.config.predict_vocab_size), axis=1)
            logits = None
        else:
            logits = tf.layers.dense(tf.concat([y_, SS, cs], axis=1), self.config.predict_vocab_size)
            # pred = tf.argmax(logits, axis=1)
        return logits

    def decode(self, inputs, init_state, eval=True):
        # decoder definitions
        cellU = self.cell(double=True)
        cellS = self.cell(double=True)
        # decode operation
        if self.config.cell == 'lstm':
            stateU = (tf.random_normal(shape=(-1, 2 * self.config.gru_hidden_dim), dtype=tf.float32),
                      tf.random_normal(shape=(-1, 2 * self.config.gru_hidden_dim), dtype=tf.float32))
            stateS = (tf.random_normal(shape=(-1, 2 * self.config.gru_hidden_dim), dtype=tf.float32),
                      tf.random_normal(shape=(-1, 2 * self.config.gru_hidden_dim), dtype=tf.float32))
        elif self.config.cell == 'gru':
            stateU = tf.random_normal(shape=(-1, 2 * self.config.gru_hidden_dim), dtype=tf.float32)
            stateS = tf.random_normal(shape=(-1, 2 * self.config.gru_hidden_dim), dtype=tf.float32)
        else:
            # todo: transfromer block
            stateU = tf.random_normal(shape=(-1, 2 * self.config.gru_hidden_dim), dtype=tf.float32)
            stateS = tf.random_normal(shape=(-1, 2 * self.config.gru_hidden_dim), dtype=tf.float32)
        yt_ = tf.nn.embedding_lookup(self.data.word_embeddings, [0]*self.config.batch_size)     # y_t-1
        for i in range(self.config.sequence_length):
            # prepare data
            x = inputs[:, i, :]

            # RNN part:
            cu, _ = self.read(stateS)
            ip = tf.concat([x, cu], axis=1)
            outputsU, stateU = cellU(inputs=ip, state=stateU)
            cs, alpha = self.read_MS(stateU)
            outputsS, stateS = cellS(inputs=tf.concat([x, cs], axis=1), state=stateS)
            self.write(stateS)

            # Prediction part
            # todo: codes below need to fill in and correct.
            logits = self.predict(yt_, stateS, cs, alpha)

            stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.one_hot(self.input_dst, depth=self.config.predict_vocab_size, dtype=tf.float32),
                logits=logits,
            )
            loss = tf.reduce_mean(stepwise_cross_entropy)
            train_op = tf.train.AdamOptimizer().minimize(loss)

        pass

    def build(self):
        self.input_src = tf.placeholder(tf.int32, [None, self.config.sequence_length], name='input_src')
        self.input_dst = tf.placeholder(tf.int32, [None, self.config.sequence_length], name='input_dst')
        self.dropout = tf.placeholder(tf.float32, [None], name='keep_prob')
        if self.config.use_pretrain_embeddings == PRETRAIN_WORD_EMBEDDING:
            embedding_matrix = tf.Variable(self.data.word_embeddings, name='pretrain_embeddings', trainable=False)
        # elif self.config.use_pretrain_embeddings == TRAINABLE_WORD_EMBEDDING:
        else:
            embedding_matrix = tf.get_variable(name='trainable_embeddings',
                                               shape=[self.config.vocab_size, self.config.embedding_size])
        embedding_inputs = tf.nn.embedding_lookup(embedding_matrix, self.input_src)
        embedding_outputs = tf.nn.embedding_lookup(embedding_matrix, self.input_dst)

        encoder_outputs, encoder_states = self.encode(embedding_inputs)
        self.MU = encoder_outputs
        # 这里添加了一层非线性变换，用以将encoder的hidden_state变换后作为decoder的initial hidden_state
        decoder_states = tf.layers.dense(encoder_states[0][-1, :, :], 2 * self.config.gru_hidden_dim, activation='relu')

        with tf.control_dependencies([tf.assign(self.MS, self.MU)]):  # todo:这里不知道是不是这么做，姑且放着后面再来改
            self.decode(embedding_outputs, init_state=decoder_states)
            pass
        print('kms')
        pass


if __name__ == '__main__':
    cfg = Config(dict(train_src='./data/lcsts/output/eval.src',
                      train_dst='./data/lcsts/output/eval.dst',
                      eval_src='./data/lcsts/output/test.src',
                      eval_dst='./data/lcsts/output/test.dst',
                      use_pretrain_embeddings=True,
                      embedding_file='./word_embeddings/sgns.merge.word',
                      vocab_file='./data/lcsts/output/vocab',
                      embedding_size=300,
                      sequence_length=5,
                      dropout=1.0,
                      gru_hidden_dim=256,
                      gru_hidden_layers=1,
                      cell='gru',
                      batch_size=10,
                      predict_vocab_size=50000,
                      ))
    data = load(cfg, train=False, eval=False, test=False)
    model = Model(cfg, data)
    print('kms')
