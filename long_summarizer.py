# ********************************
#
# This file is used for a package, and will be imported in __init__.py
# By tangbin@2019-11-17
#
# ********************************

import copy
import os
import re
from pyltp import Segmentor

import numpy as np
import tensorflow as tf

from constants import EXP_DIR, LTP_DIR, BPE_CODEC_FILE, BPE_VOCAB_FILE, BPE_VOCAB_THRESHOLD
from model import MultiGPUModel
from run import predict_batch
from subword_nmt.apply_bpe import BPE, read_vocabulary
from utils import BertConfig
from utils.Batcher import PredictBatcher
from utils.Saver import Saver
from utils.data_loader import load_vocab
from constants import CLS_TOKEN, SEP_TOKEN, UNK_TOKEN, PAD_TOKEN


class Preprocessor:
    """
    用于预测的预处理工具。面向场景：对外提供摘要的python接口时，对输入的文本进行预处理后交由模型生成摘要。
    此场景下输入一般是已经读取进来的文本。
    考虑到Batcher已经与预处理的结果强相关，不适合解耦合，因此Preprocessor直接输出Batcher。
    核心方法：preprocess
    输入：list of str, 待预处理的文本列表
    输出：Batcher object, 用于迭代返回batch数据的Batcher对象
    """

    def __init__(self, word2id, seq_length, codecs_file, bpe_vocab_file,
                 vocab_threshold=30, do_lower=False, batch_size=32):
        """
        提供预处理用到的各种资源，同时初始化BPE算法用到的资源
        :param word2id: 词表
        :param seq_length: 序列最大长度
        :param codecs_file: BPE算法用到的codec文件
        :param bpe_vocab_file: BPE算法用到的词表文件
        :param vocab_threshold: BPE算法词表的频数阈值
        :param do_lower: 是否转化为小写
        :param batch_size: batcher的参数
        """
        print('Loading Segmentor Model...')
        self.segmentor = Segmentor()
        self.segmentor.load(os.path.join(LTP_DIR, 'cws.model'))
        with open(bpe_vocab_file, 'r', encoding='utf8') as f:
            vocabulary = read_vocabulary(f, vocab_threshold)
        with open(codecs_file, 'r', encoding='utf8') as f:
            self.bpe = BPE(f, -1, '@@', vocabulary)

        self.do_lower = do_lower
        self.word2id = word2id
        self.seq_length = seq_length
        self.batch_size = batch_size

    def segment_text(self, text, append_cls=False, append_sep=False):
        """
        分词
        :param text: str, 待分词的文本
        :param append_cls: 是否在开头插入<cls>标签
        :param append_sep: 是否在末尾插入<sep>标签
        :return: list, 分词后的tokens
        """
        text = re.sub('\n+', '\n', text)
        paragraphs = text.split('\n')
        result = []
        for p in paragraphs:
            print(p)
            shorts = p.split(' ')
            for short in shorts:
                result.extend(list(self.segmentor.segment(short)))
            result.append('<ln>')
        while result[0] == '<ln>':
            result.pop(0)
        while result[-1] == '<ln>':
            result.pop(-1)
        # print(result)
        if append_cls:
            result.insert(0, '<cls>')
        if append_sep:
            result.append('<sep>')
        return result

    def transform(self, subwords, use_wordpiece=False, substr_prefix='##'):
        """
        将切分好的文本转化为numpy array格式，包括tokens、token_id、mask、oov相关内容等，最终作为Batcher的输入。
        :param subwords: list of list of str(token), 切分好的文本
        :param use_wordpiece: 使用BERT中的WordPiece算法对文本进行切分（暂时不支持）
        :param substr_prefix: WordPiece算法的分隔符，仅当use_wordpiece=True时生效
        :return:
        """
        vocab = self.word2id
        seq_length = self.seq_length
        '''
        # these codes might be valid or removed later, it's uncertain.
        if not use_wordpiece:
            tokenize = lambda x: x.strip().lstrip('<cls>').rstrip('<sep>').strip().split(' ')
        else:
            tokenizer = tokenization.FullTokenizer(vocab=vocab, do_lower_case=self.do_lower, substr_prefix=substr_prefix)
            tokenize = lambda x: tokenizer.tokenize(x)
        '''

        tokens = []
        mask = []
        ids = []
        ids_extend = []
        oovs = []
        oov_size = []
        for i, l in enumerate(subwords):
            oov = []
            tmp_token = [CLS_TOKEN]
            tmp_extend = [vocab[CLS_TOKEN]]
            tmp = [vocab[CLS_TOKEN]]
            for w in l[:seq_length - 2]:
                tmp_token.append(w)
                if w in vocab:
                    tmp.append(vocab[w])
                    tmp_extend.append(vocab[w])
                elif w in oov:
                    tmp.append(vocab[UNK_TOKEN])
                    tmp_extend.append(len(vocab) + oov.index(w))
                else:
                    oov.append(w)
                    tmp.append(vocab[UNK_TOKEN])
                    tmp_extend.append(len(vocab) + oov.index(w))
            tmp_token.append(SEP_TOKEN)
            tmp_extend.append(vocab[SEP_TOKEN])
            tmp.append(vocab[SEP_TOKEN])
            mask.append(([1] * len(tmp_extend) + [0] * (seq_length - len(tmp_extend)))[:seq_length])

            tmp_extend += [0] * (seq_length - len(tmp_extend))
            tmp += [0] * (seq_length - len(tmp))

            tokens.append(tmp_token)
            ids_extend.append(tmp_extend[:seq_length])
            ids.append(tmp[:seq_length])
            oovs.append(oov)
            oov_size.append(len(oov))
        return np.array(tokens), np.array(ids), np.array(ids_extend), np.array(mask), np.array(oov_size), np.array(oovs)

    def apply_bpe(self, tokens):
        """
        对分词后的tokens使用bpe算法进行进一步切分得到subwords。
        :param tokens: 分词后的tokens
        :return: bpe算法切分后的subwords
        """
        result = self.bpe.segment_tokens(tokens=tokens)
        return result

    def preprocess(self, texts):
        """
        调用其它方法进行预处理。推荐批量预处理。
        :param texts: str or list of str, 待处理文本
        :return: 用于逐batch遍历数据的PredictBatcher
        """
        if type(texts) == str:
            texts = [texts]
        texts = [self.apply_bpe(self.segment_text(t.replace('，', ','))) for t in texts]
        src_tokens, src_ids, src_ids_extend, src_mask, src_oov_size, src_oovs = self.transform(texts)
        batcher = PredictBatcher(x_token=src_tokens,
                                 x_ids=src_ids,
                                 x_ids_extend=src_ids_extend,
                                 # x_output=src_embeddings,
                                 x_mask=src_mask,
                                 oov_size=src_oov_size,
                                 oovs=src_oovs,
                                 batch_size=self.batch_size)
        return batcher


class LongTextSummarizer:
    """
    长文本摘要类，用于提供长文本摘要python接口。
    依赖：超参config，包含checkpoint的路径（必须为ckpt文件路径或包含合法的checkpoint文件的目录），Preprocessor所需依赖。
    依赖项通常由get_long_summarizer()函数提供。
    输入：list of str, 原文文本
    输出：list of str, 生成好的摘要
    """

    def __init__(self, config, ckpt_path, ckpt_file=None):
        self.config = copy.deepcopy(config)
        self.model = None
        self.saver = None
        self.session = None
        self.ckpt_path = ckpt_path
        self.ckpt_file = ckpt_file

        self.word2id, self.id2word = None, None
        self.preprocessor = None
        self.tag_pattern = re.compile('\[(SEP|CLS|UNK|PAD)\]')

        self.ready = False

    def initialize(self):
        print('[INFO] Initialization Started.')

        print('[INFO] Vocabulary Loading.')
        self.word2id, self.id2word = load_vocab(self.config.vocab_file, do_lower=self.config.do_lower)

        print('[INFO] Preprocessor Loading.')
        self.preprocessor = Preprocessor(word2id=self.word2id,
                                         seq_length=self.config.encoder_seq_length,
                                         codecs_file=BPE_CODEC_FILE,
                                         bpe_vocab_file=BPE_VOCAB_FILE,
                                         vocab_threshold=BPE_VOCAB_THRESHOLD,
                                         do_lower=False,
                                         batch_size=self.config.batch_size)

        print('[INFO] MultiGPUModel building.')
        self.saver = Saver(ckpt_dir=self.ckpt_path)
        self.model = MultiGPUModel(config=self.config, num_gpus=1, copy_config=False)
        self.model.build(is_training=False)

        print('[INFO] Variables Initializing.')
        self.session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.saver.initialize_variables(ckpt_path=self.ckpt_file)
        self.session.run(tf.global_variables_initializer())

        print('[INFO] Initialization Finished.')
        self.ready = True

    def _preprocess_get_batcher(self, texts):
        return self.preprocessor.preprocess(texts=texts)

    def predict(self, texts, clean_tags=True):
        if not self.ready:
            self.initialize()
        batcher = self._preprocess_get_batcher(texts)
        print('[Info] Start Generating...')
        hyp = predict_batch(
            model=self.model, sess=self.session, eval_batcher=batcher,
            seq_length=self.config.decoder_seq_length, word2id=self.word2id, id2word=self.id2word,
            use_pointer=self.config.use_pointer, substr_prefix=self.config.substr_prefix,
        )
        if clean_tags:
            for i, cand in enumerate(hyp['cand']):
                hyp['cand'][i] = self.tag_pattern.sub('', cand).strip()
        print('[Info] Finished.')
        return hyp['cand']

    @staticmethod
    def get_long_summarizer(ckpt_dir='checkpoint_2019-11-24-22-10-14', ckpt_file=None, init_now=False):
        ckpt_path = os.path.join(EXP_DIR, ckpt_dir)
        assert os.path.exists(ckpt_path) and tf.train.latest_checkpoint(ckpt_path) is not None \
            or os.path.isfile(ckpt_path)
        hyper_parameter_filepath = Saver.parse_hyper_parameter_filepath(ckpt_path=ckpt_path)
        config = BertConfig.from_json_file(hyper_parameter_filepath)

        summarizer = LongTextSummarizer(config=config, ckpt_path=ckpt_path, ckpt_file=ckpt_file)
        if init_now:
            summarizer.initialize()
        return summarizer


def test():
    def get_texts():
        text = []
        x = 0
        fp = './news_text/%d.txt' % x
        while os.path.exists(fp):
            with open(fp, 'r', encoding='utf8') as f:
                t = ''.join(f.readlines())
                text.append(t)
        return text
    summarizer = LongTextSummarizer.get_long_summarizer(ckpt_file='best-3')

    texts = get_texts()
    texts += [
        '''央视网消息（新闻联播）：中共中央总书记、国家主席习近平11月25日在人民大会堂会见由党的最高委员会主席格雷兹洛夫率领的俄罗斯统一俄罗斯党代表团。
习近平祝贺统一俄罗斯党成功举行第十九次代表大会，欢迎格雷兹洛夫率团来华参加中俄执政党对话机制第七次会议。习近平表示，今年是中俄建交70周年，我和普京总统共同宣布发展中俄新时代全面战略协作伙伴关系，共同引领两国关系朝着守望相助、深度融通、开拓创新、普惠共赢的目标和方向迈进。当前中俄双方形成有力战略支撑，对维护世界和平与发展具有重大战略意义。
习近平表示，中国共产党和统一俄罗斯党作为各自国家的执政党，肩负着推动新时代中俄关系取得更大发展的责任和使命。希望双方与会代表充分交流沟通，为深化中俄新时代全面战略协作、促进世界和平与繁荣贡献智慧和力量。
格雷兹洛夫祝贺中共十九届四中全会成功举行，感谢习近平会见统一俄罗斯党代表团，表示近年来俄中关系达到了前所未有的高水平，两国各领域互利合作蓬勃发展，在解决国际及地区热点问题中协同努力，取得良好成效。统一俄罗斯党愿与中国共产党加强交流合作，推动俄中新时代全面战略协作伙伴关系进一步发展。''',
        '''美国国会近日通过所谓“2019年香港人权与民主法案”，粗暴干涉中国内政，公然为激进暴力犯罪分子打气，企图再次通过煽动暴乱来祸害他国。“天下苦美久矣”，美方一些人长期以来实行霸权主义与强权政治，令国际社会深恶痛绝。他们此次把黑手伸向香港、为暴行开“绿灯”的恶劣行径，遭到国际社会强烈谴责。
大量事实表明，美国插手香港事务不是一天两天了。据统计，自1984至2014年，美国国会提出过60多项涉及香港法案。2011年维基解密披露美国驻港总领馆电文显示，美方多次就香港事务发表干涉性意见，并频繁会见反对派人士。此次修例风波中，美国的黑手更是从幕后伸向台前。从美国民主基金会给反中乱港分子提供各种支持，到多位美国政客公然会见“港独”头目；从美国一些媒体混淆黑白颠倒是非，到通过所谓涉港法案妄图实施“长臂管辖”，美方一些人搞乱香港、趁火打劫、牵制中国的险恶用心昭然若揭，国际社会对此洞若观火，普遍予以抨击谴责。
比如，美国库恩基金会主席罗伯特·库恩指出，任何国家都不能允许暴力破坏社会，扰乱经济，美国这一法案对美国、对中国、对世界都是有害的。伊朗外交部发言人穆萨维表示，美国通过此类措施违反国际规范的进程，必将对全球稳定造成严重危害。阿富汗中国友好协会会长苏丹·艾哈迈德·巴辛指出，阿富汗人深知暴力活动会带来怎样的后果，也深受其苦，香港事务纯属中国内政，别国无权干涉。这充分证明，美国为一己之私煽动暴力的行径已成众矢之的，其插手香港事务的图谋不得人心。
然而，在美国一些政客的观念里，“暴乱”是有两套外衣的。如果发生在本国，那绝对不容姑息，必须进行强力打压。于是人们看到，美国警方对2011年“占领华尔街”运动、2015年美国马里兰州巴尔的摩骚乱等都进行了强硬处置。一旦涉及对外事务，这帮政客就立马换了一副嘴脸，将“暴乱”美化为“美丽的风景线”，到处煽风点火，策动骚乱、发动战争、挑起“颜色革命”，唯恐天下不乱，以谋取政治利益、维护美国的全球霸主地位。
过去几十年间，从伊拉克到叙利亚，从阿富汗到利比亚，但凡美国插手之地，都深陷动荡、贫瘠、混乱的泥淖！更严重的是，美国借反恐之名以暴易暴，给人类社会制造了巨大风险隐患。正是因为看透了这一点，当美国一些政客发出所谓“香港人，我们与你在一起”的论调时，网民纷纷嘲讽说，“拜托别了。上次你与利比亚、叙利亚、伊拉克、也门等站在一起的时候……它们都烧了个精光。”这表明，美方一些人的恶言恶行犹如“过街老鼠”，遭到人们的厌恶与鄙夷。
得道多助，失道寡助。美方一些人为暴力“开绿灯”，严重违反国际法与国际关系基本准则，违背人类共同价值观，挑战人类道德与文明底线，实则为本国的未来“亮红灯”，不仅自毁信誉和形象，也将遭到暴力的反噬。比如美国在中东多地挑动战乱酿成大规模难民危机，对美西方社会秩序造成巨大冲击，到头来损害了自身利益。
香港是中国的香港，不是美国某些人手中的风筝，想怎么扯就怎么扯，不会任由美方一些人胡作非为。中国政府维护国家主权、安全、发展利益的决心坚定不移，贯彻“一国两制”方针的决心坚定不移，反对任何外部势力干涉香港事务的决心坚定不移。企图煽动暴力来牵制中国，根本行不通！必将遭到坚决反制！（国际锐评评论员）'''
    ]
    hyps = summarizer.predict(texts)
    for i, summary in enumerate(hyps['cand']):
        print('{}: {}'.format(i, summary))


if __name__ == '__main__':
    test()
