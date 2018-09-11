# Copied from https://github.com/ruotianluo/NeuralDialog-CVAE-pytorch/blob/master/data_apis/corpus.py
# Copyright (C) 2017 Tiancheng Zhao, Carnegie Mellon University

import pickle as pkl
from collections import Counter
import numpy as np
import nltk


class SWDACorpus(object):
    dialog_act_id = 0
    sentiment_id = 1
    liwc_id = 2

    def __init__(self, path, vocab_size=10000, wordvec_path=None, wordvec_dim=None):
        """
        :param path: the folder that contains the SWDA dialog corpus
        """
        data = pkl.load(open(path+'full_swda_clean_42da_sentiment_dialog_corpus.p', "rb"))
        
        self.emb_dim = wordvec_dim
        self.word2vec = None
        self.sil_utt = ["<s>", "<sil>", "</s>"]
        
        self.train_corpus = self.process(data["train"])
        self.valid_corpus = self.process(data["valid"])
        self.test_corpus = self.process(data["test"])
        self.build_vocab(vocab_size)
        self.load_word2vec(wordvec_path)
        print("Done loading corpus")

    def process(self, data):
        """new_dialog: [(a, 1/0), (a,1/0)], new_meta: (a, b, topic), new_utt: [[a,b,c)"""
        """ 1 is own utt and 0 is other's utt"""
        new_dialog = []
        new_meta = []
        new_utts = []
        bod_utt = ["<s>", "<d>", "</s>"] # indicator of a start of a dialog
        all_lenes = []

        for l in data:
            lower_utts = [(caller, ["<s>"] + nltk.WordPunctTokenizer().tokenize(utt.lower()) + ["</s>"], feat)
                          for caller, utt, feat in l["utts"]]
            
            all_lenes.extend([len(u) for c, u, f in lower_utts])

            a_age = float(l["A"]["age"])/100.0
            b_age = float(l["B"]["age"])/100.0
            a_edu = float(l["A"]["education"])/3.0
            b_edu = float(l["B"]["education"])/3.0
            vec_a_meta = [a_age, a_edu] + ([0, 1] if l["A"]["sex"] == "FEMALE" else [1, 0])
            vec_b_meta = [b_age, b_edu] + ([0, 1] if l["B"]["sex"] == "FEMALE" else [1, 0])

            # for joint model we mode two side of speakers together. if A then its 0 other wise 1
            meta = (vec_a_meta, vec_b_meta, l["topic"])
            dialog = [(bod_utt, 0, None)] + [(utt, int(caller=="B"), feat) for caller, utt, feat in lower_utts]

            new_utts.extend([bod_utt] + [utt for caller, utt, feat in lower_utts])
            new_dialog.append(dialog)
            new_meta.append(meta)

        print("Max utt len %d, mean utt len %.2f" % (np.max(all_lenes), float(np.mean(all_lenes))))
        return new_dialog, new_meta, new_utts

    def build_vocab(self, vocab_size):
        all_words = []
        for tokens in self.train_corpus[2]: # utterances
            all_words.extend(tokens)
        vocab_count = Counter(all_words).most_common()
        raw_vocab_size = len(vocab_count)
        discard_wc = np.sum([c for t, c, in vocab_count[vocab_size:]])
        vocab_count = vocab_count[0:vocab_size]

        # create vocabulary list sorted by count
        print("Load corpus with train size %d, valid size %d, "
              "test size %d raw vocab size %d vocab size %d at cut_off %d OOV rate %f"
              % (len(self.train_corpus), len(self.valid_corpus), len(self.test_corpus),
                 raw_vocab_size, len(vocab_count), vocab_count[-1][1], float(discard_wc) / len(all_words)))

        self.vocab = ["<pad>", "<unk>"] + [t for t, cnt in vocab_count]
        self.ivocab = {t: idx for idx, t in enumerate(self.vocab)}
        self.unk_id = self.ivocab["<unk>"]
        self.sos_id = self.ivocab["<s>"]
        self.eos_id = self.ivocab["</s>"]
        print("<d> index %d" % self.ivocab["<d>"])
        print("<sil> index %d" % self.ivocab.get("<sil>", -1))

        # create topic vocab
        all_topics = []
        for a, b, topic in self.train_corpus[1]:
            all_topics.append(topic)
        self.topic_vocab = [t for t, cnt in Counter(all_topics).most_common()]
        self.rev_topic_vocab = {t: idx for idx, t in enumerate(self.topic_vocab)}
        print("%d topics in train data" % len(self.topic_vocab))

        # get dialog act labels
        all_dialog_acts = []
        for dialog in self.train_corpus[0]:
            all_dialog_acts.extend([feat[self.dialog_act_id] for caller, utt, feat in dialog if feat is not None])
        self.dialog_act_vocab = [t for t, cnt in Counter(all_dialog_acts).most_common()]
        self.rev_dialog_act_vocab = {t: idx for idx, t in enumerate(self.dialog_act_vocab)}
        print(self.dialog_act_vocab)
        print("%d dialog acts in train data" % len(self.dialog_act_vocab))

    def load_word2vec(self, word_vec_path):
        if word_vec_path is None:
            return None
        with open(word_vec_path, "r") as f:
            lines = f.readlines()
        raw_word2vec = {}
        for l in lines:
            w, vec = l.split(" ", 1)
            raw_word2vec[w] = vec
        # clean up lines for memory efficiency
        self.word2vec = None
        oov_cnt = 0
        for v in self.vocab:
            str_vec = raw_word2vec.get(v, None)
            if str_vec is None:
                oov_cnt += 1
                vec = np.random.randn(self.emb_dim) * 0.1
            else:
                vec = np.fromstring(str_vec, sep=" ")
            vec=np.expand_dims(vec, axis=0)
            self.word2vec=np.concatenate((self.word2vec, vec),0) if self.word2vec is not None else vec
        print("word2vec cannot cover %f vocab" % (float(oov_cnt)/len(self.vocab)))
    def get_utts(self):
        def _to_id_corpus(data):
            results = []
            for line in data:
                results.append([self.ivocab.get(t, self.unk_id) for t in line])
            return results
        # convert the corpus into ID
        id_train = _to_id_corpus(self.train_corpus[2])
        id_valid = _to_id_corpus(self.valid_corpus[2])
        id_test = _to_id_corpus(self.test_corpus[2])
        return {'train': id_train, 'valid': id_valid, 'test': id_test}

    def get_dialogs(self):
        def _to_id_corpus(data):
            results = []
            for dialog in data:
                temp = []
                # convert utterance and feature into numeric numbers
                for utt, floor, feat in dialog:
                    if feat is not None:
                        id_feat = list(feat)
                        id_feat[self.dialog_act_id] = self.rev_dialog_act_vocab[feat[self.dialog_act_id]]
                    else:
                        id_feat = None
                    temp.append(([self.ivocab.get(t, self.unk_id) for t in utt], floor, id_feat))
                results.append(temp)
            return results
        id_train = _to_id_corpus(self.train_corpus[0])
        id_valid = _to_id_corpus(self.valid_corpus[0])
        id_test = _to_id_corpus(self.test_corpus[0])
        return {'train': id_train, 'valid': id_valid, 'test': id_test}

    def get_metas(self):
        def _to_id_corpus(data):
            results = []
            for m_meta, o_meta, topic in data:
                results.append((m_meta, o_meta, self.rev_topic_vocab[topic]))
            return results

        id_train = _to_id_corpus(self.train_corpus[1])
        id_valid = _to_id_corpus(self.valid_corpus[1])
        id_test = _to_id_corpus(self.test_corpus[1])
        return {'train': id_train, 'valid': id_valid, 'test': id_test}

    

# Data feed
class SWDADataLoader(object):
    def __init__(self, name, data, meta_data, max_utt_len):
        assert len(data) == len(meta_data)
        self.batch_size = 0
        self.context_size = 0
        self.step_size = 0
        self.ptr = 0
        self.num_batch = None
        self.batch_indexes = None
        self.grid_indexes = None
        self.prev_alive_size = 0

        self.name = name
        self.data = data
        self.meta_data = meta_data
        self.data_size = len(data)
        self.data_lens = all_lens = [len(line) for line in self.data]
        self.max_utt_size = max_utt_len
        print("Max len %d and min len %d and avg len %f" % (np.max(all_lens), np.min(all_lens),
                                                            float(np.mean(all_lens))))
        self.indexes = list(np.argsort(all_lens))

    def _shuffle_batch_indexes(self):
        np.random.shuffle(self.batch_indexes)

    def epoch_init(self, batch_size, context_size, step_size, shuffle=True, intra_shuffle=True):
        assert len(self.indexes) == self.data_size and len(self.data_lens) == self.data_size

        self.ptr = 0
        self.batch_size = batch_size
        self.context_size = context_size
        self.step_size = step_size
        self.prev_alive_size = batch_size

        # create batch indexes
        temp_num_batch = self.data_size // batch_size
        self.batch_indexes = []
        for i in range(temp_num_batch):
            self.batch_indexes.append(self.indexes[i*self.batch_size:(i+1)*self.batch_size])

        left_over = self.data_size-temp_num_batch*batch_size
        # shuffle batch indexes
        if shuffle:
            self._shuffle_batch_indexes()

        # create grid indexes
        self.grid_indexes = []
        for idx, b_ids in enumerate(self.batch_indexes):
            # assume the b_ids are sorted
            all_lens = [self.data_lens[i] for i in b_ids]
            max_len = self.data_lens[b_ids[-1]]
            min_len = self.data_lens[b_ids[0]]
            assert np.max(all_lens) == max_len
            assert np.min(all_lens) == min_len
            num_seg = (max_len-self.context_size) // self.step_size
            if num_seg > 0:
                cut_start = list(range(0, num_seg*self.step_size, step_size))
                cut_end = list(range(self.context_size, num_seg*self.step_size+self.context_size, step_size))
                assert cut_end[-1] < max_len
                cut_start = [0] * (self.context_size-2) +cut_start # since we give up on the seq training idea
                cut_end = list(range(2, self.context_size)) + cut_end
            else:
                cut_start = [0] * (max_len-2)
                cut_end = list(range(2, max_len))

            new_grids = [(idx, s_id, e_id) for s_id, e_id in zip(cut_start, cut_end) if s_id < min_len-1]
            if intra_shuffle and shuffle:
                np.random.shuffle(new_grids)
            self.grid_indexes.extend(new_grids)

        self.num_batch = len(self.grid_indexes)
        print("%s begins with %d batches with %d left over samples" % (self.name, self.num_batch, left_over))

    def next_batch(self):
        if self.ptr < self.num_batch:
            current_grid = self.grid_indexes[self.ptr]
            if self.ptr > 0:
                prev_grid = self.grid_indexes[self.ptr-1]
            else:
                prev_grid = None
            self.ptr += 1
            return self._prepare_batch(cur_grid=current_grid, prev_grid=prev_grid)
        else:
            return None

    def pad_to(self, tokens, do_pad=True):
        if len(tokens) >= self.max_utt_size:
            return tokens[0:self.max_utt_size-1] + [tokens[-1]]
        elif do_pad:
            return tokens + [0] * (self.max_utt_size-len(tokens))
        else:
            return tokens

    def _prepare_batch(self, cur_grid, prev_grid):
        # the batch index, the starting point and end point for segment
        b_id, s_id, e_id = cur_grid

        batch_ids = self.batch_indexes[b_id]
        rows = [self.data[idx] for idx in batch_ids]
        meta_rows = [self.meta_data[idx] for idx in batch_ids]
        dialog_lens = [self.data_lens[idx] for idx in batch_ids]

        topics = np.array([meta[2] for meta in meta_rows])
        cur_pos = [np.minimum(1.0, e_id/float(l)) for l in dialog_lens]

        # input_context, context_lens, floors, topics, a_profiles, b_Profiles, outputs, output_lens
        context_utts, context_lens,utt_lens,floors, out_utts,out_lens,out_floors,out_das = [],[],[],[],[],[],[],[]
        for row in rows:
            if s_id < len(row)-1:
                cut_row = row[s_id:e_id]
                in_row = cut_row[0:-1]
                out_row = cut_row[-1]
                out_utt, out_floor, out_feat = out_row
                
                context_utts.append([self.pad_to(utt) for utt, floor, feat in in_row])
                utt_lens.append([min(len(utt),self.max_utt_size) for utt, floor, feat in in_row])
                context_lens.append(len(cut_row) - 1)
                floors.append([int(floor==out_floor) for utt, floor, feat in in_row])

                out_utt = self.pad_to(out_utt, do_pad=False)
                out_utts.append(out_utt)
                out_lens.append(len(out_utt))
                out_floors.append(out_floor)
                out_das.append(out_feat[0])
            else:
                print(row)
                raise ValueError("S_ID %d larger than row" % s_id)
        #print(context_lens)
        # my_profiles = np.array([meta[out_floors[idx]] + [cur_pos[idx]] for idx, meta in enumerate(meta_rows)])
        my_profiles = np.array([meta[out_floors[idx]] for idx, meta in enumerate(meta_rows)], dtype=np.float32)
        ot_profiles = np.array([meta[1-out_floors[idx]] for idx, meta in enumerate(meta_rows)], dtype=np.float32)
        vec_context_lens = np.array(context_lens, dtype=np.int64)
        vec_context = np.zeros((self.batch_size, max(context_lens), self.max_utt_size), dtype=np.int64)
        #print(utt_lens)
        vec_utt_lens = np.ones((self.batch_size, max(context_lens)), dtype=np.int64)+1 #np.array(utt_lens, dtype=np.int64)
        vec_floors = np.zeros((self.batch_size, np.max(vec_context_lens)), dtype=np.int64)
        vec_outs = np.zeros((self.batch_size, np.max(out_lens)), dtype=np.int64)
        vec_out_lens = np.array(out_lens, dtype=np.int64)
        vec_out_das = np.array(out_das, dtype=np.int64)

        for b_id in range(self.batch_size):
            vec_outs[b_id, 0:vec_out_lens[b_id]] = out_utts[b_id]
            vec_floors[b_id, 0:vec_context_lens[b_id]] = floors[b_id]
            vec_context[b_id, 0:vec_context_lens[b_id], :] = np.array(context_utts[b_id])
            vec_utt_lens[b_id, 0:vec_context_lens[b_id]] = utt_lens[b_id]

        return vec_context, vec_context_lens, vec_utt_lens, vec_floors, topics, \
               my_profiles, ot_profiles, vec_outs, vec_out_lens, vec_out_das


        
        


class DailyDialCorpus(object):
    dialog_act_id = 0

    def __init__(self, path, vocab_size=10000, wordvec_path=None, wordvec_dim=None):
        """
        :param path: the folder that contains the SWDA dialog corpus
        """
        train_data = open(path+'train.utts.txt', "r").readlines()
        valid_data = open(path+'valid.utts.txt', "r").readlines()
        test_data = open(path+'test.utts.txt', "r").readlines()
        
        self.emb_dim = wordvec_dim
        self.word2vec = None
        self.sil_utt = ["<s>", "<sil>", "</s>"]
        
        self.train_corpus = self.process(train_data)
        self.valid_corpus = self.process(valid_data)
        self.test_corpus = self.process(test_data)
        
        self.build_vocab(vocab_size)
        self.load_word2vec(wordvec_path)
        print("Done loading corpus")

    def process(self, data):
        """new_dialog: [(a, 1/0), (a,1/0)], new_utt: [[a,b,c)"""
        """ 1 is own utt and 0 is other's utt"""
        new_dialog = []
        new_utts = []
        bod_utt = ["<s>", "<d>", "</s>"] # indicator of a start of a dialog
        all_lenes = []

        for l in data:        
            lower_utts = [["<s>"] + nltk.WordPunctTokenizer().tokenize(utt.lower()) + ["</s>"]
                          for utt in l.split('__eou__')[:-1]]
            all_lenes.extend([len(u) for u in lower_utts])
            
            dialog = [(bod_utt, 0)]
            floor = 1
            for utt in lower_utts:
                floor = floor+1
                dialog = dialog + [(utt, int(floor%2==0))]
            new_utts.extend([bod_utt] + [utt for utt in lower_utts])
            new_dialog.append(dialog)

        print("Max utt len %d, mean utt len %.2f" % (np.max(all_lenes), float(np.mean(all_lenes))))
        return new_dialog, new_utts

    def build_vocab(self, vocab_size):
        all_words = []
        for tokens in self.train_corpus[1]: # utterances
            all_words.extend(tokens)
        vocab_count = Counter(all_words).most_common()
        raw_vocab_size = len(vocab_count)
        discard_wc = np.sum([c for t, c, in vocab_count[vocab_size:]])
        vocab_count = vocab_count[0:vocab_size]

        # create vocabulary list sorted by count
        print("Load corpus with train size %d, valid size %d, "
              "test size %d raw vocab size %d vocab size %d at cut_off %d OOV rate %f"
              % (len(self.train_corpus), len(self.valid_corpus), len(self.test_corpus),
                 raw_vocab_size, len(vocab_count), vocab_count[-1][1], float(discard_wc) / len(all_words)))

        self.vocab = ["<pad>", "<unk>"] + [t for t, cnt in vocab_count]
        self.ivocab = {t: idx for idx, t in enumerate(self.vocab)}
        self.unk_id = self.ivocab["<unk>"]
        self.sos_id = self.ivocab["<s>"]
        self.eos_id = self.ivocab["</s>"]
        print("<d> index %d" % self.ivocab["<d>"])
        print("<sil> index %d" % self.ivocab.get("<sil>", -1))


    def load_word2vec(self, word_vec_path):
        if word_vec_path is None:
            return None
        with open(word_vec_path, "r") as f:
            lines = f.readlines()
        raw_word2vec = {}
        for l in lines:
            w, vec = l.split(" ", 1)
            raw_word2vec[w] = vec
        # clean up lines for memory efficiency
        self.word2vec = None
        oov_cnt = 0
        for v in self.vocab:
            str_vec = raw_word2vec.get(v, None)
            if str_vec is None:
                oov_cnt += 1
                vec = np.random.randn(self.emb_dim) * 0.1
            else:
                vec = np.fromstring(str_vec, sep=" ")
            vec=np.expand_dims(vec, axis=0)
            self.word2vec=np.concatenate((self.word2vec, vec),0) if self.word2vec is not None else vec
        print("word2vec cannot cover %f vocab" % (float(oov_cnt)/len(self.vocab)))
    def get_utts(self):
        def _to_id_corpus(data):
            results = []
            for line in data:
                results.append([self.ivocab.get(t, self.unk_id) for t in line])
            return results
        # convert the corpus into ID
        id_train = _to_id_corpus(self.train_corpus[1])
        id_valid = _to_id_corpus(self.valid_corpus[1])
        id_test = _to_id_corpus(self.test_corpus[1])
        return {'train': id_train, 'valid': id_valid, 'test': id_test}

    def get_dialogs(self):
        def _to_id_corpus(data):
            results = []
            for dialog in data:
                temp = []
                # convert utterance and feature into numeric numbers
                for utt, floor in dialog:
                    temp.append(([self.ivocab.get(t, self.unk_id) for t in utt], floor))
                results.append(temp)
            return results
        id_train = _to_id_corpus(self.train_corpus[0])
        id_valid = _to_id_corpus(self.valid_corpus[0])
        id_test = _to_id_corpus(self.test_corpus[0])
        return {'train': id_train, 'valid': id_valid, 'test': id_test}
    def get_metas(self):
        return {'train': None, 'valid': None, 'test': None}

    

# Data feed
class DailyDialDataLoader(object):
    def __init__(self, name, data, meta_data, max_utt_len):
        self.batch_size = 0
        self.context_size = 0
        self.step_size = 0
        self.ptr = 0
        self.num_batch = None
        self.batch_indexes = None
        self.grid_indexes = None
        self.prev_alive_size = 0

        self.name = name
        self.data = data
        self.data_size = len(data)
        self.data_lens = all_lens = [len(line) for line in self.data]
        self.max_utt_size = max_utt_len
        print("Max len %d and min len %d and avg len %f" % (np.max(all_lens), np.min(all_lens),
                                                            float(np.mean(all_lens))))
        self.indexes = list(np.argsort(all_lens))

    def _shuffle_batch_indexes(self):
        np.random.shuffle(self.batch_indexes)

    def epoch_init(self, batch_size, context_size, step_size, shuffle=True, intra_shuffle=True):
        assert len(self.indexes) == self.data_size and len(self.data_lens) == self.data_size

        self.ptr = 0
        self.batch_size = batch_size
        self.context_size = context_size
        self.step_size = step_size
        self.prev_alive_size = batch_size

        # create batch indexes
        temp_num_batch = self.data_size // batch_size
        self.batch_indexes = []
        for i in range(temp_num_batch):
            self.batch_indexes.append(self.indexes[i*self.batch_size:(i+1)*self.batch_size])

        left_over = self.data_size-temp_num_batch*batch_size
        # shuffle batch indexes
        if shuffle:
            self._shuffle_batch_indexes()

        # create grid indexes
        self.grid_indexes = []
        for idx, b_ids in enumerate(self.batch_indexes):
            # assume the b_ids are sorted
            all_lens = [self.data_lens[i] for i in b_ids]
            max_len = self.data_lens[b_ids[-1]]
            min_len = self.data_lens[b_ids[0]]
            assert np.max(all_lens) == max_len
            assert np.min(all_lens) == min_len
            num_seg = (max_len-self.context_size) // self.step_size
            if num_seg > 0:
                cut_start = list(range(0, num_seg*self.step_size, step_size))
                cut_end = list(range(self.context_size, num_seg*self.step_size+self.context_size, step_size))
                assert cut_end[-1] < max_len
                cut_start = [0] * (self.context_size-2) +cut_start # since we give up on the seq training idea
                cut_end = list(range(2, self.context_size)) + cut_end
            else:
                cut_start = [0] * (max_len-2)
                cut_end = list(range(2, max_len))

            new_grids = [(idx, s_id, e_id) for s_id, e_id in zip(cut_start, cut_end) if s_id < min_len-1]
            if intra_shuffle and shuffle:
                np.random.shuffle(new_grids)
            self.grid_indexes.extend(new_grids)

        self.num_batch = len(self.grid_indexes)
        print("%s begins with %d batches with %d left over samples" % (self.name, self.num_batch, left_over))

    def next_batch(self):
        if self.ptr < self.num_batch:
            current_grid = self.grid_indexes[self.ptr]
            if self.ptr > 0:
                prev_grid = self.grid_indexes[self.ptr-1]
            else:
                prev_grid = None
            self.ptr += 1
            return self._prepare_batch(cur_grid=current_grid, prev_grid=prev_grid)
        else:
            return None

    def pad_to(self, tokens, do_pad=True):
        if len(tokens) >= self.max_utt_size:
            return tokens[0:self.max_utt_size-1] + [tokens[-1]]
        elif do_pad:
            return tokens + [0] * (self.max_utt_size-len(tokens))
        else:
            return tokens

    def _prepare_batch(self, cur_grid, prev_grid):
        # the batch index, the starting point and end point for segment
        b_id, s_id, e_id = cur_grid

        batch_ids = self.batch_indexes[b_id]
        rows = [self.data[idx] for idx in batch_ids]
        dialog_lens = [self.data_lens[idx] for idx in batch_ids]

        cur_pos = [np.minimum(1.0, e_id/float(l)) for l in dialog_lens]

        # input_context, context_lens, floors, topics, a_profiles, b_Profiles, outputs, output_lens
        context_utts, context_lens,utt_lens,floors, out_utts,out_lens,out_floors,out_das = [],[],[],[],[],[],[],[]
        for row in rows:
            if s_id < len(row)-1:
                cut_row = row[s_id:e_id]
                in_row = cut_row[0:-1]
                out_row = cut_row[-1]
                out_utt, out_floor = out_row
                
                context_utts.append([self.pad_to(utt) for utt, floor in in_row])
                utt_lens.append([min(len(utt),self.max_utt_size) for utt, floor in in_row])
                context_lens.append(len(cut_row) - 1)
                floors.append([int(floor==out_floor) for utt, floor in in_row])

                out_utt = self.pad_to(out_utt, do_pad=False)
                out_utts.append(out_utt)
                out_lens.append(len(out_utt))
                out_floors.append(out_floor)
            else:
                print(row)
                raise ValueError("S_ID %d larger than row" % s_id)
        #print(context_lens)
        vec_context_lens = np.array(context_lens, dtype=np.int64)
        vec_context = np.zeros((self.batch_size, max(context_lens), self.max_utt_size), dtype=np.int64)
        #print(utt_lens)
        vec_utt_lens = np.ones((self.batch_size, max(context_lens)), dtype=np.int64)+1 #np.array(utt_lens, dtype=np.int64)
        vec_floors = np.zeros((self.batch_size, np.max(vec_context_lens)), dtype=np.int64)
        vec_outs = np.zeros((self.batch_size, np.max(out_lens)), dtype=np.int64)
        vec_out_lens = np.array(out_lens, dtype=np.int64)

        for b_id in range(self.batch_size):
            vec_outs[b_id, 0:vec_out_lens[b_id]] = out_utts[b_id]
            vec_floors[b_id, 0:vec_context_lens[b_id]] = floors[b_id]
            vec_context[b_id, 0:vec_context_lens[b_id], :] = np.array(context_utts[b_id])
            vec_utt_lens[b_id, 0:vec_context_lens[b_id]] = utt_lens[b_id]

        return vec_context, vec_context_lens, vec_utt_lens, vec_floors, None, \
               None, None, vec_outs, vec_out_lens, None
        
        
        