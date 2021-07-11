import random
import numpy as np
import os
import json
import torch

class DataUtil(object):

  def __init__(self, hparams, decode=True):
    self.hparams = hparams

    self.src_i2w, self.src_w2i = self._build_vocab(self.hparams.src_vocab, max_vocab_size=self.hparams.src_vocab_size)
    self.hparams.src_vocab_size = len(self.src_i2w)
    self.src_speaker_limits = json.load(open(hparams.src_speaker_limits_file))
    self.dev_speaker_limits = json.load(open(hparams.dev_speaker_limits_file))
    self.test_speaker_limits = json.load(open(hparams.test_speaker_limits_file))

    print("src_vocab_size={}".format(self.hparams.src_vocab_size))

    if not self.hparams.decode:
      self.train_x = []
      self.train_y = []

      self.train_size = 0
      self.n_train_batches = 0

      train_x_lens = []
      self.train_x, self.train_y, train_x_lens = self._build_parallel(self.hparams.train_src_file, self.hparams.train_trg_file)
      self.train_size = len(self.train_x)

      dev_src_file = self.hparams.dev_src_file
      dev_trg_file = self.hparams.dev_trg_file
      self.dev_x, self.dev_y, src_len = self._build_parallel(dev_src_file, dev_trg_file, is_train=False)
      self.dev_size = len(self.dev_x)
      self.dev_index = 0
    else:
      test_src_file = self.hparams.test_src_file
      test_trg_file = self.hparams.test_trg_file
      self.test_x, self.test_y, src_len = self._build_parallel(test_src_file, test_trg_file, is_train=False)
      self.test_size = len(self.test_x)
      self.test_index = 0

  def load_pretrained(self, pretrained_emb_file):
    f = open(pretrained_emb_file, 'r', encoding='utf-8')
    header = f.readline().split(' ')
    count = int(header[0])
    dim = int(header[1])
    matrix = np.zeros((count, dim), dtype=np.float32)
    i2w = []
    w2i = {}

    for i in range(count):
      word, vec = f.readline().split(' ', 1)
      w2i[word] = len(w2i)
      i2w.append(word)
      matrix[i] = np.fromstring(vec, sep=' ', dtype=np.float32)
    return torch.FloatTensor(matrix), i2w, w2i

  def reset_train(self):
    self.n_train_batches = len(self.src_speaker_limits)
    self.train_queue = np.random.permutation(self.n_train_batches)
    self.train_index = 0

  def next_train(self):
    start_index = self.src_speaker_limits[self.train_queue[self.train_index]]["st"]
    end_index = self.src_speaker_limits[self.train_queue[self.train_index]]["end"]

    x_train = self.train_x[start_index:end_index]
    y_train = self.train_y[start_index:end_index]

    self.train_index += 1
    batch_size = len(x_train)
    y_count = sum([len(y) for y in y_train])
    # pad
    x_train, x_mask, x_count, x_len, x_pos_emb_idxs = self._pad(x_train, self.hparams.pad_id)
    y_train, y_mask, y_count, y_len, y_pos_emb_idxs = self._pad(y_train, self.hparams.trg_pad_id)
    # sample some y
    # y_sampled = [self.sample_y() for _ in range(batch_size)]
    # y_sampled, y_sampled_mask, y_sampled_count, y_sampled_len, y_sampled_pos_emb_idxs = self._pad(y_sampled, self.hparams.trg_pad_id)
    #print("@"*80)
    #print(y_train)
    #y_sampled = torch.LongTensor([ 0 if x>0 else 58 for x in y_train])
    #if self.hparams.cuda:
       # y_sampled = y_sampled.cuda()
    
    y_sampled = (self.hparams.trg_vocab_size-1) * ( y_train == 0 )
   #y_sampled = 1 - y_train
    y_sampled_mask = y_mask
    y_sampled_count = y_count
    y_sampled_len = y_len
    y_sampled_pos_emb_idxs = y_pos_emb_idxs

    if self.train_index >= self.n_train_batches:
      self.reset_train()
      eop = True
    else:
      eop = False
    return x_train, x_mask, x_count, x_len, x_pos_emb_idxs, y_train, y_mask, y_count, y_len, y_pos_emb_idxs, y_sampled, y_sampled_mask, y_sampled_count, y_sampled_len, y_sampled_pos_emb_idxs, batch_size, eop

  def sample_y(self):
    # first how many attrs?
    attn_num = random.randint(1, (self.hparams.trg_vocab_size-1)//2)
    # then select attrs
    y = np.random.binomial(1, 0.5, attn_num)
    y = y + np.arange(attn_num) * 2
    return y.tolist()

  def next_dev(self, dev_batch_size=1, sort=True):
    start_index = self.dev_speaker_limits[self.dev_index]["st"]
    end_index = self.dev_speaker_limits[self.dev_index]["end"]
    batch_size = end_index - start_index

    x_dev = self.dev_x[start_index:end_index]
    y_dev = self.dev_y[start_index:end_index]
    index = None

    x_dev, x_mask, x_count, x_len, x_pos_emb_idxs = self._pad(x_dev, self.hparams.pad_id)
    y_dev, y_mask, y_count, y_len, y_pos_emb_idxs = self._pad(y_dev, self.hparams.trg_pad_id)

    y_neg =  (y_dev == 0 ) * (self.hparams.trg_vocab_size-1)
    #y_neg = 1 - y_dev
		
    if end_index >= self.dev_size:
      eop = True
      self.dev_index = 0
    else:
      eop = False
      self.dev_index += 1

    return x_dev, x_mask, x_count, x_len, x_pos_emb_idxs, y_dev, y_mask, y_count, y_len, y_pos_emb_idxs, y_neg, batch_size, eop, index

  def reset_test(self, test_src_file, test_trg_file):
    self.test_x, self.test_y, src_len = self._build_parallel(test_src_file, test_trg_file, is_train=False)
    self.test_size = len(self.test_x)
    self.test_index = 0

  def next_test(self, test_batch_size=10):
    start_index = self.dev_speaker_limits[self.dev_index]["st"]
    end_index = self.dev_speaker_limits[self.dev_index]["end"]
    batch_size = end_index - start_index

    x_test = self.test_x[start_index:end_index]
    y_test = self.test_y[start_index:end_index]
    index=None

    x_test, x_mask, x_count, x_len, x_pos_emb_idxs = self._pad(x_test, self.hparams.pad_id)
    y_test, y_mask, y_count, y_len, y_pos_emb_idxs = self._pad(y_test, self.hparams.trg_pad_id)

    y_neg = (y_test == 0 ) * (self.hparams.trg_vocab_size -1)
    #y_neg = 1  - y_test
    if end_index >= self.test_size:
      eop = True
      self.test_index = 0
    else:
      eop = False
      self.test_index += 1

    return x_test, x_mask, x_count, x_len, x_pos_emb_idxs, y_test, y_mask, y_count, y_len, y_pos_emb_idxs, y_neg, batch_size, eop, index

  def _pad(self, sentences, pad_id, char_kv=None, char_dim=None, char_sents=None):
    batch_size = len(sentences)
    lengths = [len(s) for s in sentences]
    count = sum(lengths)
    max_len = max(lengths)
    padded_sentences = [s + ([pad_id]*(max_len - len(s))) for s in sentences]

    mask = [[0]*len(s) + [1]*(max_len - len(s)) for s in sentences]
    padded_sentences = torch.LongTensor(padded_sentences)
    mask = torch.ByteTensor(mask)
    pos_emb_indices = [[i+1 for i in range(len(s))] + ([0]*(max_len - len(s))) for s in sentences]
    pos_emb_indices = torch.FloatTensor(pos_emb_indices)
    if self.hparams.cuda:
      padded_sentences = padded_sentences.cuda()
      pos_emb_indices = pos_emb_indices.cuda()
      mask = mask.cuda()
    return padded_sentences, mask, count, lengths, pos_emb_indices

  def _build_parallel(self, src_file_name, trg_file_name, is_train=True):
    print("loading parallel sentences from {} {}".format(src_file_name, trg_file_name))
    with open(src_file_name, 'r', encoding='utf-8') as f:
      src_lines = f.read().split('\n')
    with open(trg_file_name, 'r', encoding='utf-8') as f:
      trg_lines = f.read().split('\n')
    src_data = []
    trg_data = []
    line_count = 0
    skip_line_count = 0
    src_unk_count = 0
    trg_unk_count = 0

    src_lens = []
    src_unk_id = self.hparams.unk_id
    for src_line, trg_line in zip(src_lines, trg_lines):
      src_tokens = src_line.split()
      trg_tokens = trg_line.split()
      if is_train and not src_tokens or not trg_tokens:
        skip_line_count += 1
        continue
      if is_train and self.hparams.max_len and (len(src_tokens) > self.hparams.max_len or len(trg_tokens) > self.hparams.max_len):
        skip_line_count += 1
        continue

      src_lens.append(len(src_tokens))
      src_indices, trg_indices = [self.hparams.bos_id], []
      src_w2i = self.src_w2i
      for src_tok in src_tokens:
        #print(src_tok)
        if src_tok not in src_w2i:
          src_indices.append(src_unk_id)
          src_unk_count += 1
          #print("unk {}".format(src_unk_count))
        else:
          src_indices.append(src_w2i[src_tok])
          #print("src id {}".format(src_w2i[src_tok]))
        # calculate char ngram emb for src_tok

      trg_w2i = self.trg_w2i
      for trg_tok in trg_tokens:
        if trg_tok not in trg_w2i:
          print("trg attribute cannot have oov!")
          exit(0)
        else:
          trg_indices.append(trg_w2i[trg_tok])
        # calculate char ngram emb for trg_tok

      src_indices.append(self.hparams.eos_id)
      src_data.append(src_indices)
      trg_data.append(trg_indices)
      line_count += 1
      if line_count % 10000 == 0:
        print("processed {} lines".format(line_count))
    if is_train:
      src_data, trg_data, _ = self.sort_by_xlen(src_data, trg_data, descend=False)
    print("src_unk={}, trg_unk={}".format(src_unk_count, trg_unk_count))
    assert len(src_data) == len(trg_data)
    print("lines={}, skipped_lines={}".format(len(src_data), skip_line_count))
    return src_data, trg_data, src_lens

  def _build_vocab(self, vocab_file, max_vocab_size=None):
    i2w = []
    w2i = {}
    i = 0
    with open(vocab_file, 'r', encoding='utf-8') as f:
      for line in f:
        w = line.strip()
        w2i[w] = i
        i2w.append(w)
        i += 1
        if max_vocab_size and i >= max_vocab_size:
          break

    return i2w, w2i


