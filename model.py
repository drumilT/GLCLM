import torch
import torch.nn.init as init
from torch.autograd import Variable
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
import torch.optim as optim

class Attn(nn.Module):
  def __init__(self, hparams):
    super(Attn, self).__init__()
    self.hparams = hparams
    self.dropout = nn.Dropout(hparams.dropout)
    self.w_trg = nn.Linear(self.hparams.d_model, self.hparams.d_model)
    self.w_att = nn.Linear(self.hparams.d_model, 1)

  def forward(self,H,atten_mask=None):
    bsz,max_len,h_dim = H.shape
    assert(h_dim==self.hparams.d_model)
    att_src_hidden = torch.tanh(self.w_trg(H))
    att_src_weights = self.w_att(att_src_hidden).squeeze(2)
    if atten_mask is not None:
      att_src_weights = (
                att_src_weights.float()
                .masked_fill_(atten_mask, float("-inf"))
                .type_as(att_src_weights)
            ) 
    att_src_weights = F.softmax(att_src_weights, dim=-1)
    att_src_weights = self.dropout(att_src_weights)
    ctx = torch.bmm(att_src_weights.unsqueeze(1), H).squeeze(1)
    assert(ctx.shape[0]==bsz and ctx.shape[1]==self.hparams.d_model)
    return ctx

class UttEncoder(nn.Module):
  def __init__(self, hparams, *args, **kwargs):
    super(UttEncoder, self).__init__()

    self.hparams = hparams
    self.word_emb = nn.Embedding(self.hparams.src_vocab_size,
                                 self.hparams.d_word_vec,
                                 padding_idx=hparams.pad_id)

    self.attention = Attn(hparams)
    self.layer = nn.LSTM(self.hparams.d_word_vec,
                         self.hparams.d_model,
                         num_layers=hparams.n_layers,
                         batch_first=True,
                         bidirectional=True,
                         dropout=hparams.dropout)


    self.dropout = nn.Dropout(self.hparams.dropout)

  def forward(self, x_train, x_mask, x_len):
    """Performs a forward pass.
    Args:
      x_train: Torch Tensor of size [batch_size, max_len]
      x_mask: Torch Tensor of size [batch_size, max_len]. 1 means to ignore a
        position.
      x_len: [batch_size,]
    Returns:
      enc_output: Tensor of size [batch_size, max_len, d_model].
      ctx: Tensor of size [batch_size,d_model]
    """  
    batch_size, max_len = x_train.size()
    word_emb = self.word_emb(x_train)
    word_emb = self.dropout(word_emb)
    packed_word_emb = pack_padded_sequence(word_emb, x_len, batch_first=True)
    enc_output, (ht, ct) = self.layer(packed_word_emb)
    enc_output, _ = pad_packed_sequence(enc_output, batch_first=True,
      padding_value=self.hparams.pad_id)
    ctx = self.attention(enc_output, attn_mask=x_mask)

    return enc_output,ctx

class ContextEncoder(nn.Module):
  def __init__(self, hparams, *args, **kwargs):
    super(ContextEncoder, self).__init__()

    self.hparams = hparams
    self.layer = nn.LSTM(self.hparams.d_model,
                         self.hparams.d_ctx_embed,
                         num_layers=hparams.n_layers,
                         batch_first=True,
                         bidirectional=True,
                         dropout=hparams.dropout)
    self.dropout = nn.Dropout(self.hparams.dropout)

  def forward(self, atten_enc,enc_mask):
    bsz,atten_dim = atten_enc.shape
    assert(atten_dim==self.hparams.d_model)
    context_output, (ht, ct) = self.layer(atten_enc)
    assert(context_output.shape[0]==bsz and context_output.shape[1]==self.hparams.d_ctx_embed*2)
    return context_output

class UttDecoder(nn.Module):
  def __init__(self, hparams, word_emb):
    super(UttDecoder, self).__init__()
    self.hparams = hparams
    self.lstm_dim = self.hparams.d_word_vec + 2*self.hparams.d_ctx_embed
    self.layer = nn.LSTMCell(self.lstm_dim, self.hparams.d_dec_out)
    self.dropout = nn.Dropout(self.hparams.dropout)
    self.word_emb = word_emb
    self.readout = nn.Linear(self.hparams.d_dec_out, self.hparams.src_vocab_size,bias= False) 


  def forward(self, x_train, x_mask, x_len, context_left_right,dec_init=None):
    # get decoder init state and cell, use x_ct
    x_wrd_emb = self.word_emb(x_train[:, :-1])
    bsz_x,x_max_len,x_dim = x_wrd_emb.shape
    bsz_c,c_dim = context_left_right.shape
    assert(bsz_x==bsz_c)
    assert(x_dim ==self.hparams.d_word_vec)
    assert(c_dim== 2*self.hparams.d_ctx_embed)
    rep_context = context_left_right.unsqueeze(1)
    rep_context = rep_context.repeat(1,x_max_len,1)
    dec_inp = torch.cat([x_wrd_emb,rep_context], dim=-1)

    pre_readouts = []
    logits = []

    hidden = dec_init

    if dec_init is None:
      hidden = (torch.zeros_like(bsz_x,self.hparams.d_dec_out),torch.zeros_like(bsz_x,self.hparams.d_dec_out))


    for t in range(x_max_len-1):
      dec_inp_step = dec_inp[:, t, :]
      h_t, c_t = self.layer(dec_inp_step, hidden)
      pre_readout = self.dropout(h_t)
      pre_readouts.append(pre_readout)
      hidden = (h_t, c_t)

    # [len_y, batch_size, trg_vocab_size]
    logits = self.readout(torch.stack(pre_readouts)).transpose(0, 1).contiguous()
    return logits
    
  def step(self, x_word_emb, context_left_right, dec_state=None):
    #y_emb_tm1 = self.word_emb(y_tm1)
    y_input = torch.cat([x_word_emb, context_left_right], dim=1)
    if dec_state is not None:
      h_t, c_t = self.layer(y_input, dec_state)
    else:
      h_t, c_t = self.layer(y_input)
    logits = self.readout(h_t)

    return logits, (h_t, c_t)


class GLCLM(nn.Module):
  def __init__(self,hparams,data) -> None:
      super(GLCLM,self).__init__()
      self.hparams = hparams
      self.utt_enc = UttEncoder(hparams)
      self.ctx_enc = ContextEncoder(hparams)
      self.utt_dec = UttDecoder(hparams,word_emb=self.utt_enc.word_emb)
      self.data = data
      self.criteron = BCEWithLogitsLoss()
      #self.optimizer = optim.Adam()

  def forward(self, x_train, x_mask, x_len, y_train, y_mask,
    y_len, eval=False):

    bsz,max_len = x_train.shape
    _,S = self.utt_enc.forward(x_train,x_mask,x_len)
    LR = self.ctx_enc.forward(S)

    if eval: 
      logits = self.utt_dec.forward(x_train,x_mask,x_len,LR)
      assert(logits.shape[0]==bsz and logits.shape[1]==max_len  and logits.shape[2]==  self.hparams.src_vocab_size)
      output = logits.argmax(-1)
      return logits,output,-1
    else:
      #self.optimizer.zero_grad()
      logits = self.utt_dec.forward(x_train,x_mask,x_len,LR)
      assert(logits.shape[0]==bsz and logits.shape[1]==max_len  and logits.shape[2]==  self.hparams.src_vocab_size)
      output = logits.argmax(-1)
      loss = self.criteron(logits,y_train)
      #loss.backward()
      #self.optimizer.step()
      return logits,output,loss