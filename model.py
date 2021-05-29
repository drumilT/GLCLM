import torch
import torch.nn.init as init
from torch.autograd import Variable
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


class Attn(nn.Module):
  def __init__(self, hparams):
    super(Attn, self).__init__()
    self.hparams = hparams
    self.dropout = nn.Dropout(hparams.dropout)
    self.w_trg = nn.Linear(self.hparams.d_model, self.hparams.d_model)
    self.w_att = nn.Linear(self.hparams.d_model, 1)

  def forward(self,H,atten_mask=None):
    bsz,h_dim = H.shape
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
    self.layer = nn.LSTM(self.hparams.d_word_vec,
                         self.hparams.d_model,
                         num_layers=hparams.n_layers,
                         batch_first=True,
                         bidirectional=True,
                         dropout=hparams.dropout)
    self.dropout = nn.Dropout(self.hparams.dropout)

  def forward(self, atten_enc,enc_mask):
    """Performs a forward pass.
    Args:
      x_train: Torch Tensor of size [batch_size, max_len]
      x_mask: Torch Tensor of size [batch_size, max_len]. 1 means to ignore a
        position.
      x_len: [batch_size,]
    Returns:
      enc_output: Tensor of size [batch_size, max_len, d_model].
    """
    context_output, (ht, ct) = self.layer(atten_enc)
    return context_output

class Decoder(nn.Module):
  def __init__(self, hparams, word_emb):
    super(Decoder, self).__init__()
    

  def forward(self, x_enc, x_enc_k, dec_init, x_mask, y_train, y_mask, y_len, x_train, x_len,x_old_mask):
    # get decoder init state and cell, use x_ct
    """
    x_enc: [batch_size, max_x_len, d_model * 2]
    """
    
    
   