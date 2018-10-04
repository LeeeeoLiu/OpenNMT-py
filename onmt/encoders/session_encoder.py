"""Define RNN-based encoders."""
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from onmt.encoders.encoder import EncoderBase
from onmt.utils.rnn_factory import rnn_factory


class RNNEncoder(EncoderBase):
    """ A generic recurrent neural network encoder.

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """

    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, embeddings=None,
                 use_bridge=False):
        super(RNNEncoder, self).__init__()
        assert embeddings is not None

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings

        self.rnn, self.no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=embeddings.embedding_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        bidirectional=bidirectional)


    def forward(self, src, lengths=None):
        "See :obj:`EncoderBase.forward()`"
        self._check_args(src, lengths)

        emb = self.embeddings(src)
        # s_len, batch, emb_dim = emb.size()

        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Tensor.
            lengths = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths)

        memory_bank, encoder_final = self.rnn(packed_emb)

        if lengths is not None and not self.no_pack_padded_seq:
            memory_bank = unpack(memory_bank)[0]
        return encoder_final, memory_bank

class MConn(nn.Module):
    """ My custom connection module
    """
    def __init__(self, _dim_1, _dim_2, _dim_3, _linear=False, _ln_size=None):
        super(MConn, self).__init__()
        self.linear1 = nn.Linear(_dim_1, _dim_2)
        self.linear2 = nn.Linear(_dim_2, _dim_3)
        if not _linear:
            # residual connect weight
            self.linearw = nn.Linear(_dim_1, _dim_3)
            self.USE_RS = True
        else:
            self.USE_RS = False
        if _ln_size != None:
            # layer norm
            self.layer_norm = nn.LayerNorm(_ln_size)
            self.USE_LN = True
        else:
            self.USE_LN = False

    def forward(self, _input):
        _output = self.linear2(F.leaky_relu(F.dropout(self.linear1(_input), p=0.3), inplace=True))
        if self.USE_RS:
            output = F.dropout(self.linearw(_input), p=0.3)
            return self.layer_norm(_output + output) if self.USE_LN else _output + output
        else:
            return self.layer_norm(_output) if self.USE_LN else _output


class MConnBlock(nn.Module):
    """  My custom connection block module
    """

    def __init__(self, _dim_1, _dim_2, _dim_3, _linear=False, _ln_size=None):
        super(MConnBlock, self).__init__()
        _mid_ln_size = (_ln_size[0], _dim_2) if _ln_size else None
        self.MConn1 = MConn(_dim_1, _dim_2, _dim_2, _linear, _mid_ln_size)
        # self.MConn2 = MConn(_dim_2, _dim_2, _dim_2, _linear, _mid_ln_size)
        # self.MConn3 = MConn(_dim_2, _dim_2, _dim_2, _linear, _mid_ln_size)
        # self.MConn4 = MConn(_dim_2, _dim_2, _dim_2, _linear, _mid_ln_size)
        # self.MConn5 = MConn(_dim_2, _dim_2, _dim_2, _linear, _mid_ln_size)
        self.MConn6 = MConn(_dim_2, _dim_2, _dim_2, _linear, _mid_ln_size)
        self.MConn7 = MConn(_dim_2, _dim_2, _dim_3, _linear, _ln_size)

    def forward(self, _input):
        _output = self.MConn1(_input)
        # _output = self.MConn2(_output)
        # _output = self.MConn3(_output)
        # _output = self.MConn4(_output)
        # _output = self.MConn5(_output)
        _output = self.MConn6(_output)
        _output = self.MConn7(_output)
        return _output



class SessionEncoder(nn.Module):
    def __init__(self, item_embeddings, user_log_embeddings, user_op_embeddings, user_site_cy_embeddings, user_site_pro_embeddings, user_site_ct_embeddings, shd=100, nl=1):
        super(SessionEncoder, self).__init__()
        

        """ Session Click Prediction Part"""
        # self.uirc = MConn(100/self.nl, self.shd/2, self.shd, _ln_size=(self.nl, self.shd))

        self.items_embeddings = item_embeddings
        self.user_log_embeddings =user_log_embeddings
        self.user_op_embeddings = user_op_embeddings
        self.user_site_cy_embeddings = user_site_cy_embeddings
        self.user_site_pro_embeddings = user_site_pro_embeddings
        self.user_site_ct_embeddings = user_site_ct_embeddings

        self.sed = self.items_embeddings.emb_luts[0].embedding_dim  # session embedding dim
        self.shd = shd  # session hidden dim
        self.ivs = self.items_embeddings.emb_luts[0].num_embeddings  # item vocab size
        self.nl = nl    # num layer
        self.chd = 500
        

        # self.user_embed = UIEmbedding(_device)
        # local encoder GRU
        self.gru_l = nn.GRU(self.sed, self.shd, self.nl, batch_first=True)
        # global encoder GRU
        self.gru_g = nn.GRU(self.sed, self.shd, self.nl, batch_first=True)
        self.atten_1 = nn.Linear(self.shd, self.shd)
        self.atten_2 = nn.Linear(self.shd, self.shd)
        self.bl = nn.Linear(self.shd, 1)
        self.ctMB = MConnBlock(self.nl*2+5, self.nl*2, 1, _ln_size=(self.shd,1))
        self.ctTS = MConnBlock(self.shd, self.shd+self.chd,self.chd*2 ,_ln_size=(1,self.chd*2))
        # self.ierc = MConnBlock(self.sed, self.sed+self.shd, self.shd)
        # self.s2crc = MConnBlock(self.shd, self.shd+self.chd,self.chd ,_ln_size=(1,self.chd))

    def forward(self, session, user, stm, session_lengths):

        # Get Session Embedding.
        session_embed = self.items_embeddings(session.unsqueeze(2)) # len X Batch X embedding dim

        # Get User Embedding. 
        log_embed = self.user_log_embeddings(
            user[:, 0].unsqueeze(0).unsqueeze(2))
        op_embed = self.user_op_embeddings(
            user[:, 1].unsqueeze(0).unsqueeze(2))
        cy_embed = self.user_site_cy_embeddings(
            user[:, 2].unsqueeze(0).unsqueeze(2))
        pro_embed = self.user_site_pro_embeddings(
            user[:, 3].unsqueeze(0).unsqueeze(2))
        ct_embed = self.user_site_ct_embeddings(
            user[:, 4].unsqueeze(0).unsqueeze(2))
        user_embed_list = [log_embed, op_embed, cy_embed, pro_embed, ct_embed]
        user_embed = torch.cat(user_embed_list, 0) # 5 X Batch X embedding dim

        # STM len  x batch

        packed_session_emb = session_embed
        if session_lengths is not None:
            # Lengths data is wrapped inside a Tensor.
            session_lengths = session_lengths.view(-1).tolist()
            packed_session_emb = pack(session_embed, session_lengths)

        _, hidden_g = self.gru_g(packed_session_emb)
        c_g = hidden_g[-1:]  # 1 X batch X embedding

        output_l, hidden_l = self.gru_l(packed_session_emb) 
        # hidden_l  num layzer X batch X embedding
        unpacked_output, _ = unpack(output_l)   # len X Batch X embedding dim

        # Get Attention
        ht_ex = hidden_l[-1:].expand_as(unpacked_output)    # for computing easily [len X Batch X embedding dim]
        _atten_p1 = F.dropout(self.atten_1(unpacked_output), p=0.3)
        _atten_p2 = F.dropout(self.atten_2(ht_ex), p=0.3)
        atten_1 = self.bl(F.leaky_relu(_atten_p1+_atten_p2, inplace=True))  # len X batch X 1
        atten_2 = (stm.float()/stm.max(0)[0].unsqueeze(0).expand_as(stm).float()).unsqueeze(2) # len X batch X 1
        atten_a = atten_1 + atten_2 # len X batch X 1
        c_l = torch.sum(atten_a*unpacked_output,0).unsqueeze(0) # len x Batch X embedding dim

        # Get Sequntial Represtation
        c_t = torch.cat((user_embed, c_l, c_g), 0)  # 7 x batch x embedding dim
        c_t_trans = self.ctMB(c_t.permute(1,2,0)).squeeze() # batch x embedding dim

        # Get Item Embeddings
        item_indices = torch.linspace(0, self.ivs -1, steps=self.ivs).long().to(c_t.device)
        item_embeddings = self.items_embeddings(item_indices.unsqueeze(0).unsqueeze(2)).squeeze()  # [ivs X  embedding dim]

        # Calculate Click Score
        click_score = F.log_softmax(torch.mm(c_t_trans, item_embeddings.permute(1,0)), dim=1) # [b X ivs]
        concate_ret = (self.ctTS(c_t_trans.unsqueeze(1)).view(click_score.size(0),2,-1)).permute(1,0,2)   # [2 X b X chd]

        return click_score, concate_ret
