""" Onmt NMT Model base class definition """
import torch.nn as nn


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """

    def __init__(self,session_encoder, encoder, decoder, multigpu=False):
        self.multigpu = multigpu
        super(NMTModel, self).__init__()
        self.session_encoder = session_encoder
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, session, user, stm, src, tgt, session_lengths, lengths, dec_state=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """
        if self.session_encoder is not None:
            click_score, session_final = self.session_encoder(session, user, stm, session_lengths)
        else:
            click_score = None
            session_final = None
        tgt = tgt[:-1]  # exclude last target from inputs

        enc_final, memory_bank = self.encoder(src, lengths)
        if self.session_encoder is not None:
            if isinstance(enc_final, tuple):  # LSTM
                enc_final = tuple([enc_hid+session_final for enc_hid in enc_final])
            else:  # GRU
                enc_final = enc_final + session_final
        else:
            if isinstance(enc_final, tuple):  # LSTM
                enc_final = tuple([enc_hid for enc_hid in enc_final])
            else:  # GRU
                enc_final = enc_final
        enc_state = \
            self.decoder.init_decoder_state(src, memory_bank, enc_final)
        decoder_outputs, dec_state, attns = \
            self.decoder(tgt, memory_bank,
                         enc_state if dec_state is None
                         else dec_state,
                         memory_lengths=lengths)
        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None
        return click_score, decoder_outputs, attns, dec_state
