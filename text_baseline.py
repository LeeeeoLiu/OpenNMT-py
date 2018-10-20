# coding=utf8
from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.summarizers.lex_rank import LexRankSummarizer as LexRank
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

import argparse
from rouge import Rouge

def LexRank_Text(originalText, LANGUAGE="chinese"):
    """Get LexRank output from a text.

    Get LexRank output from a text.

    Args:
        originalText: Text.
        LANGUAGE: The language of text.
       
    Returns:
        str
    """
    #print(originalText)
    parser = PlaintextParser.from_string(originalText, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)
    summarizer = LexRank(stemmer)
    # summarizer.stop_words = get_stop_words(LANGUAGE)
    # print(summarizer(parser.document, 1))
    for sentence in summarizer(parser.document, 1):
        return str(sentence)


def read_and_filter(_file_path):
    with open(_file_path, 'r') as f:
        _new_lines = []
        for line in f:
            line = line.decode('utf-8')
            sessions = line.strip('\n').split('||')
            #if len(sessions) < 20:
            for s in sessions:
                assert len(s.split('\t')) == 11
            item_name = [s.split('\t')[9].split() for s in sessions]
            item_comment = [s.split('\t')[10].split() for s in sessions]
            _new_line = []
            for tmp_name, tmp_comment in zip(item_name, item_comment):
                _new_line.extend(tmp_name)
                _new_line.extend(tmp_comment)
            _new_lines.append(_new_line)
        return _new_lines
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parameters description")
    parser.add_argument('-src', type=str, help="Source File Path")
    parser.add_argument('-tgt', type=str, help="Target File Path")
    parser.add_argument('-baseline', type=str,
                        help="Baseline Method for Explanation Generation")
    args = parser.parse_args()
    
    assert args.baseline in ['lexrank']
    if args.baseline == 'lexrank':
        _src_lines = read_and_filter(args.src)
        _tgt_lines = read_and_filter(args.tgt)
        assert len(_src_lines) == len(_tgt_lines)
        LANGUAGE="chinese"
        _sum_lines = []
        i = 0
        for _line in _src_lines:
            print('process: {}/{}'.format(i,len(_src_lines)))
            _sum_line = LexRank_Text(_line, LANGUAGE) if LexRank_Text(_line, LANGUAGE) != None else ''
            _sum_lines.append(_sum_line)
            i += 1
        #_sum_lines = [LexRank_Text(_line, LANGUAGE) if LexRank_Text(_line, LANGUAGE) != None else '' for _line in _src_lines]
        rouge = Rouge()
        scores = rouge.get_scores(_sum_lines, _tgt_lines, avg=True)
        
        _sentence_msg = 'ROUGE-1: P:{}\tR:{}\tF1:{}'.format(
            scores['rouge-1']['p'], scores['rouge-1']['r'], scores['rouge-1']['f'])
        _sentence_msg += '\nROUGE-2: P:{}\tR:{}\tF1:{}'.format(
            scores['rouge-2']['p'], scores['rouge-2']['r'], scores['rouge-2']['f'])
        _sentence_msg += '\nROUGE-L: P:{}\tR:{}\tF1:{}'.format(
            scores['rouge-l']['p'], scores['rouge-l']['r'], scores['rouge-l']['f'])
        print(_sentence_msg)
