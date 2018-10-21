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
import os

import sys 
reload(sys)
sys.setdefaultencoding('utf-8') 
sys.setrecursionlimit(1000000)

def LexRank_Text(originalText, LANGUAGE="chinese"):
    """Get LexRank output from a text.

    Get LexRank output from a text.

    Args:
        originalText: Text.
        LANGUAGE: The language of text.
       
    Returns:
        str
    """
    parser = PlaintextParser.from_string(originalText, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)
    summarizer = LexRank(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    # print(summarizer(parser.document, 1))
    for sentence in summarizer(parser.document, 1):
        return str(sentence)


def read_and_filter(_file_path):
    with open(_file_path, 'r') as f:
        _new_lines = []
        _session_length = []
        for line in f:
            #print('----')
            #line = line.decode('utf-8')
            sessions = line.strip('\n').split('||')
            for s in sessions:
                assert len(s.split('\t')) == 11
            item_name = [s.split('\t')[9].split() for s in sessions]
            item_comment = [s.split('\t')[10].split() for s in sessions]
            _new_line = []
            for tmp_name, tmp_comment in zip(item_name, item_comment):
                _new_line.extend(tmp_name)
                _new_line.extend(tmp_comment)
            _new_lines.append(_new_line)
            _session_length.append(len(tmp_name))
        return _new_lines, _session_length
        

def write_file(_lines, _path):
    #print(_lines)
    #_lines = [_line.encode('utf-8') for _line in _lines]
    #print(_lines)
    with open(_path, 'w') as f:
        f.write('\n'.join(_lines))

def read_file(_path):
    with open(_path, 'r') as f:
        lines = [line.strip('\n') for line in f]
        return lines


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parameters description")
    parser.add_argument('-src', type=str, help="Source file path.")
    parser.add_argument('-tgt', type=str, help="Target file path.")
    parser.add_argument('-baseline', type=str,
                        help="Baseline Method for Explanation Generation.")
    parser.add_argument('-save', type=str, help="Path to save the output file.")
    parser.add_argument('-test', type=int,default=-1, help='Test the code with a small number of examples.')
    parser.add_argument('-re', type=bool, default= False, help="Reprocess data.")
    args = parser.parse_args()
    LANGUAGE = "chinese"

    assert args.baseline in ['lexrank']
    if args.baseline == 'lexrank':
        if os.path.exists(args.save+'/lexrank_src.txt') and os.path.exists(args.save+'/lexrank_tgt.txt') and args.re==False:
            src_lines = read_file(args.save+'/lexrank_src.txt')
            tgt_lines = read_file(args.save+'/lexrank_tgt.txt')
        else:
            _src_lines, _lengths = read_and_filter(args.src)
            #print(_lengths)
            _tgt_lines, _ = read_and_filter(args.tgt)
            src_lines = []
            tgt_lines = []
            assert len(_src_lines) == len(_tgt_lines), len(
                _src_lines) == len(_lengths)
            for src_line, tgt_line, length in zip(_src_lines, _tgt_lines, _lengths):
                if length < 20:
                    src_lines.append(' '.join(src_line))
                    tgt_lines.append(' '.join(tgt_line))
            write_file(src_lines, args.save+'/lexrank_src.txt')
            write_file(tgt_lines, args.save+'/lexrank_tgt.txt')
        
        _sum_lines = []
        if args.test > 0:
            src_lines = src_lines[0:args.test]
            tgt_lines = tgt_lines[0:args.test]
        for i, _line in enumerate(src_lines):
            print('process: {}/{}'.format(i,len(src_lines)))
            _sum_line = LexRank_Text(_line, LANGUAGE) if LexRank_Text(_line, LANGUAGE) != None else 'None'
            _sum_lines.append(_sum_line)

        assert len(_sum_lines) == len(tgt_lines)
        rouge = Rouge()
        write_file(_sum_lines, args.save+'/lexrank_sumy.txt')
        scores = rouge.get_scores(_sum_lines, tgt_lines, avg=True)
        
        _sentence_msg = 'ROUGE-1: P:{}\tR:{}\tF1:{}'.format(
            scores['rouge-1']['p'], scores['rouge-1']['r'], scores['rouge-1']['f'])
        _sentence_msg += '\nROUGE-2: P:{}\tR:{}\tF1:{}'.format(
            scores['rouge-2']['p'], scores['rouge-2']['r'], scores['rouge-2']['f'])
        _sentence_msg += '\nROUGE-L: P:{}\tR:{}\tF1:{}'.format(
            scores['rouge-l']['p'], scores['rouge-l']['r'], scores['rouge-l']['f'])
        print(_sentence_msg)
        

        
