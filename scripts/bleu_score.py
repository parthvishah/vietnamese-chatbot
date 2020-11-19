## from https://github.com/mjpost/sacreBLEU

from typing import List, Iterable, Tuple
from collections import Counter, namedtuple
from itertools import zip_longest
import math
import logging



class BLEU_SCORE(object):
    def __init__(self, NGRAM_ORDER = 4, DEFAULT_TOKENIZER = lambda x:x ):
        self.NGRAM_ORDER = NGRAM_ORDER # this is the n-gram order of the bleu score we want to calculate. default is BLEU-4.
        self.DEFAULT_TOKENIZER = DEFAULT_TOKENIZER # tokenizer to use, default is none.
        self.BLEU = namedtuple('BLEU', 'score, counts, totals, precisions, bp, sys_len, ref_len') # bleu object to returnn
    
    def _my_log(self, num):
        """
        log(0) is not mathematically defined.
        we use a small number instead of 0 for p2, p3, p4.
        the bleu would be 0 in this case.

        Floors the log function
        :param num: the number
        :return: log(num) floored to a very low number
        """

        if num == 0.0:
            return -9999999999
        return math.log(num)
    
    def _extract_ngrams(self, line, min_order=1, max_order=None):
        """
        Extracts all the ngrams (1 <= n <= NGRAM_ORDER) from a sequence of tokens.
        If max_order = 2, we will get a container of counts for all unigrams and bigrams.
        :param line: a segment containing a sequence of words
        :param max_order: collect n-grams from 1<=n<=max 
        :return: a dictionary containing ngrams and counts
        """
        if max_order is None:
            max_order = self.NGRAM_ORDER
        ngrams = Counter()
        tokens = line.split()
        for n in range(min_order, max_order + 1):
            for i in range(0, len(tokens) - n + 1):
                ngram = ' '.join(tokens[i: i + n])
                ngrams[ngram] += 1

        return ngrams

    def _ref_stats(self, output, refs):
        """
        Extracts reference statistics for a given segment.
        :param refs: A list of segment tokens.
        :param output_len: Hypothesis length for this segment.
        :return: a tuple of (ngrams, closest_diff, closest_len)
        
        ngrams is a counter objects of all ngrams for all reference sentences
        closest_diff is the closest diff of the reference and candidate
        closest_len is the closest length of the referencee and candidate

        example
        -------
        ref1: 'the cat is on the mat' (length 6)
        ref2: 'there is a cat on the mat' (length 7)
        candidate: 'the cat the cat on the mat' (length 7)
        
        closest_len would be 7, closest_diff would be 0
        """
        ngrams = Counter()
        closest_diff = None
        closest_len = None
        for ref in refs:
            tokens = ref.split()
            reflen = len(tokens)
            # check the len diff between candidate and reference
            diff = abs(len(output.split()) - reflen)
            # update closest_diff 
            if closest_diff is None or diff < closest_diff:
                closest_diff = diff
                closest_len = reflen
            elif diff == closest_diff:
                if reflen < closest_len:
                    closest_len = reflen

            ngrams_ref = self._extract_ngrams(ref)
            for ngram in ngrams_ref.keys():
                ngrams[ngram] = max(ngrams[ngram], ngrams_ref[ngram])

        return ngrams, closest_diff, closest_len

    def _compute_bleu(self, correct, total, sys_len, ref_len, smooth = 'none', smooth_floor = 0.01,
                 use_effective_order = False):
        """Computes BLEU score from its sufficient statistics. Adds smoothing.
        :param correct: List of counts of correct ngrams, 1 <= n <= NGRAM_ORDER
        :param total: List of counts of total ngrams, 1 <= n <= NGRAM_ORDER
        :param sys_len: The cumulative system length
        :param ref_len: The cumulative reference length
        :param smooth: The smoothing method to use ('exp','floor')
        Smoothing methods (citing "A Systematic Comparison of Smoothing Techniques for Sentence-Level BLEU",
        Boxing Chen and Colin Cherry, WMT 2014: http://aclweb.org/anthology/W14-3346)
        - exp: Method 3: NIST smoothing method 
        - floor: Method 1 (just add small positive value to smooth)
        - none: do nothing.
        :param smooth_floor: The smoothing value added, if smooth method 'floor' is used
        :param use_effective_order: If true, use the length of `correct` for the n-gram order instead of NGRAM_ORDER.
        :return: A BLEU object with the score (100-based) and other statistics.
        """

        precisions = [0 for x in range(self.NGRAM_ORDER)]

        smooth_mteval = 1.
        effective_order = self.NGRAM_ORDER
        for n in range(self.NGRAM_ORDER):
            if total[n] == 0:
                break

            if use_effective_order:
                effective_order = n + 1

            # if there are no n-gram matches, we use smoothing
            if correct[n] == 0:
                # NIST smoothing method (method 3)
                if smooth == 'exp':
                    smooth_mteval *= 2
                    precisions[n] = 100. / (smooth_mteval * total[n])
                # floor smoothing (method 1 in Chen paper)
                elif smooth == 'floor':
                    precisions[n] = 100. * smooth_floor / total[n]
            # no smoothing, do nothing
            else:
                precisions[n] = 100. * correct[n] / total[n]

        # If the system guesses no i-grams, 1 <= i <= NGRAM_ORDER, the BLEU score is 0 (technically undefined).
        # This is a problem for sentence-level BLEU or a corpus of short sentences, where systems will get no credit
        # if sentence lengths fall under the NGRAM_ORDER threshold. This fix scales NGRAM_ORDER to the observed
        # maximum order. It is only available through the API and off by default

        # add BP penalty
        brevity_penalty = 1.0
        if sys_len < ref_len:
            brevity_penalty = math.exp(1 - ref_len / sys_len) if sys_len > 0 else 0.0

        # calculat BLEU score
        bleu = brevity_penalty * math.exp(sum(map(self._my_log, precisions[:effective_order])) / effective_order)

        # create BLEU instance of named tuple type
        return self.BLEU._make([bleu, correct, total, precisions, brevity_penalty, sys_len, ref_len])
    
    def corpus_bleu(self, sys_stream, ref_streams, smooth='exp', smooth_floor=0.0, force=False, lowercase=False,
                tokenize=None, use_effective_order=False):
        """Produces BLEU scores along with its sufficient statistics from a source against one or more references.
        :param sys_stream: The system stream (a sequence of segments)
        :param ref_streams: A list of one or more reference streams (each a sequence of segments)
        :param use_effective_order: Account for references that are shorter than the largest n-gram.
        :param smooth: The smoothing method to use
        :param smooth_floor: For 'floor' smoothing, the floor to use
        :param force: Ignore data that looks already tokenized
        :param lowercase: Lowercase the data
        :param tokenize: The tokenizer to use
        :return: a BLEU object containing everything you'd want
        """
        if tokenize is None:
            tokenize = self.DEFAULT_TOKENIZER
        # Add some robustness to the input arguments
        if isinstance(sys_stream, str):
            sys_stream = [sys_stream]
        if isinstance(ref_streams, str):
            ref_streams = [[ref_streams]]

        sys_len = 0
        ref_len = 0

        correct = [0 for n in range(self.NGRAM_ORDER)]
        total = [0 for n in range(self.NGRAM_ORDER)]

        # look for already-tokenized sentences
        tokenized_count = 0

        fhs = [sys_stream] + ref_streams
        for lines in zip_longest(*fhs):
            if None in lines:
                raise EOFError("Source and reference streams have different lengths!")

            if lowercase:
                lines = [x.lower() for x in lines]

            if not (force or tokenize == 'none') and lines[0].rstrip().endswith(' .'):
                tokenized_count += 1

#                 if tokenized_count == 100:
#                     logging.warning('That\'s 100 lines that end in a tokenized period (\'.\')')
#                     logging.warning('It looks like you forgot to detokenize your test data, which may hurt your score.')
#                     logging.warning('If you insist your data is detokenized, or don\'t care, you can suppress this message with \'--force\'.')

            output, *refs = [tokenize(x.rstrip()) for x in lines]

            ref_ngrams, closest_diff, closest_len = self._ref_stats(output, refs)

            sys_len += len(output.split())
            ref_len += closest_len

            sys_ngrams = self._extract_ngrams(output)
            for ngram in sys_ngrams.keys():
                n = len(ngram.split())
                correct[n-1] += min(sys_ngrams[ngram], ref_ngrams.get(ngram, 0))
                total[n-1] += sys_ngrams[ngram]

        return self._compute_bleu(correct, total, sys_len, ref_len, smooth, smooth_floor, use_effective_order)