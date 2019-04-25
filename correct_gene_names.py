import fuzzywuzzy
from fuzzywuzzy import process, fuzz
from fuzzywuzzy.process import default_scorer, default_processor
from loadexcldata import load_genename_from_excl

# import os, unicodedata, re
# from alignment.sequence import Sequence
# from alignment.vocabulary import Vocabulary
# from alignment.sequencealigner import SimpleScoring, GlobalSequenceAligner

# def results_alignment(result_words, library_words):
#     # Create a vocabulary and encode the sequences.
#     v = Vocabulary()
#     aEncoded = v.encodeSequence(result_words)
#     bEncoded = v.encodeSequence(library_words)
#
#     # Create a scoring and align the sequences using global aligner.
#     scoring = SimpleScoring(2, -1)
#     aligner = GlobalSequenceAligner(scoring, -2)
#     #score, encodeds = aligner.align(aEncoded, bEncoded, backtrace= True)
#     score = aligner.align(aEncoded, bEncoded, backtrace= False)
#     return score
#
#
#
# def deburr(input_str):
#     nfkd_form = unicodedata.normalize('NFKD', input_str)
#     return [u"".join([c for c in nfkd_form if not unicodedata.combining(c)])]
#
#
# def nfkc(word):
#     return [unicodedata.normalize("NFKC", word)]
#
# def upper(word):
#     return [word.upper()]
#
# # NOTE: all single letter symbols have already been removed from the lexicon
# # NOTE: all double letter symbols have already been removed from prev_symbol, alias_symbol;
# #       some remain from current HGNC symbols and bioentities sources, e.g., GK, GA and HR.
# # NOTE: entries should be upper and alphanumeric-only
# stop_list = ["2", "CO2", "HR", "GA", "CA2", "TYPE",
#         "DAMAGE", "GK", "S21", "TAT", "L10","CYCLIN",
# 	"CAMP","FOR","DAG","PIP","FATE","ANG",
# 	"NOT","CAN","MIR","CEL","ECM","HITS","AID","HDS",
# 	"REG","ROS", "D1", "CALL", "BEND3"]
#
# normalize_re = re.compile('[^a-zA-Z0-9]')
#
# def alphanumeric(word):
#     return [normalize_re.sub('', word)]
#
# def stop(word):
#     alphanumerics = alphanumeric(word)
#     if len(alphanumerics) > 0 and alphanumerics[0].upper() not in stop_list:
#         return [word]
#     else:
#         return []
#
#
#
# #it needs modification for word-to-word mode

# def map_result_to_dictionary(test_results, user_words, bounding_box):
#
#     results = []
#     for test, idx in zip(test_results, range(len(test_results))):
#       if test == '':
#         continue
#       word_score = {}
#       for word in user_words:
#         score = results_alignment(Sequence(test), Sequence(
#             word))
#         word_score.update({word: score})
#
#       # find out corrective word with best score
#       best_correction, best_score = sorted(word_score.items(),
#                                            key=lambda word_score: word_score[1],
#                                            reverse=True)[0]
#
#       # treat harf of length of test as threshold
#       if float(best_score) > len(test) / 2.0:
#         #match one gene name
#         results.append(str(best_correction))
#       else:
#         #bounding_box.remove(idx)
#         results.append('[]\n')
#
#       del word_score
#
#     return results

#
# def map_result_to_dictionary(test_results, user_words):
#   results = []
#   for test, idx in zip(test_results, range(len(test_results))):
#     if test == '':
#       results.append('[]\n')
#       continue
#     word_score = {}
#     for word in user_words:
#       score = results_alignment(Sequence(test), Sequence(
#           word))
#       word_score.update({word: score})
#
#     # find out corrective word with best score
#     best_correction, best_score = sorted(word_score.items(),
#                                          key=lambda word_score: word_score[1],
#                                          reverse=True)[0]
#
#     # treat harf of length of test as threshold
#     if float(best_score) > len(test) / 2.0:
#       # match one gene name
#       results.append(str(best_correction))
#     else:
#       results.append('[]\n')
#
#     del word_score
#
#   return results

# def postprocessing_OCR(word, word_dictionary, bonding_box):
#     # word = deburr(word)
#     # #word = upper(word)
#     # word = nfkc(word)
#     # word = alphanumeric(word)
#     # word = stop(word)
#     return map_result_to_dictionary(word, word_dictionary, bonding_box)


def map_result_to_dictionary(test_results, user_words):

    results = []
    for test in test_results:
        if test == '':
            results.append('[]\n')
            continue
        correction=process.extractOne(test,user_words,processor=default_processor,scorer=fuzz.QRatio,score_cutoff=95)
        if correction is not None:
            results.append(correction[0])
        else:
            results.append(test)
    del correction

    return results




if __name__ == '__main__':
    # test_file = r'C:\Users\LSC-110\Desktop\text_results_from_to' \
    #             r'\cin_00094.txttextpredict_to.txt'
    test_file = r'D:\Study\Master\1st deep learning project\Dima data(2)\Dima data\imagetest\cin_00004_OCR1.txt'
    OCR_results=[]
    bounding_results=[]
    filename = r'D:\Study\Master\1st deep learning project\Dima data(2)\Dima data\dictionary.xlsx'
    user_words= load_genename_from_excl(filename)
    with open(test_file, 'r') as test_fp:
      test_results = test_fp.readlines()

    for line in test_results:
        OCR_result = line.split('\t', 1)[0]
        bounding_box=line.split('\t', 1)[1]
        OCR_results.append(OCR_result)
        bounding_results.append(bounding_box)
    corrections = map_result_to_dictionary(OCR_results, user_words)
    with open(test_file[:-4]+'_correction.txt','w') as res_fp:
        for idx in range(len(corrections)):
            res_fp.write(str(corrections[idx])+'\t'+str(bounding_results[idx]))
    del corrections