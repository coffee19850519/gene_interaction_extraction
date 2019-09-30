import requests
import sys, os
import json
import excel_concat_excel as ece
import pipeline as pl

# PMC5970946

class write_into_txt_xls(object):

    def writer(self, path, text):
        write_flag = True
        with open(path, 'a', encoding='utf-8') as f:
            #f.write(name + '\n')
            f.writelines(text)
            f.write('\n')

    def writer2(self, path):
        write_flag = True
        with open(path, 'w', encoding='utf-8') as f:
            #f.write(name + '\n')
            #f.writelines(text)
            f.write('\n')

def get_annotated_sentences(pubtatorobj):
    '''
    shape:
        pubtatorobj -- full object
            passages -- a passage of text
                annotations -- identify each labeled phrase
                    text -- phrase identified
                    infons -- information about label
                        type -- labeled entity type (Gene, Disease, Species, etc.)
                text -- the passage written out in plain text


    pubtatorobj is an dictionary
    passages in pubtatorobj contains every passage of text in the article
    annotations in passages contains the annotations and information about the passage
    text in annotation is the identified phrase from 'named entity recogniction' (NER)
    text in passage is the full passage (paragraph, sentence, etc.)

    return:
        result
            fulltext - the text from the article unbroken
            annotations - the full list of annotations from the article based on annotation text
                entityname: location, type

    '''
    result = {} # this result will parse out the pubtatorobj and put it into a different format
    result['fulltext'] = ''
    result['annotations'] = {}
    result['important_sentences'] = set()
    for idx, passage in enumerate(pubtatorobj['passages']):
        annotations = set()
        for annotation in passage['annotations']:
            entityname = annotation['text']
            if entityname not in result['annotations']:
                result['annotations'][entityname] = []
            result['annotations'][entityname].append((annotation['locations'], annotation['infons']['type']))
            annotations.add((annotation['text'], annotation['infons']['type']))

            '''if annotation['infons']['type'] == 'Gene':
                entityname = annotation['text']
                if entityname not in result['annotations']:
                    result['annotations'][entityname] = []
                result['annotations'][entityname].append((annotation['locations'], annotation['infons']['type']))
                annotations.add((annotation['text'], annotation['infons']['type']))'''

        text = passage['text'] # used for identifying the passage text, and building full text
        result['fulltext'] += passage['text'].strip()
        sentences = text.split('.')
        important_sentences = set()
        for idx, sentence in enumerate(sentences):
            entity_count = 0
            for annotation in annotations:
                entity, entity_type = annotation
                if entity in sentence:
                    sentence = '<{}: {}>'.format(entity, entity_type).join(sentence.split(entity))
                    entity_count += 1
            if entity_count > 2 and sentences[idx] not in important_sentences:
                # add original sentence and modified sentence to important sentences
                result['important_sentences'].add((sentences[idx], sentence))
                important_sentences.add(sentences[idx])
    return result

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if len(sys.argv) == 2:
            articleid = sys.argv[1]
            idtype = 'pmcids' if 'PMC' in articleid else 'pmids'
        if len(sys.argv) == 3:
            idtype = sys.argv[2]
            articleid = sys.argv[1]


        response = requests.get('https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/biocjson?{}={}&concepts=gene'.format(idtype, articleid))
        pubtatorobj = json.loads(response.content)
        text = ''
        result = get_annotated_sentences(pubtatorobj)

        filename = 'xxx.txt'
        wt = write_into_txt_xls()
        wt.writer2(filename)
        #result1 = result['annotations'].keys()

        for important_sentence in result['annotations']:
            # print(important_sentence[0])
            # print(' - - - - - - - - - - - - - - - - - - - - - - -')
            # print(important_sentence)
            wt.writer(filename, important_sentence)
            # print('----------------------------------------------')
            # print('----------------------------------------------')

        inputfileTxt = 'xxx.txt'
        outfileExcel = 'text_result.xlsx'
        ece.txt_to_xlsx(inputfileTxt, outfileExcel)
        ece.excel_one_line_to_list()

    else:
        print('Need to specify id and/or id type')

    pl.pipeline_go()
