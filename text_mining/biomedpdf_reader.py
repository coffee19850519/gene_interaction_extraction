import gene_dictionary_extractor
from gene_dictionary_reader import gene_dictionary_reader
import extract_text
import gene_stat_collector
import sys
import json
from fuzzy_gene_symbol_reader import fuzzy_gene_symbol_reader


with open('biomedpdf.config') as config:
    config_file = json.load(config)

class BioMedPdfReader:
    def __init__(self):
        gene_dictionary_location=config_file['gene_dictionary_location']
        gene_dictionary = gene_dictionary_extractor.extract_dictionary(gene_dictionary_location)
        self.dictionary_reader = gene_dictionary_reader(gene_dictionary)

    def get_gene_pair_cooccurrence_counts(self, pdflocation):
        paper = extract_text.convert_file_to_text(pdflocation)
        cooccurrence_counts = gene_stat_collector.get_co_occurrences(self.dictionary_reader, paper)
        return cooccurrence_counts


if __name__ == '__main__':
    biomed_reader = BioMedPdfReader()
    counts = biomed_reader.get_gene_pair_cooccurrence_counts('../PapersForTextMining/NSCLC_pathways/Apigenin potentiates TRAIL therapy of non-small cell lung cancer via upregulating DR4_DR5 expression in a p53-dependent manner.pdf')
    print(counts)
    print(gene_stat_collector.get_pair_counts(counts, [('TRAL', 'TRAL'), ('p53', 'DR'), ('TRAL', 'HSP26'), ('DR1', 'DR1')]))
