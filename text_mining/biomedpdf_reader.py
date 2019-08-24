from . import gene_dictionary_extractor
from .gene_dictionary_reader import gene_dictionary_reader
from . import extract_text
from . import gene_stat_collector
import sys
import json
from .fuzzy_gene_symbol_reader import fuzzy_gene_symbol_reader

class biomedpdf_reader:
    def __init__(self, gene_list):
        gene_dictionary = gene_dictionary_extractor.extract_dictionary(gene_list)
        self.dictionary_reader = gene_dictionary_reader(gene_dictionary)

    def get_gene_pair_cooccurrence_counts(self, pdflocation):
        paper = extract_text.convert_file_to_text(pdflocation)
        cooccurrence_counts = gene_stat_collector.get_co_occurrences(self.dictionary_reader, paper)
        return cooccurrence_counts


if __name__ == '__main__':
    biomed_reader = biomedpdf_reader()
    counts = biomed_reader.get_gene_pair_cooccurrence_counts('apigeninExample.pdf')
    print(counts)
    print(gene_stat_collector.get_pair_counts(counts, [('TRAL', 'TRAL'), ('p53', 'DR'), ('TRAL', 'HSP26'), ('DR1', 'DR1')]))
