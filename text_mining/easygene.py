import gene_dictionary_extractor
from gene_dictionary_reader import gene_dictionary_reader

gene_dictionary = gene_dictionary_extractor.extract_dictionary()
gdr = gene_dictionary_reader(gene_dictionary)

def standard_name(gene_name):
    split_gene_name = gene_name.split(' ')
    for word in split_gene_name:
        if not gdr.examine_word(word):
            return ([], False)
    extracted_name, is_valid, standard_name = gdr.extract_name()
    return standard_name, is_valid
