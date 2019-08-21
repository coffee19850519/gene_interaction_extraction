class Sentence:
    def __init__(self, text, important, genes):
        self.text = text
        self.important = important
        self.genes = genes


def read_sentence(gene_dictionary_reader, sentence, threshold):
    words_in_sentence = sentence.split(' ')
    gene_count = 0
    constructed_sentence = ''
    first_gene = None
    genes_in_sentence = []
    for word in words_in_sentence:
        if not gene_dictionary_reader.examine_word(word):
            extracted_gene_name, valid_name, standard_gene_name = gene_dictionary_reader.extract_name()
            if valid_name:
                standard_gene_name = standard_gene_name[0]
                gene_count += 1
                constructed_sentence += ' <{}: {}>'.format(extracted_gene_name, standard_gene_name)
                genes_in_sentence.append(standard_gene_name)
        if not gene_dictionary_reader.is_reading_name():
            constructed_sentence += ' ' + word
    if gene_count >= threshold:
        return Sentence(text=constructed_sentence, important=True, genes=genes_in_sentence)
    else:
        return Sentence(text=constructed_sentence, important=False, genes=genes_in_sentence)

def get_gene_pairs(gene_dictionary_reader, sentence):
    words_in_sentence = sentence.split(' ')
    constructed_sentence = ''
    genes_in_sentence = []
    for word in words_in_sentence:
        if not gene_dictionary_reader.examine_word(word):
            extracted_gene_name, valid_name, standard_gene_name = gene_dictionary_reader.extract_name()
            if valid_name:
                standard_gene_name = standard_gene_name[0]
                constructed_sentence += ' <{}: {}>'.format(extracted_gene_name, standard_gene_name)
                genes_in_sentence.append(extracted_gene_name)
        if not gene_dictionary_reader.is_reading_name():
            constructed_sentence += ' ' + word
    gene_pairs = []
    i = 0
    while i < len(genes_in_sentence):
        j = i + 1
        while j < len(genes_in_sentence):
            gene_pairs.append((genes_in_sentence[i], genes_in_sentence[j]))
            j += 1
        i += 1
    return gene_pairs

def split_sentences(text):
    return text.split('.')


if __name__ == "__main__":
    import gene_dictionary_extractor
    from gene_dictionary_reader import gene_dictionary_reader
    gene_dictionary = gene_dictionary_extractor.extract_dictionary()
    dictionary_reader = gene_dictionary_reader(gene_dictionary)
    example_sentence = "We next speculated that the observed activation of STAT3 caused by AKR1C1 was probably due to a potential interaction between these proteins ."
    examined_sentence = read_sentence(dictionary_reader, example_sentence,
    2)

    if examined_sentence.important:
        print(examined_sentence.text)
        print(examined_sentence.genes)
        gene_pairs = get_gene_pairs(dictionary_reader, example_sentence)
        print(gene_pairs)
    else:
        print('not so important')
