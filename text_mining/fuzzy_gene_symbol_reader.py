from fuzzywuzzy.fuzz import ratio

class fuzzy_gene_symbol_reader:
    def __init__(self, gene_dictionary):
        self.__current_gene = ''
        self.__gene_dictionary = gene_dictionary
        self.__threshold = 70
        self.__confidence = 0

    def examine_word(self, word):
        closest_match = 0
        match = None
        for gene in self.__gene_dictionary:
            r = ratio(word, gene)
            if not match:
                match = gene
                closest_match = r
            elif r > closest_match:
                match = gene
                closest_match = r
        if closest_match >= self.__threshold:
            self.__current_gene = match
            self.__confidence = closest_match
            return True
        return False

    def extract_name(self):
        valid = self.__confidence > self.__threshold
        gene = self.__current_gene
        self.__confidence = 0
        self.__current_gene = ''
        return gene, valid, [gene]

    def is_reading_name(self):
        return len(self.__current_gene) > 0
