class gene_dictionary_reader:
    def __init__(self, gene_dictionary):
        self.__current_gene = ''
        self.__gene_dictionary = gene_dictionary
        self.__current_page = self.__gene_dictionary

    def examine_word(self, word):
        if word in self.__current_page[0]:
            self.__current_page = self.__current_page[0][word]
            self.__current_gene = self.__current_gene + ' ' + word
            return True
        return False

    def extract_name(self):
        extracted_gene_name = self.__current_gene
        valid_gene_name = self.__current_page[1]
        self.__current_gene = ''
        self.__current_page = self.__gene_dictionary
        return extracted_gene_name.strip(), valid_gene_name

    def is_reading_name(self):
        return len(self.__current_gene) > 0
