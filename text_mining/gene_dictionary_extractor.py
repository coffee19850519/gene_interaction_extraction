def add_gene(gene_dictionary, gene_name):
    split_gene_name = gene_name.split(' ')
    sub_dictionary = gene_dictionary
    for name in split_gene_name:
        if name not in sub_dictionary[0]:
            sub_dictionary[0][name] = [{}, False]
        sub_dictionary = sub_dictionary[0][name]
    sub_dictionary[1] = True

def extract_dictionary(gene_list=None):
    if gene_list is None:
        return
    gene_dictionary = [{}, False] # Keys, Valid, Standard Name, Ambiguous
    for gene in gene_list:
        add_gene(gene_dictionary, gene)
    return gene_dictionary


if __name__ == "__main__":
    print(extract_dictionary())
