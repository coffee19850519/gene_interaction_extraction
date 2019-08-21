def add_gene(gene_dictionary, gene_name, standard_name, ambiguities):
    split_gene_name = gene_name.split(' ')
    sub_dictionary = gene_dictionary
    for name in split_gene_name:
        if name not in sub_dictionary[0]:
            sub_dictionary[0][name] = [{}, False, '', False]
        sub_dictionary = sub_dictionary[0][name]
    if not sub_dictionary[3] and sub_dictionary[1]:
        sub_dictionary[1] = False
        sub_dictionary[3] = True
        sub_dictionary[2].append(standard_name)
        ambiguities += 1
    elif not sub_dictionary[1]:
        sub_dictionary[1] = True
        sub_dictionary[2] = [standard_name]
    return ambiguities

def remove_quotes(gene):
    if gene and gene[0] == '"' and gene[-1] == '"':
        return gene[1:-1]
    return gene
def separate_comma_separated_gene_names(gene_names):
    genes = gene_names.split(', ')
    current_gene_name = ''
    new_gene_list = []
    for gene in genes:
        if gene:
            if gene[0] == '"' and gene[-1] == '"':
                gene = remove_quotes(gene)
            elif gene[0] == '"':
                current_gene_name += gene[1:] + ' '
            elif gene[-1] == '"':
                current_gene_name += gene[:-1] + ' '
                new_gene_list.append(current_gene_name.strip(' '))
            else:
                new_gene_list.append(gene)
    return new_gene_list

def extract_dictionary(dictionary_location='../gene_dictionary/Sheet3-表1.tsv'):
    gene_dictionary_file = open(dictionary_location, 'r')
    i = 0
    gene_dictionary = [{}, False, '', False]
    lines = gene_dictionary_file.readlines()
    ambiguities = 0
    for line in lines[3:]:
        temp_gene_vocab = line.split('\t')

        gene_vocabulary = [temp_gene_vocab[index_of_gene] for index_of_gene in (0, 2, 3)]
        standard_name = gene_vocabulary[0]

        for gene in gene_vocabulary:
            gene = remove_quotes(gene)
            if gene.find(', ') != -1:
                pass
                genes = separate_comma_separated_gene_names(gene)
                for gene in genes:
                    if len(gene) > 1:
                        ambiguities = add_gene(gene_dictionary, gene, standard_name, ambiguities)
            else:
                if len(gene) > 1:
                    ambiguities = add_gene(gene_dictionary, gene, standard_name, ambiguities)
    gene_dictionary_file.close()
    #print('warning: found {} ambiguities in gene dictionary'.format(ambiguities))
    return gene_dictionary

if __name__ == "__main__":
    print(extract_dictionary())
