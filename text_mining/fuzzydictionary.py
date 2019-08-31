def get_fuzzy_gene_list(interacting_pairs):
    genes = set()
    for interacting_pair in interacting_pairs:
        genes.add(interacting_pair[0])
        genes.add(interacting_pair[1])
    return genes
