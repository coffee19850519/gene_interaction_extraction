import json
def extract_genes():
    with open('genefromocr.tsv', 'r') as genefile:
        genes = set()
        for line in genefile.readlines():
            values = line.split('\t')
            if len(values) > 2 and ('.png' not in values[1] and '.png' not in values[2]):
                genes.add(values[1])
                genes.add(values[2])
        return genes


genes = list(extract_genes())

with open('gene_list.json', 'w') as genefile:
    json.dump(genes, genefile)
