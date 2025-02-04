from collections import Counter
from . import sentence_interpreter
from fuzzywuzzy.fuzz import ratio

threshold = 70
def get_unordered_pair(gene_a, gene_b):
    if gene_a < gene_b:
        return (gene_a, gene_b)
    return (gene_b, gene_a)

def get_co_occurrences(gene_dictionary_reader, text):
    paper = sentence_interpreter.split_sentences(text)
    gene_co_occurrence = Counter()
    for sentence in paper:
        gene_pairs = sentence_interpreter.get_gene_pairs(gene_dictionary_reader, sentence)
        gene_co_occurrence.update([get_unordered_pair(pair[0], pair[1]) for pair in gene_pairs])
    return gene_co_occurrence

def get_ordered_counts(gene_dictionary_reader, text):
    paper = sentence_interpreter.split_sentences(text)
    gene_counts = Counter()
    for sentence in paper:
        gene_pairs = sentence_interpreter.get_gene_pairs(gene_dictionary_reader, sentence)
        gene_co_occurrence.update(gene_pairs)
    return gene_co_occurrence

def get_pair_counts(counts, pairs):
    results = Counter()
    for gene in pairs:
        actor = gene[0]
        receiver = gene[1]
        best_match = (0, 0)
        match = None
        for count in counts:
            geneA = count[0]
            geneB = count[1]
            actorRatio = max(ratio(geneA, actor), ratio(geneB, actor))
            receiverRatio = max(ratio(geneA, receiver), ratio(geneB, receiver))
            if best_match[0] < actorRatio and best_match[1] < receiverRatio:
                if actorRatio > threshold and receiverRatio > threshold:
                    best_match = actorRatio, receiverRatio
                    match = count
        if match:
            results[gene] = counts[match]
        else:
            results[gene] = 0
    return results

def counted_score(count, bound):
    if count > bound:
        return 2
    if count <= bound and count > 0:
        return 1
    return 0