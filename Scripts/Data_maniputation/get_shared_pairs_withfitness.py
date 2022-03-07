"""
DATE CREATED: 11/29/21
CREATED BY: Ally Schumacher
REVIEWED BY: Shinhan Shiu
PURPOSE: Combine pathway files and gene list, create labels for all gene
pairs that have provided, and get SMF/DMF data

USAGE:
python review_get_shared_pairs/get_shared_pairs_in_fitness_REVIEW.py
-p ./Pathways-of-gene-All--in-Saccharomyces-cerevisiae-S288c_metacyc.txt
-g ./All-instances-of-Genes-in-Saccharomyces-cerevisiae-S288c_metacyc.txt
-f ./raw_filtered_genepairs_SMF_DMF.csv

OUTPUT FILE LOCATION (HPCC): "/mnt/gs18/scratch/users/schum193/" +\
                             "pathways_matrix_generationfiles/" +\
                             "all_pathways_with_fitness_metacyc.csv"
"""

import argparse
import itertools
from pathlib import Path

import pandas as pd
from tqdm import tqdm

#################################################################
########## COMMENT OUT THE FOLLOWING AND USE ARGEPARSE ##########
#################################################################
### HPPC USAGE
### PATHWAY-GENE
# col1: pathways, col2:genes // genes // ...
# file1 hpcc = "/mnt/gs18/scratch/users/schum193/" +\
#              "pathways_matrix_generationfiles/" +\
#              "Pathways-of-gene-All--in-Saccharomyces-cerevisiae
#              -S288c_metacyc.txt"
#
#
### GENE-PATHWAY
# col1:all genes in yeast, col2: pathways // pathways // ...
# file2 hpcc = "/mnt/gs18/scratch/users/schum193/" +\
#              "pathways_matrix_generationfiles/" +\
#              "All-instances-of-Genes-in-Saccharomyces-cerevisiae
#               -S288c_metacyc.txt"
#
# #FITNESS
# #file3 = "/mnt/gs18/scratch/users/schum193/" +\
#          "pathways_matrix_generationfiles/" +\
#          "genepairs_SMF_DMF.csv"


##############################################
######### READ IN FILES WITH ARGPARSE ########
##############################################

# define arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('-p', '--pathway_gene',
                       help='Path to the pathway-gene file', required=True)
argparser.add_argument('-g', '--gene_pathway',
                       help='Path to the gene-pathway file', required=True)
argparser.add_argument('-f', '--fitness',
                       help='Path to the fitness file', required=True)
# SH: get arguments
args = argparser.parse_args()

# get file paths
file1 = Path(args.pathway_gene)
file2 = Path(args.gene_pathway)
file3 = Path(args.fitness)

#### PATHWAY-GENE FILE ###
df_pathgene = pd.read_csv(file1, sep='\t')
print("Path-gene shape:", df_pathgene.shape)

### GENE-PATHWAY FILE ###
df_genepathway = pd.read_csv(file2, sep='\t')
df_genepathway = df_genepathway.dropna(axis=0)
print("Gene-path shape:", '\n', df_genepathway.shape)

### FITNESS FILE ###
# Read fitness file, create two new columns
#  GenePairs: Query_Array
#  GenePairs_Reversed: Array_Query
df_fitness = pd.read_csv(file3)
print("Fitness shape df_fitness:", '\n', df_fitness.shape)

### CREATE PAIRS IN RESERVE AND COMBINED PAIRS ###
df_fitness['GenePairs'] = df_fitness['Query_allele_name'].str.cat(
    df_fitness['Array_allele_name'], sep="_")
df_fitness['GenePairs'] = df_fitness['GenePairs'].str.upper()
df_fitness['GenePairs_Reversed'] = df_fitness['Array_allele_name'].str.cat(
    df_fitness['Query_allele_name'], sep="_")
df_fitness['GenePairs_Reversed'] = df_fitness['GenePairs_Reversed'].str.upper()
print("     add cols:", '\n', df_fitness.shape)
print("df_fitness snapshot with added cols:", '\n', df_fitness.head())

### CREATE LIST OF ALL GENES IN YEAST
list_all_genes = df_genepathway['All-Genes'].tolist()
# print(list_all_genes)
print("No. of genes:", '\n', len(list_all_genes))

#######################################################
######### CREATING ALL GENE PAIR COMBINATIONS ########
#######################################################

### CREATING ALL POSSIBLE GENE PAIRS IN THE YEAST ###
# this is used to create a reference all possible pairs without
# acknowledging if the pairs with be co-functional or not
pairs = list(itertools.combinations(list_all_genes, 2))
# print(pairs)
print("No. of all possible gene combos:", '\n', len(pairs))

# Store positive and negative gene pair info
# [g1_g2, bla, blah...]
positive_set = []
negative_set = []

#### ITERATE THROUGH ALL GENE COMBOS ###
for gene1, gene2 in tqdm(pairs):
    # Getting the index of gene1 and 2 in the gene-pathway dataframe
    gene1_index = df_genepathway.index[df_genepathway['All-Genes'] == gene1]
    gene2_index = df_genepathway.index[df_genepathway['All-Genes'] == gene2]
    # Getting the pathway info for gene 1 and 2
    gene1_pathways_str = \
        df_genepathway.loc[gene1_index, "Pathways of gene"].values[0]
    gene2_pathways_str = \
        df_genepathway.loc[gene2_index, "Pathways of gene"].values[0]
    # print(gene2_pathways_str)

    # Get a list ["path1", " path2 ", " path3 "]
    gene1_pathways_list = gene1_pathways_str.split("//")
    # Fix the space issue around path names
    gene1_pathways_list = [pathway.strip().lower() for pathway in
                           gene1_pathways_list]
    # print(gene1_pathways_list)
    gene2_pathways_list = gene2_pathways_str.split("//")
    gene2_pathways_list = [pathway.strip().lower() for pathway in
                           gene2_pathways_list]

    gene1_pathways_set = set(gene1_pathways_list)
    gene2_pathways_set = set(gene2_pathways_list)
    # print(gene1_pathways_set,gene2_pathways_set)

    intersection = gene2_pathways_set.intersection(gene1_pathways_list)

    if len(intersection) > 0:
        # print(intersection_as_list)
        positive_set.append((gene1, gene2))
    else:
        negative_set.append((gene1, gene2))

### LOOKING AT NUMBER OF POS AND NEG EXAMPLES FOR GENE PAIRS ###
print("+:", '\n', len(positive_set))
# print(positive_set)
print("-:", '\n', len(negative_set))
# print("Negative set gene pairs", negative_set)


### DROPPING DUPLICATES AND ADDING LABELS TO PAIRS ###
df_pos = pd.DataFrame(positive_set,
                      columns=['Gene1', 'Gene2']).drop_duplicates()
df_pos['Label'] = 1
print("df_pos snapshot:", '\n', df_pos.head())
df_neg = pd.DataFrame(negative_set,
                      columns=['Gene1', 'Gene2']).drop_duplicates()
df_neg['Label'] = 0
print("df_neg snapshot:", '\n', df_neg.head())

print("df_pos.shape:", '\n', df_pos.shape)
print("df_neg.shape:", '\n', df_neg.shape)

##################################################
######### MERGING POS AND NEG DATAFRAMES #########
##################################################

# MERGE DATA FRAMES
data_frames = [df_neg, df_pos]
df_neg_pos = pd.concat(data_frames, axis=0).reset_index(drop=True)
print("df_neg_pos.shape:", '\n', df_neg_pos.shape)

df_neg_pos['GenePairs'] = df_neg_pos['Gene1'].str.cat(df_neg_pos['Gene2'],
                                                      sep="_")
df_neg_pos['GenePairs_Reversed'] = df_neg_pos['Gene2'].str.cat(
    df_neg_pos['Gene1'], sep="_")

# LOOK AT SHAPE OF COMBINED FRAMES
print("Col added df_neg_pos.shape:", df_neg_pos.shape)
print("Col added df_neg_pos for combined genepair and reverse format:", '\n',
      df_neg_pos.head())

################################################################################
######## MERGE POS_NEG DATA FRAME TO FITNESS TO FIND PAIRS W/ FITNESS ##########
################################################################################

merged_neg_pos_fitness = df_fitness.merge(df_neg_pos, how='inner',
                                          on=['GenePairs',
                                              'GenePairs_Reversed'])
print("merged_neg_pos_fitness.shape:", '\n', merged_neg_pos_fitness.shape)
print("merged_neg_pos_fitness", '\n', merged_neg_pos_fitness.head())

# CREATE ORDER OF COLUMNS FOR INPUT INTO PIPELINE
# pairs, label, features, ...
merged_neg_pos_fitness = merged_neg_pos_fitness.filter(
    ['GenePairs', 'Label', 'Query_Array_genepair',
     'Query_single_mutant_fitness_(SMF)', 'Array_SMF', 'Double_mutant_fitness'],
    axis=1)

# checking shape to see how many pairs were in the fitness data
print("Filtered merged_neg_pos_fitness.shape:", '\n',
      merged_neg_pos_fitness.shape)
print(merged_neg_pos_fitness.head())

### dropping duplicate gene pairs and creating average of all fitness data for
### said duplicate
all_path_metacyc_no_dups = merged_neg_pos_fitness.groupby('GenePairs').mean()
print("after averageing duplicate gene pairs with different fitness "
      "replicates", '\n', all_path_metacyc_no_dups.shape)
print(all_path_metacyc_no_dups.head())

######################################
######### EXPORTING TO CSV ###########
######################################

# print(merged_neg_pos_fitness.head(10))
all_path_metacyc_no_dups.to_csv(
    r"./all_pathways_with_fitness_metacyc.csv",
    index=True)
