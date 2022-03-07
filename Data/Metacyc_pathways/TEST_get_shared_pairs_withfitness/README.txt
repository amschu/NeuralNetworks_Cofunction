##################################
####### FILE INFORMATION #########
##################################

### PATHWAY-GENE ###
# col1: pathways, col2:genes // genes // ...
# file1 hpcc = "/mnt/gs18/scratch/users/schum193/" +\
#              "pathways_matrix_generationfiles/" +\
#              "Pathways-of-gene-All--in-Saccharomyces-cerevisiae
#              -S288c_metacyc.txt"

'''
Pathways	Genes of pathway
A: genes 1-3
B: duplicated genes 1-3 and new genes 4-5
C: 1 gene duplicated (gene1) rest not found in pathway A or B
D: no duplicated genes anywhere (gene10 & gene11)
E: same genes as pathway F (gene12 // gene13 // gene14)
F: duplicated genes and same as pathway E
G: No genes in that pathway
H: all new genes with no other pathway repeated
I: single gene with duplicated and a gene seen in pathway A
J: single new gene with no other pathway duplicates
K: gene 21 not found in any other file All_instances_of_Genes_in_Saccharomyces_cerevisiae_S288c.txt
'''

### GENE-PATHWAY ###
# col1:all genes in yeast, col2: pathways // pathways // ...
# file2 hpcc = "/mnt/gs18/scratch/users/schum193/" +\
#              "pathways_matrix_generationfiles/" +\
#              "All-instances-of-Genes-in-Saccharomyces-cerevisiae
#               -S288c_metacyc.txt"

'''
Genes	Pathways of gene
gene20: no pathway data and not in the other file (All_pathways_of_S._cerevisiae_S288c)
gene1	A // B // C // I: 3 pathways
gene2	A // B: 2 pathways and duplciated
gene3	A // B: 2 pathways and duplciated
gene4	B
gene5	B
gene6	C
gene7	C
gene8	C
gene9	C
gene10	D
gene11	D
gene12	E // F
gene13	E // F
gene14	E // F
gene15	H
gene16	H
gene17	H
gene18	H
gene20
gene22  L: No pathway L in other file All_pathways_of_S._cerevisiae_S288c)
'''

### FITNESS ###
# columns: Query_allele_name,Array_allele_name,Query_single_mutant_fitness_(SMF)
#          ,Array_SMF,Double_mutant_fitness
# Row1 example: tfc3,mcm2,0.8285,0.9254,0.7319
# file3 = "/mnt/gs18/scratch/users/schum193/" +\
#          "pathways_matrix_generationfiles/" +\
#          "genepairs_SMF_DMF.csv"

'''
Query_allele_name,Array_allele_name,Query_single_mutant_fitness_(SMF),Array_SMF,Double_mutant_fitness
Line 2: new pair and not duplicated
3: new pair and duplicated
4: duplicated gene pair  with different fitness values
5: new pair but duplicated same fitess values as well
6: new pair but duplicated same fitess values as well (same as line5)
7: new pair and NOT repeated
8: new pair and should be clean negative set
9: new pair and should be clean postive set
10: 1st gene in pair only exist in All_instances and NOT All_pathways
11: 2nd gene in pair only exist in All_instances and NOT All_pathways
12: gene 19 (frist gene in genepair) does not exist in All_instances
13: used to catch reverse of gene pair
14: used to catch reverse of gene pair
15: used to make a gene pair that was created for merging data sets after code review for postive set
16: used to make a gene pair that was created for merging data sets after code review for neg set
'''