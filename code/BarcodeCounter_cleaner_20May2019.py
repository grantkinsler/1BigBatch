import numpy as np
import pandas as p
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import copy
import datetime

data = p.read_table('../data/allBarcodeCounts_noF_doubleBC_OneAncestor.tab')

bc_fasta = SeqIO.to_dict(SeqIO.parse(open('../data/Consecutive_500pool_NoConstant_OneAncestor_ReverseComplement_BothBCs.fasta'),'fasta'))


### SAMPLES THAT WERE SWAPPED WHEN FREEZING DOWN CELLS FOR GENOMIC DNA EXTRACTION
flask_swaps = {'Y1':'CC1','Z1':'DD1','CC1':'Y1','DD1':'Z1'}

# bad_samples = ['B3-DE1-PCRb']


data['barcode'] = list(bc_fasta.keys()) + ['9999999']
data['barcode'] = [int(bc) for bc in data['barcode'].values]
data = data.sort_values('barcode')

new_cols = []

for col in data.columns:
    if '/' in col:
        new_cols.append(col.split('/')[1].split('_barcodeCounts')[0])
    else:
        new_cols.append(col)
data.columns = new_cols
data = data.drop(['BCID'],axis=1)



### CORRECT FLASK SWAPS


corrected_data = {}
for col in data.columns:
    if '-DE' in col:
        flask,de,pcr = col.split('-')
    
        if flask in flask_swaps.keys():
            corrected_data[flask_swaps[flask]+f'-{de}-{pcr}'] = data[col]
        else:
            corrected_data[col] = data[col]
    else:
        corrected_data[col] = data[col]

# corrected_data
data = p.DataFrame.from_dict(corrected_data)



### MUTATION INFORMATION 
mutation_data = p.read_table('../data/mutationsByBarcodeHighQuality.txt')

mutation_cols = [col for col in mutation_data.columns if ('fitness' not in col and 'error' not in col) ]

tor_genes = ['KOG1','TOR1','SCH9']
ras_genes = ['IRA1','IRA2','GPB1','GPB2','PDE2','RAS2','CYR1','TFS1']

genelist = ['Diploid','Diploid + Chr11Amp','Diploid + Chr12Amp'] + ras_genes + tor_genes

mutation_data = mutation_data[mutation_cols]
mutation_data = mutation_data.sort_values('barcode')

### called neutral by atish's method in ALL 5000 bc experiments (I think - need to verify this)
### [could also be below some set threshold across all experiments]
full_neutral_list = [17615,18486,42040,45014,58284,63611,73731,74185,80465,94896
,120600,125697,132511,134852,135750,190551,228237,238783,255561,298344
,308537,316954,317346,335717,411685,454359,469053] 

### from previous list but never has fitness above 3.5% (per gen) in any of 5000bc experiments
supergood_neutral = [17615, 24362, 42040, 71926, 72939, 73802, 80465, 109476, 113483, 
                     134852, 135750, 238783, 263665, 276406, 316954, 335717, 454359] 

### pulled from supergood list and spiked into 1BigBatch experiments
neutral_spikes = [17615,24362,42040,71926,73802,109476,113483,134852,263665,316954]

# neutrals = full_neutral_list
neutrals = list(np.unique(full_neutral_list+supergood_neutral+neutral_spikes))



spike_in_missense = [9000000 + i for i in range(11)]
spike_in_nonsense = [9000100 + i for i in range(11)]

bc_list = []
gene_list = []
ploidy_list = []
class_list = []
type_list = []
additional_muts = []
for bc in data['barcode'].values:
    if bc in spike_in_missense:
        gene_list.append('IRA1')
        ploidy_list.append('Haploid')
        class_list.append('PKA')
        type_list.append('missense')
        additional_muts.append('None')
    elif bc in spike_in_nonsense:
        gene_list.append('IRA1')
        ploidy_list.append('Haploid')
        class_list.append('PKA')
        type_list.append('stop_gained')
        additional_muts.append('None')
    elif bc == 9999999:
        gene_list.append('Ancestor')
        ploidy_list.append('Haploid')
        class_list.append('Ancestor')
        type_list.append('Ancestor')
        additional_muts.append('None')
    elif bc in mutation_data['barcode'].values:
        
        this_mutant = mutation_data[mutation_data['barcode']==bc]
        found_gene = this_mutant[this_mutant['gene'].isin(genelist)]

        if len(found_gene.index) == 0:
            if bc in neutrals:
                gene_list.append('other')
                type_list.append('other')
                ploidy_list.append('other')
                class_list.append('ExpNeutral')
            else:
                gene_list.append('other')
                type_list.append('other')
                ploidy_list.append('other')
                class_list.append('other')
            

                

        elif len(found_gene.index) == 1:
            gene_list.append(found_gene['gene'].values[0])
            type_list.append(found_gene['type'].values[0])
            ploidy_list.append(found_gene['ploidy'].values[0])

            if found_gene['gene'].values[0] in tor_genes:
                class_list.append('PKA')
            elif found_gene['gene'].values[0] in ras_genes:
                class_list.append('PKA')
            elif found_gene['gene'].values[0] in ['Diploid + Chr11Amp','Diploid + Chr12Amp']:
                class_list.append('Adaptive Diploid')
            elif found_gene['gene'].values[0] in ['Diploid']:
                class_list.append('Diploid')

        else:
            if 'Diploid + Chr11Amp' in found_gene['gene'].values:
                gene_list.append('Diploid + Chr11Amp')
                type_list.append('Diploid + Chr11Amp')
                ploidy_list.append(found_gene['ploidy'].values[0])
                class_list.append('Adaptive Diploid')
                
            elif 'Diploid + Chr12Amp' in found_gene['gene'].values:
                gene_list.append('Diploid + Chr12Amp')
                type_list.append('Diploid + Chr12Amp')
                ploidy_list.append(found_gene['ploidy'].values[0])
                class_list.append('Adaptive Diploid')
            
            elif 'Diploid' in found_gene['gene'].values:
                other_index = np.where(found_gene['gene'].values != 'Diploid')[0][0]
                gene_list.append('Diploid + ' + found_gene['gene'].values[other_index])
                type_list.append(found_gene['type'].values[other_index])
                ploidy_list.append(found_gene['ploidy'].values[0])
                class_list.append('Adaptive Diploid')
            else:
                print('Panic! A double mutant was found!')


        additional = this_mutant[~(this_mutant['gene'].isin(genelist))]
        if len(this_mutant[~(this_mutant['gene'].isin(genelist))]['gene'].values) > 0:
            additional_muts.append('; '.join([str(g)+'-'+str(t) for g,t in zip(additional['gene'].values,additional['type'].values)]))
        else:
            additional_muts.append('None')
        
        
    else:
        
        if bc in neutrals:
            gene_list.append('NotSequenced')
            type_list.append('NotSequenced')
            ploidy_list.append('NotSequenced')
            class_list.append('ExpNeutral')
            additional_muts.append('NotSequenced')
        else:
            gene_list.append('NotSequenced')
            ploidy_list.append('NotSequenced')
            class_list.append('NotSequenced')
            type_list.append('NotSequenced')
            additional_muts.append('NotSequenced')
            

data['gene'] = gene_list
data['type'] = type_list
data['ploidy'] = ploidy_list
data['class'] = class_list

data['additional_muts'] = additional_muts

### Write raw count data with mutations now
data.to_csv(f"../data/BarcodeCounts_noF_DoubleBC_flaskswapcorrected_{datetime.date.today().strftime('%m%d%y')}_withBCinfo.csv",index=False)


cols_by_condition = np.unique([col.split('-DE')[0] for col in data.columns])

merged_data = {}

for merge_col in cols_by_condition:
    this_data = np.zeros(len(data.index))
    for col in data.columns:
        if 'DE' in col:
            if col.split('-DE')[0] == merge_col:
                this_data = this_data + data[col].values
        elif '-' in col:
            if col == merge_col:
                this_data = this_data + data[col].values
            
    merged_data[merge_col] = this_data
merged_data['barcode'] = data['barcode'].values
merged_data['gene'] = data['gene'].values
merged_data['ploidy'] = data['ploidy'].values
merged_data['class'] = data['class'].values
merged_data['type'] = data['type'].values
merged_data['additional_muts'] = data['additional_muts'].values



merged_data = p.DataFrame.from_dict(merged_data)

merged_data.to_csv(f"../data/BarcodeCounts_noF_DoubleBC_merged+flaskswapcorrected_{datetime.date.today().strftime('%m%d%y')}_withBCinfo.csv",index=False)


