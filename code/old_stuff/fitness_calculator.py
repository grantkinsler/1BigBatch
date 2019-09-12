import numpy as np
import pandas as p
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors

from itertools import combinations
import scipy
from scipy.ndimage.filters import gaussian_filter1d
import sys
import os
import copy
atish_assay_path = 'fitness_assay_grantedits.py'
sys.path.append(os.path.dirname(os.path.expanduser(atish_assay_path)))
import fitness_assay_grantedits as atish

tools_path = '../code/tools.py'
sys.path.append(os.path.dirname(os.path.expanduser(tools_path)))
import tools

sns.set_style('white')
sns.set_style('ticks')
sns.set_color_codes()
from tools import mutant_colorset

merged_name = 'noF_DoubleBC_merged+flaskswapcorrected_062719' 

merged_data = p.read_csv(f'../data/BarcodeCounts_{merged_name}_withBCinfo.csv')

old_conditions = {
                  '3':['3.1','3.2','3.3'],
                  '6':['6.1','6.2','6.3'],
                  '13':['13.1','13.2','13.3'],
                  '18':['18.1','18.2','18.3'],
                  '20':['20.1','20.2','20.3'],
                  '21':['21.1','21.2','21.3'],
                  '23':['23.1','23.2','23.3']
                 }

bigbatch_conditions = {
                        '1BB_M3':['A','B','C','D'],
                        '1BB_Baffle':['E','F'],
                        '1BB_1.4%Gluc' :['G','H'],
                        '1BB_1.8%Gluc' :['I','J'],
                        '1BB_0.2MNaCl' :['K','L'],
                        '1BB_0.5MNaCl' :['M','N'],
                        '1BB_0.2MNaCl' :['K'], 
                        '1BB_0.2MKCl' :['O'],
                        '1BB_0.5MKCl' :['P'],
                        '1BB_8.5uMGdA' :['Q','R'],
                        '1BB_17uMGdA' :['S','T'],
                        '1BB_2ugFlu' :['U','V'],
                        '1BB_0.5ugFlu' :['W','X'],
                        '1BB_1%Raf' :['Y','Z'],
                        '1BB_0.5%Raf' :['AA','BB'],
                        '1BB_1%Gly' :['CC','DD'],
                        '1BB_1%EtOH' :['EE','FF'],
                        '1BB_SucRaf' :['GG'],
                      }



conditions = {**old_conditions, **bigbatch_conditions}

just_reps = [rep for reps in conditions.values() for rep in reps]

data = merged_data
data = data[~(data['barcode'].isin([7777777,9999999]))]
data = data.replace([np.inf, -np.inf], np.nan)
data = data.sort_values('barcode')

full_neutral_list = [17615,18486,42040,45014,58284,63611,73731,74185,80465,94896
,120600,125697,132511,134852,135750,190551,228237,238783,255561,298344
,308537,316954,317346,335717,411685,454359,469053] 

### from previous list but never has fitness above 3.5% (per gen) in any of 5000bc experiments
supergood_neutral = [17615, 24362, 42040, 71926, 72939, 73802, 80465, 109476, 113483, 
                     134852, 135750, 238783, 263665, 276406, 316954, 335717, 454359] 

### pulled from supergood list and spiked into 1BigBatch experiments
neutral_spikes = [17615,24362,42040,71926,73802,109476,113483,134852,263665,316954]

# neutrals = full_neutral_list
neutrals = sorted(list(np.unique(full_neutral_list+supergood_neutral+neutral_spikes)))

neutrals = [bc for bc in data['barcode'].values if bc in neutrals]

fitnesses = {}

for flask in ['A','B','C','D']:
    for time in range(4):
        for downcov in [1e5,1e6]:
            t0_data = data[f'{flask}{time}'].values
            t1_data = data[f'{flask}{time+1}'].values

            n_samples = 50
            down_samples = [tools.downsample_single(np.asarray([t0_data,t1_data]).swapaxes(0,1),downcov) for i in range(n_samples)]

            for sample in range(n_samples):
                timepoints = [time,time+1]

                answer = atish.inferFitness(data['barcode'].values,timepoints,{'rep1':down_samples[sample]},neutralBarcodes=neutrals,
                                                lowCoverageThresh=100,use_all_neutral=True,weightedMean=False)

                fitnesses[f'{flask}{time}_{int(downcov)}_{sample}'+'_fitness'] = answer['rep1']['aveFitness']

fitnesses['barcode'] = data['barcode'].values
resample_df = p.DataFrame(fitnesses)

resample_df.to_csv('../data/Fitness_Merged_1BB_M3_Resampled.csv',index=False)

# plt.savefig(f'Alltrajectories_old+1bb_swapscorrected_reshape.pdf',bbox_inches='tight')







