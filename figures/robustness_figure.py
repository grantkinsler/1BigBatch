import numpy as np
import pandas as p
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mtick

import sys
import os
sns.set_style('white')
sns.set_style('ticks')
sns.set_color_codes()

tools_path = '../code/tools.py'
graphs_path = '../code/graphs.py'
sys.path.append(os.path.dirname(os.path.expanduser(tools_path)))
sys.path.append(os.path.dirname(os.path.expanduser(graphs_path)))
import tools
import graphs
from tools import mutant_colorset
from tools import condition_colorset
from tools import renamed_conditions


fitness_data = p.read_csv('../data/fitness_weighted_allconditions_swapsremoved.csv')

sorted_m3_cols =   ['M3_Batch_23_fitness', '19_fitness', 'M3_Batch_18_fitness',
       'M3_Batch_20_fitness', 'M3_Batch_3_fitness', 'M3_Batch_13_fitness',
       'M3_Batch_6_fitness', '1BB_M3_fitness', 'M3_Batch_21_fitness']

sorted_nonm3_cols = ['1BB_1.4%Gluc_fitness', 'Ferm_44hr_Transfer_fitness',
       '1BB_1%Gly_fitness', '1BB_1.8%Gluc_fitness', '1BB_0.5%Raf_fitness',
       'Geldanamycin8.5uM_fitness', 'Ferm_40hr_Transfer_fitness',
       '1BB_Baffle_fitness', '1.5%_fitness', 'DMSO_fitness',
       '1BB_1%Raf_fitness', '1.7%_fitness', '1.6%_fitness',
       'Ferm_50hr_Transfer_fitness', '1.4%_fitness', '1BB_2ugFlu_fitness',
       'Ferm_54hr_Transfer_fitness', 'Resp_3Day_Transfer_fitness',
       '1BB_17uMGdA_fitness', '1.8%_fitness',
       'Resp_24hr_Transfer_fitness', '1BB_1%EtOH_fitness',
       '1BB_8.5uMGdA_fitness', '1BB_SucRaf_fitness', '2.5%_fitness',
       'Resp_4Day_Transfer_fitness', 'Resp_5Day_Transfer_fitness',
       '1BB_0.2MNaCl_fitness', '1BB_0.2MKCl_fitness',
       '1BB_0.5ugFlu_fitness', 'Ben0.4_fitness', 'Ben2_fitness',
       'Resp_6Day_Transfer_fitness', 'Resp_7Day_Transfer_fitness',
       '1BB_0.5MKCl_fitness', '1BB_0.5MNaCl_fitness']

first_nonsubtle = 16

np.random.seed(94305) # for exact figure reproducibility and sets used in main text, use this seed



this_data = fitness_data
this_data = this_data.replace([np.inf, -np.inf], np.nan)
this_data = this_data.dropna('columns',how='all')
this_data = this_data.dropna()
this_data = this_data.sort_values('barcode')

datasets = {}


gene_list = ['IRA1_nonsense','GPB2','PDE2','Diploid','ExpNeutral']

# n_trials = 10
n_trials = 1000

# param_list = [(4,10),(3,10),(5,10),(4,5),(3,5),(5,5),(4,15),(3,15),(5,15)]
param_list = [(3,10),(4,10),(5,10),(10,10),(3,5),(4,5),(5,5),(10,5),(3,15),(4,15),(5,15),(10,15)]
param_list = [(4,10),(10,15),(50,100)]
param_list = [(4,10),(4,200),(50,100),(250,250)]
param_list = [(20,200)]



for m,(max_train,max_test) in enumerate(param_list):
    print(max_train,max_test,((m+1)/(len(param_list))))
    
    plt.figure(figsize=(4*np.ceil(n_trials/2),4*2))
    plt.title(f'{(max_train,max_test)}')
    
    datasets[f'{(max_train,max_test)}'] = {}
    
    for i in range(n_trials):
#         ax = plt.subplot(2,np.ceil(n_trials/2),i+1)
#         max_train = 4
#         max_test = 10

        training_bcs, testing_bcs = tools.select_train_test_mutants(this_data,max_train=max_train,max_test=max_test)

        datasets[f'{(max_train,max_test)}'][i] = tools.situate_data(this_data,list(sorted_m3_cols) + list(sorted_nonm3_cols[:first_nonsubtle]),list(sorted_nonm3_cols[first_nonsubtle:]),
                      training_bcs,testing_bcs,gene_list,fixed_mutant_sets=True,n_cross_validations=100)


all_improvements = []

for m,(max_train,max_test) in enumerate(param_list):
    for i in range(n_trials):
        dataset = datasets[f'{(max_train,max_test)}'][i]
        guesses = dataset['guesses']
        both_new = dataset['both_new']
        
        train_conditions = dataset['train_conditions']
        test_conditions = dataset['test_conditions']
        training_bcs = dataset['training_bcs']
        testing_bcs = dataset['testing_bcs']
        
        left_out_fits = tools.leave_one_out_analysis(this_data,train_conditions,test_conditions,training_bcs,testing_bcs,weighted=True)
        left_out_fits = left_out_fits[0]

        types = this_data[this_data['barcode'].isin(dataset['testing_bcs'])]['mutation_type'].values
        
        subtle5 = []
        subtle8 = []
        
        for left_out_index in range(len(train_conditions)):
            subtle5.append(left_out_fits[train_conditions[left_out_index]][4][0])
            subtle8.append(left_out_fits[train_conditions[left_out_index]][7][0])

#         types = this_data[this_data['barcode'].isin(dataset['testing_bcs'])]['mutation_type'].values
        strong5 = np.asarray([tools.var_explained_weighted_by_type(both_new[:,i],guesses[4][:,i],types)[0] for i in range(both_new.shape[1])])
        strong8 = np.asarray([tools.var_explained_weighted_by_type(both_new[:,i],guesses[7][:,i],types)[0] for i in range(both_new.shape[1])])
        
        all_improvements.append(np.asarray(list((np.asarray(subtle8)-np.asarray(subtle5))/np.asarray(subtle8))+list((np.asarray(strong8)-np.asarray(strong5))/np.asarray(strong8))))
        
        
all_improvements = np.asarray(all_improvements)

plt.errorbar(range(len(train_conditions)+len(test_conditions)),np.mean(all_improvements,axis=0),yerr=2*np.std(all_improvements,axis=0),
marker='o',color='k',linestyle='None',alpha=0.8)

plt.axhline(0,color='k',linestyle=':')
plt.ylim(-0.1,1.5)
plt.xticks(range(len(train_conditions)+len(test_conditions)),[renamed_conditions[col.replace('_fitness','')] for col in list(train_conditions)+list(test_conditions)],rotation=90)

plt.xlim(-0.5,len(train_conditions)+len(test_conditions)-0.5)
ymin,ymax = plt.ylim()

plt.axvline(len(train_conditions)-0.5,color='k',lw=0.75)

for i in range(int(np.ceil((len(train_conditions)+len(test_conditions))/3))):
    if (i % 2) == 0:
        # print(i)
        rect = matplotlib.patches.Rectangle((8+1+3*i-0.5,ymin),3,ymax-ymin,
                                        linewidth=0,edgecolor='lightgray',facecolor='lightgray',alpha=0.2)
        plt.gca().add_patch(rect)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.xlabel('Environment')
plt.ylabel('Percent improvement due to three minor components')

arrow_left = plt.gca().transData.transform_point((1, 0))
arrow_right = plt.gca().transData.transform_point((9, 0))

arrow_width = (arrow_right[0]-arrow_left[0])/2
print(arrow_left,arrow_right,arrow_width)

trans_arrow = matplotlib.transforms.blended_transform_factory(plt.gca().transData, plt.gca().transData)

plt.annotate('Batches\nof the\nEvolution\nCondition', xy=(9/2-0.5, -0.35), xytext=(9/2-0.5, -0.5), 
    fontsize=10,ha='center', va='top',xycoords=trans_arrow,annotation_clip=False,
    arrowprops=dict(arrowstyle=f'-[, widthB={arrow_width}, lengthB=7.0', lw=1.0,mutation_scale=1.0))

plt.savefig('figureS4.pdf',bbox_inches='tight')

