import tools
import numpy as np
import pandas as p
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import gridspec
from itertools import combinations
from scipy.spatial import distance
from tools import mutant_colorset
from tools import condition_colorset
from tools import renamed_conditions
from tools import tick_base_calculator
sns.set_color_codes()


def fitness_featured_genes_figure(ax,this_data,minimal_training_bcs,minimal_testing_bcs,m3_conditions,nonm3_conditions,gene_list,
    ymin=-1.25,ymax=1.5,yticknum=12,median=True,legend=True,legend_cols=1,color_alpha=0.1,gray_alpha=0.1,fontsize=12,style='default'):

    if style == 'dark':
        plt.style.use('dark_background')
        guide_color = 'lightgray'
        guide_alpha=0.12
        mutant_colorset['other'] = 'w'

    else:
        guide_color = 'gray'
        guide_alpha = 0.07


    mutant_data = this_data[this_data['barcode'].isin((list(minimal_training_bcs)+list(minimal_testing_bcs)))]

    this_gene_data = mutant_data[mutant_data['mutation_type'].isin(gene_list)]


    this_gene_locs = np.where(np.isin(mutant_data['barcode'].values,this_gene_data['barcode'].values))[0]
    jitters = [tools.jitter_point(0,0.01) for bc in range(len(this_gene_data[m3_conditions[0]].values)) ]


    ### eye guides
    for i in range(int(np.ceil(len(nonm3_conditions)/3))):
        if (i % 2) == 0:
            # print(i)
            rect = matplotlib.patches.Rectangle((len(m3_conditions)+1+3*i-0.5,ymin),3,ymax-ymin,
                                            linewidth=0,edgecolor=guide_color,facecolor=guide_color,alpha=guide_alpha)
        
            ax.add_patch(rect)


    # plt.title(f'Predictions with {rank+1} components')
    for i,col in enumerate(m3_conditions):
        for bc in range(len(this_gene_data[col].values)):
            if mutant_colorset[this_gene_data['mutation_type'].values[bc]] in ['gray','k']:
                ax.scatter([i + 1 + jitters[bc]],this_gene_data[col].values[bc],alpha=gray_alpha,color=mutant_colorset[this_gene_data['mutation_type'].values[bc]])
            else:
                ax.scatter([i + 1 + jitters[bc]],this_gene_data[col].values[bc],alpha=color_alpha,color=mutant_colorset[this_gene_data['mutation_type'].values[bc]])


    for i,col in enumerate(nonm3_conditions):
        for bc in range(len(this_gene_data[col].values)):
            if mutant_colorset[this_gene_data['mutation_type'].values[bc]] in ['gray','k']:
                ax.scatter([len(m3_conditions)+i + 1 + jitters[bc]],this_gene_data[col].values[bc],alpha=gray_alpha,color=mutant_colorset[this_gene_data['mutation_type'].values[bc]])
            else:
                ax.scatter([len(m3_conditions)+i + 1 + jitters[bc]],this_gene_data[col].values[bc],alpha=color_alpha,color=mutant_colorset[this_gene_data['mutation_type'].values[bc]])


    if median:
        for gene in gene_list:
            this_gene_data = mutant_data[mutant_data['mutation_type']==gene]
            data = np.median(np.asarray([this_gene_data[col].values for col in (list(m3_conditions) + list(nonm3_conditions))]),axis=1)
            ax.plot(range(1,len((list(m3_conditions) + list(nonm3_conditions)))+1),data,alpha=1.0,color=mutant_colorset[gene],label=gene)

    plt.axvline(x=len(m3_conditions)+0.5,color='gray')
    plt.xticks(range(1,len(m3_conditions)+len(nonm3_conditions)+1),[renamed_conditions[col.split('_fitness')[0]] for col in (list(m3_conditions) + list(nonm3_conditions))],rotation=90)
    # plt.xticks(range(1,len(m3_conditions)+len(nonm3_conditions)+1),['' for col in (list(m3_conditions) + list(nonm3_conditions))],rotation=90)

    if legend:
        legend_split = np.ceil(len(gene_list)/legend_cols)
        for g,gene in enumerate(gene_list):
            x_loc = 0.02+np.floor((g)/legend_split)*0.3
            y_loc = 0.05*(legend_split-1)-0.05*(g%legend_split)+0.02
            plt.text(s=f"{gene.replace('_',' ')}",x=x_loc,y=y_loc,fontsize=fontsize,
                  fontweight='semibold',color=mutant_colorset[gene],transform=ax.transAxes)



    plt.xlim(0.0,len(m3_conditions)+len(nonm3_conditions)+1.0)
    plt.ylabel('Relative Fitness')
    plt.xlabel('Condition')

    plt.ylim(ymin,ymax)
    plt.yticks(np.linspace(ymin,ymax,yticknum),np.linspace(ymin,ymax,yticknum))

    return ax

def fitness_tubes_graph_replicates(ax,this_data,mean_std_bygene,bc_list,m3_conditions,nonm3_conditions,m3_assoc,nonm3_assoc,gene_list,
    ymin=-1.25,ymax=1.5,yticknum=12,legend=True,legend_cols=1,fontsize=12,style='default',side_bars=True,faded_alpha=0.3):

    if style == 'dark':
        plt.style.use('dark_background')
        faded_alpha = 0.3
        emph_alpha = 0.9
        guide_color = 'lightgray'
        guide_alpha=0.12
        below_color = 'w'
    else:
        faded_alpha = faded_alpha
        emph_alpha = 0.8
        guide_color = 'gray'
        guide_alpha = 0.07

    mutant_data = this_data[this_data['barcode'].isin(bc_list)]

    offset = {'IRA1_nonsense':0,
              'IRA1_missense':0.1/3,
              'Diploid':0,
              'GPB2':0.2/3,
              'PDE2':0.3/3,}
              
    this_gene_data = mutant_data[mutant_data['mutation_type'].isin(gene_list)]

    this_gene_locs = np.where(np.isin(mutant_data['barcode'].values,this_gene_data['barcode'].values))[0]
    jitters = [tools.jitter_point(0,0.01) for bc in range(len(this_gene_data[m3_conditions[0]].values)) ]

    all_conditions = list(m3_conditions) + list(nonm3_conditions)
    print(all_conditions)

    plt.ylim(ymin,ymax)

    plt.axvline(x=len(m3_conditions)+0.5,color='gray')

    plt.axhline(y=0.0,color='k',linestyle=':',alpha=0.2)

    ### groupings by condition
    all_assoc = m3_assoc + nonm3_assoc

    unique_out = np.unique(all_assoc,return_index=True,return_counts=True)

    assoc_names = np.asarray(all_assoc)[np.sort(unique_out[1])]
    assoc_counts = unique_out[2][np.argsort(unique_out[1])]
    cum_counts = np.cumsum(assoc_counts)

    x_tick_locs = []
    for i in range(len(assoc_names)):
        if (i % 2) == 0:
            # print(i)
            rect = matplotlib.patches.Rectangle((1+(cum_counts[i]-assoc_counts[i])-0.5,ymin),assoc_counts[i],ymax-ymin,
                                            linewidth=0,edgecolor=guide_color,facecolor=guide_color,alpha=guide_alpha)
        
            ax.add_patch(rect)
        x_tick_locs.append((cum_counts[i]-0.5*assoc_counts[i]+0.5))

    # 2 sigma rectangles in background
    for gene in gene_list:
        
        mean = mean_std_bygene[gene][0]
        twosigma_top = mean_std_bygene[gene][0]+2*mean_std_bygene[gene][1]
        twosigma_bottom = mean_std_bygene[gene][0]-2*mean_std_bygene[gene][1]
        
        # print(gene,twosigma_bottom,twosigma_top,twosigma_top-twosigma_bottom)
        diff = twosigma_top - twosigma_bottom
        
        
        plt.axhline(twosigma_top,color=mutant_colorset[gene],linewidth=1.0,alpha=0.05)
        plt.axhline(twosigma_bottom,color=mutant_colorset[gene],linewidth=1.0,alpha=0.05)
        rect = matplotlib.patches.Rectangle((0,twosigma_bottom),len(all_conditions)+2,twosigma_top-twosigma_bottom,
                                            linewidth=1,edgecolor=mutant_colorset[gene],facecolor=mutant_colorset[gene],alpha=0.02)
        
        ax.add_patch(rect)
        
        diff_transform = ax.transData.transform((0.5, diff))

        if side_bars:
        
            trans = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)

            tops = ax.transData.transform((0,ymax))
            bottoms = ax.transData.transform((0,ymin))
            y_diff = tops[1]-bottoms[1]

            # ax.annotate("", xy=(1.025+offset[gene], mean), xytext=(1.04+offset[gene], mean),xycoords=trans,
            #         arrowprops=dict(arrowstyle=f'-[, widthB={diff}, lengthB=0.1,angleB=0',mutation_scale=4*10*1.25,lw=2.0,color=mutant_colorset[gene]))
            width = diff/(ymax-ymin)*y_diff/2



            ax.annotate("", xy=(1.025+offset[gene], mean), xytext=(1.04+offset[gene], mean),xycoords=trans,
                arrowprops=dict(arrowstyle=f'-[, widthB={width}, lengthB=4,angleB=0',mutation_scale=1, lw=2.0,color=mutant_colorset[gene]))




    low_counter = 0   
    for g,gene in enumerate(gene_list):
        mean = mean_std_bygene[gene][0]
        twosigma_top = mean_std_bygene[gene][0]+2*mean_std_bygene[gene][1]
        twosigma_bottom = mean_std_bygene[gene][0]-2*mean_std_bygene[gene][1]
        
        this_gene_data = mutant_data[mutant_data['mutation_type']==gene]

        # data = np.median(np.asarray([this_gene_data[col].values for col in all_conditions]),axis=1)
        for col in all_conditions:
            if col not in this_gene_data.columns:
                print(col)
        data = np.mean(np.asarray([this_gene_data[col].values for col in all_conditions]),axis=1)
        uncertainty = np.std(np.asarray([this_gene_data[col].values for col in all_conditions]),axis=1)
       
        colors = [matplotlib.colors.to_rgba(mutant_colorset[gene]) for i in range(len(data))]
        colors = [item[:3]+(faded_alpha,) if twosigma_bottom < data[i] < twosigma_top else item[:3]+(emph_alpha,) for i,item in enumerate(colors) ]
        
        
        plt.scatter(range(1,len(all_conditions)+1),data,marker='o',color=colors,label=gene)

        toolow = np.where(data<ymin)[0]
        for entry in toolow:
            plt.annotate("", xy=(entry+1, ymin+0.2*(low_counter)), xytext=(entry+1, ymin+0.2+0.2*(low_counter)),arrowprops=dict(arrowstyle="->",lw=1.5,color=mutant_colorset[gene]))
        if len(toolow) > 0:    
            low_counter += 1
    plt.xticks(x_tick_locs,[col.split('_fitness')[0] for col in assoc_names],rotation=90)
    # plt.xticks(x_tick_locs,[renamed_conditions[col.split('_fitness')[0]] for col in assoc_names],rotation=90)
    plt.ylim(ymin,ymax)
    plt.xlim(0.5,len(all_conditions)+0.5)
    plt.yticks(np.linspace(ymin,ymax,12),np.linspace(ymin,ymax,12))

    if legend:
        legend_split = np.ceil(len(gene_list)/legend_cols)
        for g,gene in enumerate(gene_list):
            x_loc = 0.01+np.floor((g)/legend_split)*0.3
            y_loc = 0.05*(legend_split-1)-0.05*(g%legend_split)+0.02
            plt.text(s=f"{gene.replace('_',' ')}",x=x_loc,y=y_loc,fontsize=fontsize,
                  fontweight='semibold',color=mutant_colorset[gene],transform=ax.transAxes)


    plt.ylabel('Relative Fitness')
    plt.xlabel('Condition')

    return ax


def fitness_tubes_graph(ax,this_data,mean_std_bygene,bc_list,m3_conditions,nonm3_conditions,gene_list,
    ymin=-1.25,ymax=1.5,yticknum=12,legend=True,legend_cols=1,fontsize=12,style='default',side_bars=True,faded_alpha=0.3):

    # test_mutant_data = this_data[~this_data['barcode'].isin(minimal_training_bcs)]
    # sns.set_style(style)
    if style == 'dark':
        plt.style.use('dark_background')
        faded_alpha = 0.3
        emph_alpha = 0.9
        guide_color = 'lightgray'
        guide_alpha=0.12
        below_color = 'w'
    else:
        faded_alpha = faded_alpha
        emph_alpha = 0.8
        guide_color = 'gray'
        guide_alpha = 0.07

    mutant_data = this_data[this_data['barcode'].isin(bc_list)]

    offset = {'IRA1_nonsense':0,
              'IRA1_missense':0.1/3,
              'Diploid':0,
              'GPB2':0.2/3,
              'PDE2':0.3/3,}
              
    this_gene_data = mutant_data[mutant_data['mutation_type'].isin(gene_list)]

    this_gene_locs = np.where(np.isin(mutant_data['barcode'].values,this_gene_data['barcode'].values))[0]
    jitters = [tools.jitter_point(0,0.01) for bc in range(len(this_gene_data[m3_conditions[0]].values)) ]

    all_conditions = list(m3_conditions) + list(nonm3_conditions)

    plt.ylim(ymin,ymax)

    plt.axvline(x=len(m3_conditions)+0.5,color='gray')

    plt.axhline(y=0.0,color='k',linestyle=':',alpha=0.2)

    ### eye guides
    for i in range(int(np.ceil(len(nonm3_conditions)/3))):
        if (i % 2) == 0:
            # print(i)
            rect = matplotlib.patches.Rectangle((len(m3_conditions)+1+3*i-0.5,ymin),3,ymax-ymin,
                                            linewidth=0,edgecolor=guide_color,facecolor=guide_color,alpha=guide_alpha)
        
            ax.add_patch(rect)

    ## 2 sigma rectangles in background
    for gene in gene_list:
        
        mean = mean_std_bygene[gene][0]
        twosigma_top = mean_std_bygene[gene][0]+2*mean_std_bygene[gene][1]
        twosigma_bottom = mean_std_bygene[gene][0]-2*mean_std_bygene[gene][1]
        
        # print(gene,twosigma_bottom,twosigma_top,twosigma_top-twosigma_bottom)
        diff = twosigma_top - twosigma_bottom
        
        
        plt.axhline(twosigma_top,color=mutant_colorset[gene],linewidth=1.0,alpha=0.05)
        plt.axhline(twosigma_bottom,color=mutant_colorset[gene],linewidth=1.0,alpha=0.05)
        rect = matplotlib.patches.Rectangle((0,twosigma_bottom),len(all_conditions)+2,twosigma_top-twosigma_bottom,
                                            linewidth=1,edgecolor=mutant_colorset[gene],facecolor=mutant_colorset[gene],alpha=0.02)
        
        ax.add_patch(rect)
        
        diff_transform = ax.transData.transform((0.5, diff))

        if side_bars:
        
            trans = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)

            tops = ax.transData.transform((0,ymax))
            bottoms = ax.transData.transform((0,ymin))
            y_diff = tops[1]-bottoms[1]

            # ax.annotate("", xy=(1.025+offset[gene], mean), xytext=(1.04+offset[gene], mean),xycoords=trans,
            #         arrowprops=dict(arrowstyle=f'-[, widthB={diff}, lengthB=0.1,angleB=0',mutation_scale=4*10*1.25,lw=2.0,color=mutant_colorset[gene]))
            width = diff/(ymax-ymin)*y_diff/2



            ax.annotate("", xy=(1.025+offset[gene], mean), xytext=(1.04+offset[gene], mean),xycoords=trans,
                arrowprops=dict(arrowstyle=f'-[, widthB={width}, lengthB=4,angleB=0',mutation_scale=1, lw=2.0,color=mutant_colorset[gene]))




    low_counter = 0   
    for g,gene in enumerate(gene_list):
        mean = mean_std_bygene[gene][0]
        twosigma_top = mean_std_bygene[gene][0]+2*mean_std_bygene[gene][1]
        twosigma_bottom = mean_std_bygene[gene][0]-2*mean_std_bygene[gene][1]
        
        this_gene_data = mutant_data[mutant_data['mutation_type']==gene]
        # data = np.median(np.asarray([this_gene_data[col].values for col in all_conditions]),axis=1)
        data = np.mean(np.asarray([this_gene_data[col].values for col in all_conditions]),axis=1)
        uncertainty = np.std(np.asarray([this_gene_data[col].values for col in all_conditions]),axis=1)

        # print(np.asarray([this_gene_data[col].values for col in all_conditions]))

        # data = []
        # uncertainty = []

        # for col in all_conditions:
        #     d,u = tools.inverse_variance_mean(this_gene_data[col].values,this_gene_data[col.replace('fitness','error')].values)
        #     data.append(d)
        #     uncertainty.append(u)

        # data = np.asarray(data)
        # uncertainty = 2*np.sqrt(np.asarray(uncertainty))
        
       
        colors = [matplotlib.colors.to_rgba(mutant_colorset[gene]) for i in range(len(data))]
        colors = [item[:3]+(faded_alpha,) if twosigma_bottom < data[i] < twosigma_top else item[:3]+(emph_alpha,) for i,item in enumerate(colors) ]
        
        
        plt.scatter(range(1,len(all_conditions)+1),data,marker='o',color=colors,label=gene)

        # plt.errorbar(range(1,len(all_conditions)+1),data,linestyle='',capsize=2,alpha=emph_alpha,yerr=uncertainty,color=mutant_colorset[gene])
        

        toolow = np.where(data<ymin)[0]
        for entry in toolow:
            plt.annotate("", xy=(entry+1, ymin+0.2*(low_counter)), xytext=(entry+1, ymin+0.2+0.2*(low_counter)),arrowprops=dict(arrowstyle="->",lw=1.5,color=mutant_colorset[gene]))
        if len(toolow) > 0:    
            low_counter += 1

    plt.xticks(range(1,len(all_conditions)+1),[renamed_conditions[col.split('_fitness')[0]] for col in all_conditions],rotation=90)
    # plt.xticks(range(1,len(all_conditions)+1),[col.split('_fitness')[0] for col in all_conditions],rotation=90)
    plt.ylim(ymin,ymax)
    plt.xlim(0.5,len(all_conditions)+0.5)
    plt.yticks(np.linspace(ymin,ymax,12),np.linspace(ymin,ymax,12))

    if legend:
        legend_split = np.ceil(len(gene_list)/legend_cols)
        for g,gene in enumerate(gene_list):
            x_loc = 0.01+np.floor((g)/legend_split)*0.3
            y_loc = 0.05*(legend_split-1)-0.05*(g%legend_split)+0.02
            plt.text(s=f"{gene.replace('_',' ')}",x=x_loc,y=y_loc,fontsize=fontsize,
                  fontweight='semibold',color=mutant_colorset[gene],transform=ax.transAxes)


    plt.ylabel('Relative Fitness')
    plt.xlabel('Condition')

    return ax
        

def zscore_graph(ax,m3_z_scores,nonm3_z_scores,sorted_m3_cols,sorted_nonm3_cols,sigma_thresh=2,ymin=0,ymax=18,ytick_interval=5,style='default'):

    if style == 'dark':
        plt.style.use('dark_background')
        guide_color = 'lightgray'
        guide_alpha=0.12
        below_color = 'w'

    else:
        guide_color='gray'
        guide_alpha=0.07
        below_color='k'




    plt.axhline(sigma_thresh,color=below_color,linestyle='--',alpha=0.2)

    ### eye guides
    for i in range(int(np.ceil(len(sorted_nonm3_cols)/3))):
        if (i % 2) == 0:
            # print(i)
            rect = matplotlib.patches.Rectangle((len(sorted_m3_cols)+3*i-0.5,ymin),3,ymax-ymin,
                                            linewidth=0,edgecolor=guide_color,facecolor=guide_color,alpha=guide_alpha)
        
            ax.add_patch(rect)


    all_z_scores = np.asarray(sorted(m3_z_scores)+sorted(nonm3_z_scores))
    below = all_z_scores[np.where(all_z_scores < sigma_thresh)[0]]
    above = all_z_scores[np.where(all_z_scores > sigma_thresh)[0]]

    plt.plot(below,'o',color=below_color,alpha=1.0,label='Balanced Mutant Set')
    plt.plot(range(len(below),len(below)+len(above)),above,'o',color='r',alpha=1.0,label='Balanced Mutant Set')


    all_cols = list(sorted_m3_cols)+list(sorted_nonm3_cols)
    plt.xticks(range(len(all_cols)),[renamed_conditions[col.split('_fitness')[0]] for col in (all_cols)],rotation=90)

    # plt.legend()
    # plt.ylim(0,7)
    plt.xlim(-0.5,len(sorted_m3_cols)+len(sorted_nonm3_cols)-0.5)
    # plt.ylabel('Average Z Score for\nBalanced Collection')
    plt.ylabel('Average Z Score')
    plt.xlabel('Condition')
    plt.axvline(x=len(sorted_m3_cols)-0.5,color='gray')
    # plt.yscale('log')

    # plt.tight_layout()
    plt.ylim(ymin,ymax)
    plt.yticks(range(ymin,ymax+1,ytick_interval),range(ymin,ymax+1,ytick_interval))

    return ax

def largescale_predictions_graph(ax,this_fitness,train,test,both_new,guesses,models,test_conditions,dataset,this_data,n_perms=100,ymin=-0.5,ymax=1.0,guide_color='lightgray',weighted=True):

    plt.axhline(0,color='gray',linestyle=':')

    perms = np.zeros(both_new.shape[1])

    # this_gene_data = this_data[this_data['barcode'].isin(test_mutant_data)]
    types = this_data[this_data['barcode'].isin(dataset['testing_bcs'])]['mutation_type'].values

    # n_perms = 1000
    for i in range(n_perms):
        perm_out = tools.SVD_predictions_train_test(this_fitness,train,test,by_condition=True,permuted_conditions=True)
        # perm = np.asarray([tools.var_explained(both_new[:,i],perm_out[5][model][:,i])[0] for i in range(both_new.shape[1])])
        if weighted:
            perm = np.asarray([tools.var_explained_weighted_by_type(both_new[:,i],perm_out[5][models[-1]][:,i],types)[0] for i in range(both_new.shape[1])])
        else:
            perm = np.asarray([tools.var_explained(both_new[:,i],perm_out[5][models[-1]][:,i])[0] for i in range(both_new.shape[1])])

        perms = perms + perm
        plt.plot(perm,'.',color='gray',alpha=0.01)
    plt.plot((perms/n_perms),'_',color='k',alpha=0.8,label='Permutation Average')

    for model in models:
        print(model+1,tools.var_explained_weighted_by_type(both_new,guesses[model],types)[0])

    # oneD = np.asarray([tools.var_explained(both_new[:,i],guesses[0][:,i])[0] for i in range(both_new.shape[1])])
    if weighted:
        oneD = np.asarray([tools.var_explained_weighted_by_type(both_new[:,i],guesses[0][:,i],types)[0] for i in range(both_new.shape[1])])
    else:
        oneD = np.asarray([tools.var_explained(both_new[:,i],guesses[0][:,i])[0] for i in range(both_new.shape[1])])

    plt.plot(oneD,'o',markeredgecolor='k',markerfacecolor='None',linestyle='',alpha=0.8,label='1 component model')

    for model in models:
    
        if weighted:
            this_sse = np.asarray([tools.var_explained_weighted_by_type(both_new[:,i],guesses[model][:,i],types)[0] for i in range(both_new.shape[1])])
        else:
            this_sse = np.asarray([tools.var_explained(both_new[:,i],guesses[model][:,i])[0] for i in range(both_new.shape[1])])

        if model == models[-1]:
            plt.plot(this_sse,'o',label=f'{model+1} component model',color='r',alpha=0.8)
        else:
            plt.plot(this_sse,'.',label=f'{model+1} component model',color='k',alpha=0.8)

        ### eye guides
    for i in range(int(np.ceil(len(test_conditions)/4))):
        if (i % 2) == 0:
            # print(i)
            rect = matplotlib.patches.Rectangle((4*i-0.5,ymin),4,ymax-ymin,
                                            linewidth=0,edgecolor='lightgray',facecolor='lightgray',alpha=0.2)
        
            ax.add_patch(rect)
        
    # for i in range(100):
    #     perm_out = tools.SVD_predictions_train_test(this_fitness,train,test,by_condition=True,permuted_mutants=True)
    #     perm = np.asarray([tools.sum_squared_error(perm_out[5][5][:,i],both_new[:,i]) for i in range(both_new.shape[1])])
    #     plt.plot((dumb - perm)/(dumb - min_sse),color='orange',linestyle='--')

    plt.ylabel(r'Weighted Coefficient of Determination ($\widetilde R^2$)')

    plt.xticks(range(len(test_conditions)),[renamed_conditions[col.replace('_fitness','')] for col in test_conditions],rotation=90)
    plt.legend(loc='lower left')
    plt.xlim(-0.5,len(test_conditions)-0.5)

    plt.ylim(ymin,ymax)

    return ax

def predictions_figure(ax,train,test,this_fitness,both_new,guesses,model,test_conditions):
    
    plt.axhline(0,color='gray',linestyle=':')

    perms = np.zeros(both_new.shape[1])

    n_perms = 1000
    for i in range(n_perms):
        perm_out = tools.SVD_predictions_train_test(this_fitness,train,test,by_condition=True,permuted_conditions=True)
        perm = np.asarray([tools.var_explained(both_new[:,i],perm_out[5][5][:,i])[0] for i in range(both_new.shape[1])])
        perms = perms + perm
        plt.plot(perm,color='gray',alpha=0.01)
    plt.plot((perms/n_perms),color='k',alpha=0.8,label='Permutation Average')

    dumb = np.asarray([tools.var_explained(both_new[:,i],guesses[0][:,i])[0] for i in range(both_new.shape[1])])
    plt.plot(dumb,color='k',linestyle='--',alpha=0.8,label='1 component model')


    colors=['r','b','g','m','orange']
    for m,model in enumerate(range(5,6)):
        this_sse = np.asarray([tools.var_explained(both_new[:,i],guesses[model][:,i])[0] for i in range(both_new.shape[1])])
        
        plt.plot(this_sse,'o-',label=f'{model+1} component model',color='r',alpha=1.0)
        
    # for i in range(100):
    #     perm_out = tools.SVD_predictions_train_test(this_fitness,train,test,by_condition=True,permuted_mutants=True)
    #     perm = np.asarray([tools.sum_squared_error(perm_out[5][5][:,i],both_new[:,i]) for i in range(both_new.shape[1])])
    #     plt.plot((dumb - perm)/(dumb - min_sse),color='orange',linestyle='--')

    plt.ylabel(r'Weighted Coefficient of Determination ($\tilde R^2$)')

    plt.xticks(range(len(test_conditions)),[renamed_conditions[col.replace('_fitness','')] for col in test_conditions],rotation=90)
    plt.legend()

    plt.ylim(-1.0,1.0)

    return ax


def distance_comparison_figure(fig,gs,distances_x,distances_y,geom_medians_x,geom_medians_y,avg_pairwise_x,avg_pairwise_y,gene_list,ylim='default',include_ancestor=True):

    # fig = plt.figure(figsize=(4+1,4+1))
    # inner_gs = GridSpecFromSubplotSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],hspace=0.0,wspace=0.0)

    # ax1 = fig.add_sub
    ## x data histogram
    ax1 = fig.add_subplot(gs[0])
    sns.despine(ax=ax1)
    sns.distplot(distances_x,ax=ax1,kde=False,color='gray')
    # ax4 = outer_gs[1,1]
    plt.text(s='A',x=-0.1,y=1.02,fontsize=15,fontweight='semibold',transform=ax1.transAxes)


    ax1.yaxis.set_visible(False)
    ax1.xaxis.set_visible(False)
    ax1.spines['left'].set_visible(False)
    # plt.xlim(0,1.5)
    plt.xticks()

    ## y data histogram
    ax3 = fig.add_subplot(gs[3])
    sns.despine(ax=ax3)
    sns.distplot(distances_y,ax=ax3,vertical=True,kde=False,color='gray')
    ax3.xaxis.set_visible(False)
    ax3.yaxis.set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    # plt.ylim(0,1.5)
    # plt.yticks()

    ## scatter 
    ax2 = fig.add_subplot(gs[2])

    ax2.scatter(distances_x,distances_y,alpha=0.03,color='gray')

    ax2.axvline(np.mean(distances_x),color='k',alpha=0.5)
    ax2.axhline(np.mean(distances_y),color='k',alpha=0.5)

    xs = np.linspace(0,1.2)
    # plt.plot(xs,np.sqrt(4)/np.sqrt(6)*xs,'k--')

    if 'ExpNeutral' in gene_list:
        # gene_list = [gene for gene in gene_list if gene != 'ExpNeutral']
        ax2.plot(distance.euclidean(geom_medians_x['ExpNeutral'],np.zeros(geom_medians_x['ExpNeutral'].shape)),
                        distance.euclidean(geom_medians_y['ExpNeutral'],np.zeros(geom_medians_y['ExpNeutral'].shape)),
                        label=f'ExpNeutral and Ancestor',color=tools.mutant_colorset['ExpNeutral'],marker='D',fillstyle = 'left',
                    markerfacecoloralt='k',linestyle='',markersize=7,markeredgecolor='k',alpha=0.9)
        # plt.plot(avg_pairwise_x['ExpNeutral'],avg_pairwise_y['ExpNeutral'],marker='o',linestyle='',color=tools.mutant_colorset[gene],alpha=0.8,label=f'Average pairwise for {gene}')


    for gene1, gene2 in combinations([gene for gene in gene_list if gene != 'ExpNeutral'],2):
        ax2.plot(distance.euclidean(geom_medians_x[gene1],geom_medians_x[gene2]),
                    distance.euclidean(geom_medians_y[gene1],geom_medians_y[gene2]),
                    label=f'{gene1} and {gene2}',color=tools.mutant_colorset[gene1],marker='D',fillstyle = 'left',
                markerfacecoloralt=tools.mutant_colorset[gene2],linestyle='',markersize=7,markeredgecolor='k',alpha=0.9)
    if include_ancestor:
        for gene1 in gene_list:
            ax2.plot(distance.euclidean(geom_medians_x[gene1],np.zeros(geom_medians_x[gene1].shape)),
                        distance.euclidean(geom_medians_y[gene1],np.zeros(geom_medians_y[gene1].shape)),
                        label=f'{gene1} and Ancestor',color=tools.mutant_colorset[gene1],marker='D',fillstyle = 'left',
                    markerfacecoloralt='k',linestyle='',markersize=7,markeredgecolor='k',alpha=0.9)

    for gene in gene_list:
        ax2.plot(avg_pairwise_x[gene],avg_pairwise_y[gene],marker='o',linestyle='',color=tools.mutant_colorset[gene],alpha=0.8,label=f'Average pairwise for {gene}')

    plt.xlabel(f'Pairwise Distance in {geom_medians_x[gene1].shape[0]} component model')
    plt.ylabel(f'Pairwise Distance in {geom_medians_y[gene1].shape[0]} component model')
    # plt.ylim(0,1.5)
    # plt.xlim(0,1.5)
    # plt.legend(loc=(1.1,0.65),ncol=1)
    ax2.legend(loc=(1.0,0.5),ncol=1)

    return gs



def Figure4(dataset,gene_list):
    all_guesses = dataset['CV_all_guesses']
    both_old = dataset['both_old']
    dhats = dataset['dhats']
    this_fitness  = dataset['this_fitness']
    train  = dataset['train']
    test = dataset['test']
    both_new = dataset['both_new']
    guesses = dataset['guesses']
    model = dataset['CV_best_rank_index']
    test_conditions  = dataset['test_conditions']

    fig = plt.figure(figsize=(10,10))

    # fig = plt.figure(figsize=(4+1,4+1))
    outer_gs = gridspec.GridSpec(2, 2, width_ratios=[5, 5], height_ratios=[5, 5])
    # gs = GridSpec(3, 3, width_ratios=[5, 4, 1], height_ratios=[5, 1, 4])

    ax1 = fig.add_subplot(outer_gs[0,0])
    plt.text(s='A',x=-0.1,y=1.02,fontsize=15,fontweight='semibold',transform=ax1.transAxes)

    sns.violinplot(data=np.asarray(all_guesses),color='lightgray',alpha=0.1)
    plt.plot(np.mean(all_guesses,axis=0),'k',label='Average')
    plt.ylim(0,10)
    plt.xticks(range(len(np.mean(all_guesses,axis=0))),range(1,len(np.mean(all_guesses,axis=0))+1))
    plt.xlabel('Number of phenotypes')
    plt.ylabel('Mean Squared Error')


    # ax2 = fig.add_subplot(outer_gs[0,1])
    plt.text(s='B',x=-0.1,y=1.02,fontsize=15,fontweight='semibold',transform=ax2.transAxes)
    inner_gs = gridspec.GridSpecFromSubplotSpec(2, 2,subplot_spec = outer_gs[0,1], width_ratios=[4, 1], height_ratios=[1, 4],hspace=0.0,wspace=0.0)

    # plt.text(s='Visualization of Subtle predicting far',x=0.1,y=0.5)
    # plt.scatter(both_old,dhats[0],alpha=0.5,label='1 component model')
    # plt.scatter(both_old,dhats[model],alpha=0.5,label=f'{model + 1} component model')
    # plt.xlabel('Measured Training Data')
    # plt.ylabel('Estimated Training Data')
    # this_min = min([np.min(both_old),np.min(dhats[model])])
    # this_max = max([np.max(both_old),np.max(dhats[model])])
    # plt.plot([this_min,this_max],[this_min,this_max],'k--')
    # xdisplay, ydisplay = ax2.transAxes.transform_point((0.3, 0.1))

    # plt.annotate(fr'1 component model, $R^2$ = {tools.var_explained(both_old,dhats[0])[0]:.3g}',xy=(0.25,0.1),
    #              color=sns.color_palette()[0],xycoords='axes fraction')

    # xdisplay, ydisplay = ax2.transAxes.transform_point((0.3, 0.0))
    # plt.annotate(fr'{model+1} component model, $R^2$ = {tools.var_explained(both_old,dhats[model])[0]:.3g}',xy=(0.25,0.05),
    #              color=sns.color_palette()[1],xycoords='axes fraction')
    # plt.legend()

    ax3 = fig.add_subplot(outer_gs[1,0])
    plt.text(s='C',x=-0.1,y=1.02,fontsize=15,fontweight='semibold',transform=ax3.transAxes)

    largescale_predictions_graph(ax3,this_fitness,train,test,both_new,guesses,model,test_conditions)

    plt.xlabel('Condition')



    # ax5.xlabel(f'{x_d+1}')
    # ax5.ylabel(f'{y_d+1}')

    plt.tight_layout()

    # plt.savefig('Figure4_working_testBC_withC.pdf',bbox_inches='tight')

    return fig

import matplotlib.ticker as plticker

def prediction_examples(fig,gs,dataset,this_data,example_conditions,models,legend=True,legend_cols=4,weighted=True,label='default'):

    test_mutant_data = dataset['testing_bcs']
    test_conditions = dataset['test_cols']
    guesses = dataset['guesses']
    both_new = dataset['both_new']
    
    this_gene_data = test_mutant_data
    this_gene_data = this_data[this_data['barcode'].isin(test_mutant_data)]

    types = this_gene_data['mutation_type'].values

    this_gene_locs = range(len(this_gene_data))


    for c,col in enumerate(example_conditions):

        axes = []
        ax_mins = []
        ax_maxs = []

        i = test_conditions.index(col)

        ax1 = fig.add_subplot(gs[c])
        axes.append(ax1)
        if label == 'default':
            # plt.text(s=f'{chr(66+2*c+1)}',x=-0.2,y=1.02,fontsize=15,fontweight='semibold',transform=ax1.transAxes)
            plt.text(s=f'{chr(66+c+1)}',x=-0.2,y=1.02,fontsize=15,fontweight='semibold',transform=ax1.transAxes)
        elif label != False:
            plt.text(s=f'{chr(66+2*c+1+label)}',x=-0.2,y=1.02,fontsize=15,fontweight='semibold',transform=ax2.transAxes)
        plt.title(f'{renamed_conditions[col.replace("_fitness","")]}')
        plt.text(s='1 component',x=0.03,y=0.9,transform=ax1.transAxes)
        if weighted:
            plt.text(s=fr'$\widetilde R^2=${tools.var_explained_weighted_by_type(both_new[:,i],guesses[0][:,i],types)[0]:.2f}',x=0.5,y=0.03,transform=ax1.transAxes)
        else:
            plt.text(s=fr'$R^2=${tools.var_explained(both_new[:,i],guesses[0][:,i])[0]:.2f}',x=0.5,y=0.03,transform=ax1.transAxes)
        rank = 0
        for bc in range(len(this_gene_data[col].values)):
    #             plt.scatter(this_gene_data[col].values[bc],guesses[rank][this_gene_locs[bc],i],color=mutant_colorset[this_gene_data['gene'].values[bc]],alpha=0.6)
            plt.scatter(this_gene_data[col].values[bc],guesses[rank][this_gene_locs[bc],i],color=mutant_colorset[this_gene_data['mutation_type'].values[bc]],alpha=0.75)
#                 plt.errorbar(this_gene_data[col].values[bc],guesses[rank][this_gene_locs[bc],i],xerr=this_gene_data[col.replace('_fitness','_error')].values[bc],\
#                              color=mutant_colorset[this_gene_data['mutation_type'].values[bc]],alpha=0.75)
        xmin1,xmax1 = plt.xlim()
        ax_mins.append(xmin1)
        ax_maxs.append(xmax1)
        ymin1,ymax1 = plt.ylim()
        ax_mins.append(ymin1)
        ax_maxs.append(ymax1)

        for m,model in enumerate(models):

            ax2 = fig.add_subplot(gs[len(example_conditions)*(m+1)+c])
            axes.append(ax2)
            # if label == 'default':
            #     plt.text(s=f'{chr(66+2*c+2)}',x=-0.2,y=1.02,fontsize=15,fontweight='semibold',transform=ax2.transAxes)
            # elif label != False:
            #     plt.text(s=f'{chr(66+2*c+2+label)}',x=-0.2,y=1.02,fontsize=15,fontweight='semibold',transform=ax2.transAxes)

            rank = model
            plt.text(s=f'{rank+1} components',x=0.03,y=0.9,transform=ax2.transAxes)
            if weighted:
                plt.text(s=fr'$\widetilde R^2=${tools.var_explained_weighted_by_type(both_new[:,i],guesses[rank][:,i],types)[0]:.2f}',x=0.5,y=0.03,transform=ax2.transAxes)
            else:
                plt.text(s=fr'$R^2=${tools.var_explained(both_new[:,i],guesses[rank][:,i])[0]:.2f}',x=0.5,y=0.03,transform=ax2.transAxes)

            
            for bc in range(len(this_gene_data[col].values)):
        #             plt.scatter(this_gene_data[col].values[bc],guesses[rank][this_gene_locs[bc],i],color=mutant_colorset[this_gene_data['gene'].values[bc]],alpha=0.6)
                plt.scatter(this_gene_data[col].values[bc],guesses[rank][this_gene_locs[bc],i],color=mutant_colorset[this_gene_data['mutation_type'].values[bc]],alpha=0.75)
    #                 plt.errorbar(this_gene_data[col].values[bc],guesses[rank][this_gene_locs[bc],i],xerr=this_gene_data[col.replace('_fitness','_error')].values[bc],\
    #                              color=mutant_colorset[this_gene_data['mutation_type'].values[bc]],alpha=0.75)
            xmin1,xmax1 = plt.xlim()
            ax_mins.append(xmin1)
            ax_maxs.append(xmax1)
            ymin1,ymax1 = plt.ylim()
            ax_mins.append(ymin1)
            ax_maxs.append(ymax1)
        
        bigmin = min(ax_mins)
        bigmax = max(ax_maxs)

        for a,ax in enumerate(axes):
            plt.sca(ax)
            plt.plot([bigmin,bigmax],[bigmin,bigmax],'k--')
            # plt.tight_layout()
            plt.xlim(bigmin,bigmax)
            plt.ylim(bigmin,bigmax)
            plt.gca().set_aspect('equal')

            loc = plticker.MultipleLocator(base=tick_base_calculator(bigmin,bigmax)) # this locator puts ticks at regular intervals
            ax.xaxis.set_major_locator(loc)
            ax.yaxis.set_major_locator(loc)

            if not a == len(axes)-1:
                # print(a,len(axes)-1)
                labels = [item.get_text() for item in ax.get_xticklabels()]
                empty_string_labels = ['']*len(labels)
                ax.set_xticklabels(empty_string_labels)

            plt.tight_layout()
            
            if c == 0:
                plt.ylabel('Predicted Fitness')
        
        plt.xlabel('Measured Fitness')

        if legend and c == 0:
            gene_list = np.unique(types)
            legend_split = np.ceil(len(gene_list)/legend_cols)
            for g,gene in enumerate(gene_list):
                x_loc = 0.01+np.floor((g)/legend_split)*1.2
                y_loc = 0.05*(legend_split-1)-0.15*(g%legend_split)-0.65
                plt.text(s=f"{gene.replace('_',' ')}",x=x_loc,y=y_loc,fontsize=12,
                      fontweight='semibold',color=mutant_colorset[gene],transform=ax.transAxes)

    return fig

def all_example_predictions(dataset,gene_list,this_data,example_conditions,model='default',weighted=True,per_row=10):

    all_guesses = dataset['CV_all_guesses']
    both_old = dataset['both_old']
    dhats = dataset['dhats']
    this_fitness  = dataset['this_fitness']
    train  = dataset['train']
    test = dataset['test']
    both_new = dataset['both_new']
    guesses = dataset['guesses']
    test_conditions  = dataset['test_conditions']
    if model == 'default':
        model = dataset['CV_best_rank_index']

    n_rows = int(np.ceil(len(example_conditions)/per_row))

    fig = plt.figure(figsize=(2*per_row,4*n_rows))
    outer_gs = gridspec.GridSpec(n_rows, 1)

    for row in range(n_rows):
        # plt.text(s='B',x=-0.1,y=1.02,fontsize=15,fontweight='semibold',transform=ax2.transAxes)
        # inner_gs = gridspec.GridSpec(2*len(n_rows),per_row,hspace=0.1)
        inner_gs = gridspec.GridSpecFromSubplotSpec(2,per_row,subplot_spec = outer_gs[row])
        prediction_examples(fig,inner_gs,dataset,this_data,example_conditions[row*per_row:(row+1)*per_row],model,weighted=weighted,legend=False,label=False)

    return fig


def Figure4_w_examples(dataset,gene_list,this_data,example_conditions,models='default',weighted=True):
    all_guesses = dataset['CV_all_guesses']
    both_old = dataset['both_old']
    dhats = dataset['dhats']
    this_fitness  = dataset['this_fitness']
    train  = dataset['train']
    test = dataset['test']
    both_new = dataset['both_new']
    guesses = dataset['guesses']
    test_conditions  = dataset['test_conditions']
    if models == 'default':
        models = [dataset['CV_best_rank_index']]

    fig = plt.figure(figsize=(6+(2)*len(example_conditions),12))

    # fig = plt.figure(figsize=(4+1,4+1))
    outer_gs = gridspec.GridSpec(2, 2, width_ratios=[6, 2*len(example_conditions)], height_ratios=[4, 4])
    # gs = GridSpec(3, 3, width_ratios=[5, 4, 1], height_ratios=[5, 1, 4])

    ax1 = fig.add_subplot(outer_gs[0,0])
    plt.text(s='A',x=-0.1,y=1.02,fontsize=15,fontweight='semibold',transform=ax1.transAxes)

    largescale_predictions_graph(ax1,this_fitness,train,test,both_new,guesses,models,test_conditions,dataset,this_data,weighted=weighted)

    plt.xlabel('Condition')


    # plt.text(s='B',x=-0.1,y=1.02,fontsize=15,fontweight='semibold',transform=ax2.transAxes)
    inner_gs = gridspec.GridSpecFromSubplotSpec(1+len(models),len(example_conditions),subplot_spec = outer_gs[0,1],hspace=0.1,wspace=0.2)
    prediction_examples(fig,inner_gs,dataset,this_data,example_conditions,models,weighted=weighted)



    return fig


def Figure5(dataset,gene_list,model='default'):

    all_guesses = dataset['CV_all_guesses']
    both_old = dataset['both_old']
    dhats = dataset['dhats']
    this_fitness  = dataset['this_fitness']
    train  = dataset['train']
    test = dataset['test']
    both_new = dataset['both_new']
    guesses = dataset['guesses']
    test_conditions  = dataset['test_conditions']
    if model == 'default':
        model = dataset['CV_best_rank_index']
        x_d = 0
        y_d = model
    else:
        x_d = model[0]
        y_d = model[1]



    fig = plt.figure(figsize=(8,8))

    outer_gs = gridspec.GridSpec(2, 2, width_ratios=[5, 5], height_ratios=[5, 5])
    inner_gs = gridspec.GridSpecFromSubplotSpec(2, 2,subplot_spec = outer_gs[1,1], width_ratios=[4, 1], height_ratios=[1, 4],hspace=0.0,wspace=0.0)

    x_d = model[0]
    y_d = model

    distance_comparison_figure(fig,inner_gs,dataset['distances'][x_d],dataset['distances'][y_d],
                                     dataset['centroids'][x_d],dataset['centroids'][y_d],
                                     dataset['avg_pairwise'][x_d],dataset['avg_pairwise'][y_d],
                                      gene_list,include_ancestor=False)

    return fig


def smoothsegment(seg, Nsmooth=100):
    return np.concatenate([[seg[0]], np.linspace(seg[1], seg[2], Nsmooth), [seg[3]]])

def plot_dendrogram(icoord,dcoord,labels,leaves,figsize, polar=False,gap=0.05,coloring=False,thresh=0.22,cmap=tools.mutant_colorset,namemap=False):
    if polar:
        dcoord = -np.log(dcoord+1)

        imax = icoord.max()
        imin = icoord.min()
        icoord = ((icoord - imin)/(imax - imin)*(1-gap) + gap/2)*2*np.pi
    with plt.style.context("seaborn-white"):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, polar=polar)

        for xs, ys in zip(icoord, dcoord):
            if polar:
                xs = smoothsegment(xs)
                ys = smoothsegment(ys)
                
#             if coloring:
#                 if ys < 
#             else:
                
            ax.plot(xs,ys, color="black")

        if polar:
            ax.spines['polar'].set_visible(False)
            ax.xaxis.grid(False)
            ax.set_rlabel_position(0)
            
            tick_locs = []
            for i,d in zip(icoord,dcoord):
                if d[0] == 0:
                    tick_locs.append(i[0])
                if d[3] == 0:
                    tick_locs.append(i[3])
        
            for label, angle, color in zip(labels, tick_locs, [cmap[key] for key in labels]):
                if namemap != False:
                    label = namemap[label]

                x = angle
                y = 0.02
                
                deg_angle = np.degrees(angle)

                if 90 < deg_angle < 180:
                    lab = ax.text(x,y,label,color=color,
                                  rotation=180+deg_angle,rotation_mode='anchor',
                                 ha='right',va='center')
                elif 180 <= deg_angle < 270: 
                    lab = ax.text(x,y,label,color=color,
                                  rotation=deg_angle-180,rotation_mode='anchor',
                                 ha='right',va='center')
                else:
                    lab = ax.text(x,y,label,color=color,
                                  rotation=deg_angle,rotation_mode='anchor',
                                 ha='left',va='center')

            ax.set_xticklabels([])
                



