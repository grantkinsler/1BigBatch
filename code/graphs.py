import tools
import numpy as np
import pandas as p
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
from itertools import combinations
from scipy.spatial import distance
from tools import mutant_colorset
from tools import condition_colorset
from tools import renamed_conditions
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


def fitness_tubes_graph(ax,this_data,mean_std_bygene,minimal_training_bcs,minimal_testing_bcs,m3_conditions,nonm3_conditions,gene_list,
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

    mutant_data = this_data[this_data['barcode'].isin((list(minimal_training_bcs)+list(minimal_testing_bcs)))]

    # m3_conditions = sorted_m3_cols
    # nonm3_conditions = sorted_nonm3_cols


    # this_gene = ['GPB2','IRA1','PDE2','Diploid']
    # this_gene = ['IRA1_nonsense','IRA1_missense','GPB2','PDE2','Diploid']

    offset = {'IRA1_nonsense':0,
              'IRA1_missense':0.1/3,
              'Diploid':0,
              'GPB2':0.2/3,
              'PDE2':0.3/3,}
              

    # this
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
        data = np.median(np.asarray([this_gene_data[col].values for col in all_conditions]),axis=1)
        
       
        colors = [matplotlib.colors.to_rgba(mutant_colorset[gene]) for i in range(len(data))]
        colors = [item[:3]+(faded_alpha,) if twosigma_bottom < data[i] < twosigma_top else item[:3]+(emph_alpha,) for i,item in enumerate(colors) ]
        
        
        plt.scatter(range(1,len(all_conditions)+1),data,marker='o',color=colors,label=gene)
        
        
        

        toolow = np.where(data<ymin)[0]
        for entry in toolow:
            plt.annotate("", xy=(entry+1, ymin+0.2*(low_counter)), xytext=(entry+1, ymin+0.2+0.2*(low_counter)),arrowprops=dict(arrowstyle="->",lw=1.5,color=mutant_colorset[gene]))
        if len(toolow) > 0:    
            low_counter += 1

    plt.xticks(range(1,len(all_conditions)+1),[renamed_conditions[col.split('_fitness')[0]] for col in all_conditions],rotation=90)
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

def largescale_predictions_graph(ax,this_fitness,train,test,both_new,guesses,model,test_conditions):

    plt.axhline(0,color='gray',linestyle=':')

    perms = np.zeros(both_new.shape[1])

    n_perms = 1000
    for i in range(n_perms):
        perm_out = tools.SVD_predictions_train_test(this_fitness,train,test,by_condition=True,permuted_conditions=True)
        perm = np.asarray([tools.var_explained(both_new[:,i],perm_out[5][model][:,i])[0] for i in range(both_new.shape[1])])
        perms = perms + perm
        plt.plot(perm,color='gray',alpha=0.01)
    plt.plot((perms/n_perms),color='k',alpha=0.8,label='Permutation Average')

    dumb = np.asarray([tools.var_explained(both_new[:,i],guesses[0][:,i])[0] for i in range(both_new.shape[1])])
    plt.plot(dumb,color='k',linestyle='--',alpha=0.8,label='1 component model')

    this_sse = np.asarray([tools.var_explained(both_new[:,i],guesses[model][:,i])[0] for i in range(both_new.shape[1])])
        
    plt.plot(this_sse,'o-',label=f'{model+1} component model',color='r',alpha=1.0)
        
    # for i in range(100):
    #     perm_out = tools.SVD_predictions_train_test(this_fitness,train,test,by_condition=True,permuted_mutants=True)
    #     perm = np.asarray([tools.sum_squared_error(perm_out[5][5][:,i],both_new[:,i]) for i in range(both_new.shape[1])])
    #     plt.plot((dumb - perm)/(dumb - min_sse),color='orange',linestyle='--')

    plt.ylabel(r'Coefficient of Determination ($R^2$)')

    plt.xticks(range(len(test_conditions)),[renamed_conditions[col.replace('_fitness','')] for col in test_conditions],rotation=90)
    plt.legend(loc='lower left')

    plt.ylim(-1.0,1.0)

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

    plt.ylabel(r'Coefficient of Determination ($R^2$)')

    plt.xticks(range(len(test_conditions)),[renamed_conditions[col.replace('_fitness','')] for col in test_conditions],rotation=90)
    plt.legend()

    plt.ylim(-1.0,1.0)

    return ax

def Figure4(dataset):
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

    fig = plt.figure(figsize=(8,8))

    ax1 = plt.subplot(221)

    plt.text(s='Visualization of folding process',x=0.1,y=0.5)


    ax2 = plt.subplot(222)

    sns.violinplot(data=np.asarray(all_guesses),color='lightgray',alpha=0.1)
    plt.plot(np.mean(all_guesses,axis=0),'k',label='Average')
    # plt.ylim(0,30)
    plt.xticks(range(len(np.mean(all_guesses,axis=0))),range(1,len(np.mean(all_guesses,axis=0))+1))
    plt.xlabel('Number of phenotypes')
    plt.ylabel('MSE')


    ax3 = plt.subplot(223)

    # plt.text(s='Visualization of Subtle predicting far',x=0.1,y=0.5)
    plt.scatter(both_old,dhats[0],alpha=0.5,label='1 component model')
    plt.scatter(both_old,dhats[model],alpha=0.5,label=f'{model + 1} component model')
    plt.xlabel('Measured Training Data')
    plt.ylabel('Estimated Training Data')
    this_min = min([np.min(both_old),np.min(dhats[model])])
    this_max = max([np.max(both_old),np.max(dhats[model])])
    plt.plot([this_min,this_max],[this_min,this_max],'k--')
    plt.annotate(fr'1 component model, $R^2$ = {tools.var_explained(both_old,dhats[0])[0]:.3g}',xy=(0.3*this_max,0.1),
                 color=sns.color_palette()[0],transform=ax3.transAxes)
    plt.annotate(fr'{model+1} component model, $R^2$ = {tools.var_explained(both_old,dhats[model])[0]:.3g}',xy=(0.3*this_max,0.0),
                 color=sns.color_palette()[1],transform=ax3.transAxes)
    # plt.legend()

    ax4 = plt.subplot(224)

    largescale_predictions_graph(ax4,this_fitness,train,test,both_new,guesses,model,test_conditions)

    plt.xlabel('Condition')

    plt.tight_layout()

    # plt.savefig('Figure4_working_testBC_withC.pdf',bbox_inches='tight')

    return fig





def distance_comparison_figure(distances_x,distances_y,geom_medians_x,geom_medians_y,avg_pairwise_x,avg_pairwise_y,gene_list,ylim='default',include_ancestor=True):

    fig = plt.figure(figsize=(4+1,4+1))
    gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],hspace=0.0,wspace=0.0)


    ## x data histogram
    ax1 = fig.add_subplot(gs[0])
    sns.despine(ax=ax1)
    sns.distplot(distances_x,kde=False,color='gray')

    # ax1.yaxis.set_visible(False)
    # ax1.spines['left'].set_visible(False)
    # plt.xlim(0,1.5)
    plt.xticks()

    ## y data histogram
    ax3 = fig.add_subplot(gs[3])
    sns.despine(ax=ax3)
    sns.distplot(distances_y,vertical=True,kde=False,color='gray')
    # ax3.xaxis.set_visible(False)
    # ax3.spines['bottom'].set_visible(False)
    # plt.ylim(0,1.5)
    plt.yticks()

    ## scatter 
    ax2 = fig.add_subplot(gs[2])

    plt.scatter(distances_x,distances_y,alpha=0.03,color='gray')

    plt.axvline(np.mean(distances_x),color='k',alpha=0.5)
    plt.axhline(np.mean(distances_y),color='k',alpha=0.5)

    xs = np.linspace(0,1.2)
    # plt.plot(xs,np.sqrt(4)/np.sqrt(6)*xs,'k--')



    for gene1, gene2 in combinations(gene_list,2):
        plt.plot(distance.euclidean(geom_medians_x[gene1],geom_medians_x[gene2]),
                    distance.euclidean(geom_medians_y[gene1],geom_medians_y[gene2]),
                    label=f'{gene1} and {gene2}',color=tools.mutant_colorset[gene1],marker='D',fillstyle = 'left',
                markerfacecoloralt=tools.mutant_colorset[gene2],linestyle='',markersize=7,markeredgecolor='k',alpha=0.9)
    if include_ancestor:
        for gene1 in gene_list:
            plt.plot(distance.euclidean(geom_medians_x[gene1],np.zeros(geom_medians_x[gene1].shape)),
                        distance.euclidean(geom_medians_y[gene1],np.zeros(geom_medians_y[gene1].shape)),
                        label=f'{gene1} and Ancestor',color=tools.mutant_colorset[gene1],marker='D',fillstyle = 'left',
                    markerfacecoloralt='k',linestyle='',markersize=7,markeredgecolor='k',alpha=0.9)

    for gene in gene_list:
        plt.plot(avg_pairwise_x[gene],avg_pairwise_y[gene],marker='o',linestyle='',color=tools.mutant_colorset[gene],alpha=0.8,label=f'Average pairwise for {gene}')

    # plt.ylim(0,1.5)
    # plt.xlim(0,1.5)
    # plt.legend(loc=(1.1,0.65),ncol=1)
    plt.legend(loc=(1.0,0.6),ncol=1)

    return fig








