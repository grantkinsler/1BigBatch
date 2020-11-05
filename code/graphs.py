import tools
import numpy as np
import pandas as p
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import gridspec
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.ticker as mtick
from itertools import combinations
from scipy.spatial import distance
from tools import mutant_colorset
from tools import condition_colorset
from tools import renamed_conditions
from tools import tick_base_calculator
from matplotlib.patches import ConnectionPatch
from adjustText import adjust_text
import copy
import matplotlib.ticker as plticker
sns.set_color_codes()



def improvement_delta(ax,ve_improvements,cutoff,train_conditions,test_conditions,focal_conditions,contrast_condition,contrast_color,jitters=[tools.jitter_point(0,0.5) for i in range(500)],style='Default',show_condition_names=False):
    
    if style == 'Default':
        line_color ='k'
        strong_color = 'k'

    elif style == 'dark':
        line_color = 'w'
        strong_color = 'gray'

    running_subtle = np.zeros(len(ve_improvements[0,:]))
    for c,col in enumerate(train_conditions):
        ax.scatter([-0.2+i+jitters[len(train_conditions)*c+cutoff+i] for i in range(len(ve_improvements[c,cutoff:]))],ve_improvements[c,cutoff:],
                    s=50,color='lightgray',marker='.',alpha=0.9)
        running_subtle += ve_improvements[c,:]

    print('Subtle Mean',np.mean(ve_improvements[range(len(train_conditions)),cutoff:],axis=0))
    print('Subtle Max',np.max(ve_improvements[range(len(train_conditions)),cutoff:],axis=0))
        
    ax.scatter([i-0.2 for i in range(len(ve_improvements[0,cutoff:]))],np.mean(ve_improvements[range(len(train_conditions)),cutoff:],axis=0),color=line_color,alpha=0.75,marker='_')

    print('Strong Mean',np.mean(ve_improvements[len(train_conditions):,cutoff:],axis=0))
    print({test_conditions[c]:ve_improvements[len(train_conditions)+c,cutoff:] for c in range(len(test_conditions))})

    for c,col in enumerate(test_conditions):
        c=len(train_conditions)+c
        if (col not in focal_conditions.keys()) and (col not in contrast_condition.keys()):
            sizes = [50 for i in range(len(ve_improvements[c,:]))]
            sizes = sizes[cutoff:]
            ax.scatter([0.2+i+jitters[len(test_conditions)*c+cutoff+i] for i in range(len(ve_improvements[c,cutoff:]))],ve_improvements[c,cutoff:],
                    s=sizes,color=strong_color,marker='.',alpha=0.5,label=renamed_conditions[col.replace('_fitness','')])

    for c,col in enumerate(test_conditions):
        c=len(train_conditions)+c
        if col in focal_conditions.keys():
            colors = [matplotlib.colors.to_hex(strong_color) for i in range(len(ve_improvements[c,:]))]
            for entry in focal_conditions[col][0]:
                colors[entry-2] = focal_conditions[col][1]
            colors = colors[cutoff:]
            
            sizes = [50 for i in range(len(ve_improvements[c,:]))]
            for entry in focal_conditions[col][0]:
                sizes[entry-2] = 50
            sizes = sizes[cutoff:]
            
            ax.scatter([0.2+i+jitters[len(test_conditions)*c+cutoff+i] for i in range(len(ve_improvements[c,cutoff:]))],ve_improvements[c,cutoff:],
                    s=sizes,color=colors,marker='.',alpha=1.0)
 
    for c,col in enumerate(test_conditions):
        if col in contrast_condition.keys():
            colors = [matplotlib.colors.to_hex(strong_color) for i in range(len(ve_improvements[c,:]))]
            for entry in contrast_condition[col]:
                colors[entry-2] = contrast_color
            colors = colors[cutoff:]
            
            sizes = [50 for i in range(len(ve_improvements[c,:]))]
            for entry in contrast_condition[col]:
                sizes[entry-2] = 50
            sizes = sizes[cutoff:]
            ax.scatter([0.2+i+jitters[len(test_conditions)*c+cutoff+i]  for i in range(len(ve_improvements[c,cutoff:]))],ve_improvements[c,cutoff:],
                    s=sizes,color=colors,marker='.',alpha=1.0) 

    if show_condition_names:
        for n,name in enumerate(focal_conditions.keys()):

            models = focal_conditions[name][0]
            color = focal_conditions[name][1]

            c = np.where(np.isin(test_conditions,name))[0][0]
            c = len(train_conditions) + c

            outputs = [ve_improvements[c][m-2] for m in models]
            # print(outputs)

            # for m,model in enumerate(models):
                # print(m)
                # plt.annotate(s=tools.renamed_conditions[name.replace('_fitness','')],xy=(model-2-cutoff,outputs[m]),xytext=(np.mean(models)-2-cutoff,np.mean(outputs)),
                #     color=color,ha='center',va='center')

            plt.text(s=tools.renamed_conditions[name.replace('_fitness','')],x=0.05,y=0.9-0.1*n,color=color,transform=plt.gca().transAxes)
    else:
        plt.text(s='Subtle Pertubations',x=0.95,y=0.9,color='gray',transform=plt.gca().transAxes,ha='right')
        plt.text(s='Strong Pertubations',x=0.95,y=0.8,color='k',transform=plt.gca().transAxes,ha='right')

    end = len(ve_improvements[0,:])+1
    ax.axhline(0,linestyle=':',color=line_color)
    plt.xticks(range(end-1-cutoff),range(2+cutoff,end+1))
    # plt.xticks(full_model[:-1],range(2,len(full_model)+1))
    plt.ylabel(r'Improvement due to component ($\Delta \widetilde R^2$)')
    # plt.ylabel(r'Improvement due to component ($\Delta R^2$)')
    plt.xlabel('Component')

    return ax


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
    plt.xlabel('Environment')

    plt.ylim(ymin,ymax)
    plt.yticks(np.linspace(ymin,ymax,yticknum),np.linspace(ymin,ymax,yticknum))

    return ax

# def fitness_tubes_graph_replicates(ax,this_data,mean_std_bygene,bc_list,m3_conditions,nonm3_conditions,m3_assoc,nonm3_assoc,gene_list,
#     ymin=-1.25,ymax=1.5,yticknum=12,legend=True,legend_cols=1,fontsize=12,style='default',side_bars=True,faded_alpha=0.3):

#     if style == 'dark':
#         plt.style.use('dark_background')
#         faded_alpha = 0.3
#         emph_alpha = 0.9
#         guide_color = 'lightgray'
#         guide_alpha=0.12
#         below_color = 'w'
#     else:
#         faded_alpha = faded_alpha
#         emph_alpha = 0.8
#         guide_color = 'gray'
#         guide_alpha = 0.07

#     mutant_data = this_data[this_data['barcode'].isin(bc_list)]

#     offset = {'IRA1_nonsense':0,
#               'IRA1_missense':0.1/3,
#               'Diploid':0,
#               'GPB2':0.2/3,
#               'PDE2':0.3/3,}
              
#     this_gene_data = mutant_data[mutant_data['mutation_type'].isin(gene_list)]

#     this_gene_locs = np.where(np.isin(mutant_data['barcode'].values,this_gene_data['barcode'].values))[0]
#     jitters = [tools.jitter_point(0,0.01) for bc in range(len(this_gene_data[m3_conditions[0]].values)) ]

#     all_conditions = list(m3_conditions) + list(nonm3_conditions)
#     print(all_conditions)

#     plt.ylim(ymin,ymax)

#     plt.axvline(x=len(m3_conditions)+0.5,color='gray',lw=1.0)

#     plt.axhline(y=0.0,color='k',linestyle=':',alpha=0.2)

#     ### groupings by condition
#     all_assoc = m3_assoc + nonm3_assoc

#     unique_out = np.unique(all_assoc,return_index=True,return_counts=True)

#     assoc_names = np.asarray(all_assoc)[np.sort(unique_out[1])]
#     assoc_counts = unique_out[2][np.argsort(unique_out[1])]
#     cum_counts = np.cumsum(assoc_counts)

#     x_tick_locs = []
#     for i in range(len(assoc_names)):
#         if (i % 2) == 0:
#             # print(i)
#             rect = matplotlib.patches.Rectangle((1+(cum_counts[i]-assoc_counts[i])-0.5,ymin),assoc_counts[i],ymax-ymin,
#                                             linewidth=0,edgecolor=guide_color,facecolor=guide_color,alpha=guide_alpha)
        
#             ax.add_patch(rect)
#         x_tick_locs.append((cum_counts[i]-0.5*assoc_counts[i]+0.5))

#     # 2 sigma rectangles in background
#     for gene in gene_list:
        
#         mean = mean_std_bygene[gene][0]
#         twosigma_top = mean_std_bygene[gene][0]+2*mean_std_bygene[gene][1]
#         twosigma_bottom = mean_std_bygene[gene][0]-2*mean_std_bygene[gene][1]
        
#         # print(gene,twosigma_bottom,twosigma_top,twosigma_top-twosigma_bottom)
#         diff = twosigma_top - twosigma_bottom
        
        
#         plt.axhline(twosigma_top,color=mutant_colorset[gene],linewidth=1.0,alpha=0.05)
#         plt.axhline(twosigma_bottom,color=mutant_colorset[gene],linewidth=1.0,alpha=0.05)
#         rect = matplotlib.patches.Rectangle((0,twosigma_bottom),len(all_conditions)+2,twosigma_top-twosigma_bottom,
#                                             linewidth=1,edgecolor=mutant_colorset[gene],facecolor=mutant_colorset[gene],alpha=0.02)
        
#         ax.add_patch(rect)
        
#         diff_transform = ax.transData.transform((0.5, diff))

#         if side_bars:
        
#             trans = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)

#             tops = ax.transData.transform((0,ymax))
#             bottoms = ax.transData.transform((0,ymin))
#             y_diff = tops[1]-bottoms[1]

#             # ax.annotate("", xy=(1.025+offset[gene], mean), xytext=(1.04+offset[gene], mean),xycoords=trans,
#             #         arrowprops=dict(arrowstyle=f'-[, widthB={diff}, lengthB=0.1,angleB=0',mutation_scale=4*10*1.25,lw=2.0,color=mutant_colorset[gene]))
#             width = diff/(ymax-ymin)*y_diff/2



#             ax.annotate("", xy=(1.025+offset[gene], mean), xytext=(1.04+offset[gene], mean),xycoords=trans,
#                 arrowprops=dict(arrowstyle=f'-[, widthB={width}, lengthB=4,angleB=0',mutation_scale=1, lw=2.0,color=mutant_colorset[gene]))




#     low_counter = 0   
#     for g,gene in enumerate(gene_list):
#         mean = mean_std_bygene[gene][0]
#         twosigma_top = mean_std_bygene[gene][0]+2*mean_std_bygene[gene][1]
#         twosigma_bottom = mean_std_bygene[gene][0]-2*mean_std_bygene[gene][1]
        
#         this_gene_data = mutant_data[mutant_data['mutation_type']==gene]

#         # data = np.median(np.asarray([this_gene_data[col].values for col in all_conditions]),axis=1)
#         for col in all_conditions:
#             if col not in this_gene_data.columns:
#                 print(col)
#         data = np.mean(np.asarray([this_gene_data[col].values for col in all_conditions]),axis=1)
#         uncertainty = np.std(np.asarray([this_gene_data[col].values for col in all_conditions]),axis=1)
       
#         colors = [matplotlib.colors.to_rgba(mutant_colorset[gene]) for i in range(len(data))]
#         colors = [item[:3]+(faded_alpha,) if twosigma_bottom < data[i] < twosigma_top else item[:3]+(emph_alpha,) for i,item in enumerate(colors) ]
        
        
#         plt.scatter(range(1,len(all_conditions)+1),data,marker='o',color=colors,label=gene)

#         toolow = np.where(data<ymin)[0]
#         for entry in toolow:
#             plt.annotate("", xy=(entry+1, ymin+0.2*(low_counter)), xytext=(entry+1, ymin+0.2+0.2*(low_counter)),arrowprops=dict(arrowstyle="->",lw=1.5,color=mutant_colorset[gene]))
#         if len(toolow) > 0:    
#             low_counter += 1
#     plt.xticks(x_tick_locs,[col.split('_fitness')[0] for col in assoc_names],rotation=90)
#     # plt.xticks(x_tick_locs,[renamed_conditions[col.split('_fitness')[0]] for col in assoc_names],rotation=90)
#     plt.ylim(ymin,ymax)
#     plt.xlim(0.5,len(all_conditions)+0.5)
#     plt.yticks(np.linspace(ymin,ymax,12),np.linspace(ymin,ymax,12))

#     if legend:
#         legend_split = np.ceil(len(gene_list)/legend_cols)
#         for g,gene in enumerate(gene_list):
#             x_loc = 0.01+np.floor((g)/legend_split)*0.3
#             y_loc = 0.05*(legend_split-1)-0.05*(g%legend_split)+0.02
#             plt.text(s=f"{gene.replace('_',' ')}",x=x_loc,y=y_loc,fontsize=fontsize,
#                   fontweight='semibold',color=mutant_colorset[gene],transform=ax.transAxes)


#     plt.ylabel('Relative Fitness')
#     plt.xlabel('Environment')

#     return ax

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

    plt.axvline(x=len(m3_conditions)+0.5,color='gray',lw=1.0)

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
    plt.xticks(x_tick_locs,[renamed_conditions[col.split('_fitness')[0]] for col in assoc_names],rotation=90)
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


    plt.ylabel('Fitness advantage relative to Ancestor ($\it{s}$ per cycle)')
    plt.xlabel('Environment')

    arrow_left = ax.transData.transform_point((2, 0))
    arrow_right = ax.transData.transform_point((len(m3_conditions)-1, 0))

    arrow_width = (arrow_right[0]-arrow_left[0])/2
    # print(arrow_left,arrow_right,arrow_width)

    trans_arrow = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transData)


    plt.annotate('Batches of the Evolution Condition', xy=(len(m3_conditions)/2+0.5, -1.65), xytext=(len(m3_conditions)/2+0.5, -1.75), 
        fontsize=10,ha='center', va='top',xycoords=trans_arrow,annotation_clip=False,
        arrowprops=dict(arrowstyle=f'-[, widthB={arrow_width}, lengthB=7.0', lw=1.0,mutation_scale=1.0))

    return ax


def fitness_tubes_graph(ax,this_data,mean_std_bygene,bc_list,m3_conditions,nonm3_conditions,gene_list,
    ymin=-1.25,ymax=1.5,yticknum=12,legend=True,legend_cols=1,fontsize=11,style='default',side_bars=True,faded_alpha=0.3,vline=True,tubes=True,eye_guides=True,gene_list_labels='default'):

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

    if gene_list_labels == 'default':
        gene_list_labels =gene_list

    mutant_data = this_data[this_data['barcode'].isin(bc_list)]

    # offset = {'IRA1_nonsense':0,
    #           'IRA1_missense':0.1/3,
    #           'Diploid':0,
    #           'GPB2':0.2/3,
    #           'PDE2':0.3/3,}

    offset = {'IRA1_nonsense':0,
          # 'IRA1_missense':0.1/3,
          'Diploid':0,
          'GPB2':0.1/3,
          'PDE2':0.2/3,}
              
    this_gene_data = mutant_data[mutant_data['mutation_type'].isin(gene_list)]

    this_gene_locs = np.where(np.isin(mutant_data['barcode'].values,this_gene_data['barcode'].values))[0]
    jitters = [tools.jitter_point(0,0.01) for bc in range(len(this_gene_data[m3_conditions[0]].values)) ]

    all_conditions = list(m3_conditions) + list(nonm3_conditions)

    plt.ylim(ymin,ymax)

    if vline:
        plt.axvline(x=len(m3_conditions)+0.5,color='gray',lw=1.0)
        # plt.axvline(x=len(m3_conditions)+0.5,color='gray',lw=1.0)

    plt.axhline(y=0.0,color='k',linestyle=':',alpha=0.2)

    ### eye guides
    if eye_guides:
        for i in range(int(np.ceil(len(nonm3_conditions)/3))):
            if (i % 2) == 0:
                # print(i)
                rect = matplotlib.patches.Rectangle((len(m3_conditions)+1+3*i-0.5,ymin),3,ymax-ymin,
                                                linewidth=0,edgecolor=guide_color,facecolor=guide_color,alpha=guide_alpha)
            
                ax.add_patch(rect)

    ## 2 sigma rectangles in background
    if tubes:
        for gene in gene_list:
            
            mean = mean_std_bygene[gene][0]
            twosigma_top = mean_std_bygene[gene][0]+2*mean_std_bygene[gene][1]
            twosigma_bottom = mean_std_bygene[gene][0]-2*mean_std_bygene[gene][1]
            
            # print(gene,twosigma_bottom,twosigma_top,twosigma_top-twosigma_bottom)
            diff = twosigma_top - twosigma_bottom
            
            
            # plt.axhline(twosigma_top,color=mutant_colorset[gene],linewidth=1.0,alpha=0.05)
            # plt.axhline(twosigma_bottom,color=mutant_colorset[gene],linewidth=1.0,alpha=0.05)
            rect = matplotlib.patches.Rectangle((0,twosigma_bottom),len(all_conditions)+2,twosigma_top-twosigma_bottom,
                                                linewidth=1,edgecolor=mutant_colorset[gene],facecolor=mutant_colorset[gene],alpha=0.02)
            
            # ax.add_patch(rect)
            
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
            print('too low',gene,all_conditions[entry],data[entry])
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
            gene_label = gene_list_labels[g] 
            x_loc = 0.01+np.floor((g)/legend_split)*0.3
            y_loc = 0.05*(legend_split-1)-0.05*(g%legend_split)+0.02
            plt.text(s=f"{gene_label.replace('_',' ')}",x=x_loc,y=y_loc,fontsize=fontsize,
                  fontweight='semibold',color=mutant_colorset[gene],transform=ax.transAxes)


    plt.ylabel('Fitness advantage relative to Ancestor ($\it{s}$ per cycle)')
    plt.xlabel('Environment')


    arrow_left = ax.transData.transform_point((1, 0))
    arrow_right = ax.transData.transform_point((len(m3_conditions), 0))

    arrow_width = (arrow_right[0]-arrow_left[0])/2
    print(arrow_left,arrow_right,arrow_width)

    trans_arrow = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transData)

    plt.annotate('Batches\nof the\nEvolution\nCondition', xy=(len(m3_conditions)/2+0.5, -1.55), xytext=(len(m3_conditions)/2+0.5, -1.65), 
        fontsize=10,ha='center', va='top',xycoords=trans_arrow,annotation_clip=False,
        arrowprops=dict(arrowstyle=f'-[, widthB={arrow_width}, lengthB=7.0', lw=1.0,mutation_scale=1.0))
    

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
    plt.xlabel('Environment')
    plt.axvline(x=len(sorted_m3_cols)-0.5,color='gray',lw=1.0)
    plt.axvline(x=len(below)-0.5,color='r',lw=1.0)
    # plt.yscale('log')

    # plt.tight_layout()
    plt.ylim(ymin,ymax)
    plt.yticks(range(ymin,ymax+1,ytick_interval),range(ymin,ymax+1,ytick_interval))

    # plt.text(s='Batches',x=len(sorted_m3_cols)/2-0.5, y=np.mean([ymin,ymax]),
    #     fontsize=12,fontweight='semibold',color='k',ha='center',va='center')

    plt.text(s='Subtle',x=len(below)/2-0.5, y=np.mean([ymin,ymax]),
        fontsize=12,fontweight='semibold',color='k',ha='center',va='center')

    plt.text(s='Strong',x=len(below)+len(above)/2-0.5, y=np.mean([ymin,ymax]),
        fontsize=12,fontweight='semibold',color='r',ha='center',va='center')

    plt.text(s=r'2$\sigma$',x=len(all_z_scores)+0.5,y=2.0,
        fontsize=12,color=below_color,alpha=0.4,ha='center',va='center')

    return ax

def largescale_predictions_graph(ax,this_fitness,train,test,both_new,guesses,models,test_conditions,dataset,this_data,n_perms=100,ymin=-0.5,ymax=1.0,
    guide_color='lightgray',weighted=True,style='default',permute=True,build_up=False):


    if style == 'dark':
        model_color = 'w'

    elif style == 'default':
        model_color = 'k'


    plt.axhline(0,color='gray',linestyle=':')

    perms = np.zeros(both_new.shape[1])

    # this_gene_data = this_data[this_data['barcode'].isin(test_mutant_data)]
    types = this_data[this_data['barcode'].isin(dataset['testing_bcs'])]['mutation_type'].values

    ## eye guides
    if build_up != False:
        rect = matplotlib.patches.Rectangle((0,ymin),this_fitness.shape[1]-len(test_conditions),ymax-ymin,
                                        linewidth=0,edgecolor='lightgray',facecolor='lightgray',alpha=0.2)
    
        ax.add_patch(rect)
    else:
        for i in range(int(np.ceil(len(test_conditions)/4))):
            if (i % 2) == 0:
                # print(i)
                rect = matplotlib.patches.Rectangle((4*i-0.5,ymin),4,ymax-ymin,
                                                linewidth=0,edgecolor='lightgray',facecolor='lightgray',alpha=0.2)
            
                ax.add_patch(rect)

    # n_perms = 1000
    if permute:
        for i in range(n_perms):
            perm_out = tools.SVD_predictions_train_test(this_fitness,train,test,by_condition=True,permuted_conditions=True)
            # perm = np.asarray([tools.var_explained(both_new[:,i],perm_out[5][model][:,i])[0] for i in range(both_new.shape[1])])
            if weighted:
                perm = np.asarray([tools.var_explained_weighted_by_type(both_new[:,i],perm_out[5][models[-1]][:,i],types)[0] for i in range(both_new.shape[1])])
            else:
                perm = np.asarray([tools.var_explained(both_new[:,i],perm_out[5][models[-1]][:,i])[0] for i in range(both_new.shape[1])])

            perms = perms + perm
            plt.plot(perm,'.',color='gray',alpha=0.01)
        plt.plot((perms/n_perms),'_',color=model_color,alpha=0.8,label='Permutation Average')

    for model in models:
        print(model+1,tools.var_explained_weighted_by_type(both_new,guesses[model],types)[0])

    # oneD = np.asarray([tools.var_explained(both_new[:,i],guesses[0][:,i])[0] for i in range(both_new.shape[1])])
    if weighted:
        oneD = np.asarray([tools.var_explained_weighted_by_type(both_new[:,i],guesses[0][:,i],types)[0] for i in range(both_new.shape[1])])
    else:
        oneD = np.asarray([tools.var_explained(both_new[:,i],guesses[0][:,i])[0] for i in range(both_new.shape[1])])

    plt.plot(oneD,'o',markeredgecolor=model_color,markerfacecolor='None',linestyle='',alpha=0.8,label='1 component model')

    plt.ylabel(r'Weighted Coefficient of Determination ($\widetilde R^2$)')
    plt.xlabel('Environment')


    plt.xticks(range(len(test_conditions)),[renamed_conditions[col.replace('_fitness','')] for col in test_conditions],rotation=90)
    
    plt.xlim(-0.5,len(test_conditions)-0.5)
    plt.ylim(ymin,ymax)


    if build_up != False:
        plt.savefig(f'{build_up}_0.pdf',bbox_inches='tight')

    models = models[::-1]
    # print(models)
    for model in models:
    
        if weighted:
            this_sse = np.asarray([tools.var_explained_weighted_by_type(both_new[:,i],guesses[model][:,i],types)[0] for i in range(both_new.shape[1])])
        else:
            this_sse = np.asarray([tools.var_explained(both_new[:,i],guesses[model][:,i])[0] for i in range(both_new.shape[1])])

        if model == models[0]:
            plt.plot(this_sse,'o',label=f'{model+1} component model',color='r',alpha=0.8)
        else:
            plt.plot(this_sse,'.',label=f'{model+1} component model',color=model_color,alpha=0.8)

        if build_up != False:
            plt.savefig(f'{build_up}_{model}.pdf',bbox_inches='tight')

    plt.legend(loc='lower left')

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



def prediction_examples(fig,gs,dataset,this_data,example_conditions,models,legend=True,legend_cols=4,weighted=True,label='default',style='default',

                 gene_label_key = {'Neutral':['ExpNeutral'],
                 'Diploid':['Diploid'],
                 "Diploid w/\nadd'l mutation":['Diploid_adaptive','Diploid + Chr11Amp','Diploid + Chr12Amp','Diploid + IRA1','Diploid + IRA2'],
                  '$\\bf{IRA1}$ $\\bf{nonsense}$':['IRA1_nonsense'],
                  '$\\bf{IRA1}$ $\\bf{missense}$':['IRA1_missense'],
                  '$\\bf{IRA2}$':['IRA2'],
                  '$\\bf{GPB1}$':['GPB1'],
                  '$\\bf{GPB2}$':['GPB2'],
                  '$\\bf{PDE2}$':['PDE2'],
                  'Other RAS/PKA':['RAS2','CYR1','TFS1'],
                  'TOR/SCH9 pathway':['KOG1','TOR1','SCH9'],
                  'Other adaptive':['other_adaptive']
                 }):

    if style == 'dark':
        font_color = 'w'
    elif style == 'default':
        font_color = 'k'

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
            if c == 0:
                plt.text(s=f'{chr(66+2*c+1)}',x=-0.2,y=1.02,fontsize=15,fontweight='semibold',transform=ax1.transAxes)
                # plt.text(s='B',x=-0.2,y=1.02,fontsize=15,fontweight='semibold',transform=ax1.transAxes)
        elif label != False:
            if c == 0:
                plt.text(s='B',x=-0.2,y=1.1,fontsize=15,fontweight='semibold',transform=ax1.transAxes)
        plt.title(f'{renamed_conditions[col.replace("_fitness","")]}')
        plt.text(s='1 component',x=0.03,y=0.9,transform=ax1.transAxes)
        if weighted:
            plt.text(s=fr'$\widetilde R^2=${tools.var_explained_weighted_by_type(both_new[:,i],guesses[0][:,i],types)[0]:.2f}',x=0.5,y=0.03,transform=ax1.transAxes)
        else:
            plt.text(s=fr'$R^2=${tools.var_explained(both_new[:,i],guesses[0][:,i])[0]:.2f}',x=0.5,y=0.03,transform=ax1.transAxes)
        rank = 0
        #         for bc in range(len(this_gene_data[col].values)):
        #     #             plt.scatter(this_gene_data[col].values[bc],guesses[rank][this_gene_locs[bc],i],color=mutant_colorset[this_gene_data['gene'].values[bc]],alpha=0.6)
        #             plt.scatter(this_gene_data[col].values[bc],guesses[rank][this_gene_locs[bc],i],color=mutant_colorset[this_gene_data['mutation_type'].values[bc]],alpha=0.75)
        # #                 plt.errorbar(this_gene_data[col].values[bc],guesses[rank][this_gene_locs[bc],i],xerr=this_gene_data[col.replace('_fitness','_error')].values[bc],\
        # #                              color=mutant_colorset[this_gene_data['mutation_type'].values[bc]],alpha=0.75)
        for bc in range(len(this_gene_data[col].values)):
            if this_gene_data['mutation_type'].values[bc] == 'Diploid':
                plt.scatter(this_gene_data[col].values[bc],guesses[rank][this_gene_locs[bc],i],color=mutant_colorset[this_gene_data['mutation_type'].values[bc]],alpha=0.3)
        for bc in range(len(this_gene_data[col].values)): 
            if this_gene_data['mutation_type'].values[bc] != 'Diploid': 
                plt.scatter(this_gene_data[col].values[bc],guesses[rank][this_gene_locs[bc],i],color=mutant_colorset[this_gene_data['mutation_type'].values[bc]],alpha=0.75)
    
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
                if this_gene_data['mutation_type'].values[bc] == 'Diploid':
                    plt.scatter(this_gene_data[col].values[bc],guesses[rank][this_gene_locs[bc],i],color=mutant_colorset[this_gene_data['mutation_type'].values[bc]],alpha=0.3)
            for bc in range(len(this_gene_data[col].values)): 
                if this_gene_data['mutation_type'].values[bc] != 'Diploid': 
                    plt.scatter(this_gene_data[col].values[bc],guesses[rank][this_gene_locs[bc],i],color=mutant_colorset[this_gene_data['mutation_type'].values[bc]],alpha=0.75)

            xmin1,xmax1 = plt.xlim()
            ax_mins.append(xmin1)
            ax_maxs.append(xmax1)
            ymin1,ymax1 = plt.ylim()
            ax_mins.append(ymin1)
            ax_maxs.append(ymax1)
        
        bigmin = min(ax_mins)
        bigmax = max(ax_maxs)

        # bigmin = 0
        # bigmax = 1.0

        for a,ax in enumerate(axes):
            plt.sca(ax)
            plt.plot([bigmin,bigmax],[bigmin,bigmax],color=font_color,linestyle='--')
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
            gene_list = gene_label_key.keys()
            legend_split = np.ceil(len(gene_list)/legend_cols)
            for g,gene in enumerate(gene_list):
                x_loc = 0.01+np.floor((g)/legend_split)*1.2
                # x_loc = 0.01+np.floor((g)/legend_split)*1.3-0.5
                y_loc = 0.05*(legend_split-1)-0.15*(g%legend_split)-0.65
                plt.text(s=f"{gene.replace('_',' ')}",x=x_loc,y=y_loc,fontsize=12,va='top',
                      fontweight='semibold',color=mutant_colorset[gene_label_key[gene][0]],transform=ax.transAxes)

    return fig

def all_example_predictions(dataset,gene_list,this_data,example_conditions,model='default',weighted=True,per_row=10,style='default'):

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
        prediction_examples(fig,inner_gs,dataset,this_data,example_conditions[row*per_row:(row+1)*per_row],model,weighted=weighted,legend=False,label=False,style=style)

    return fig


def largescale_predictions_with_improvement_leaveout(fig,gs,this_fitness,train,test,both_new,guesses,models,train_conditions,test_conditions,dataset,this_data,n_perms=100,ymin=-0.5,ymax=1.0,
    guide_color='lightgray',guide_alpha=0.2,weighted=True,style='default',permute=True,build_up=False,include_subtle=True,all_shown=False,labels=False,ylim_adjust=False):

    if style == 'dark':
        model_color = 'w'

    elif style == 'default':
        model_color = 'k'

    training_bcs = dataset['training_bcs']
    testing_bcs = dataset['testing_bcs']

    left_out_fits = tools.leave_one_out_analysis(this_data,train_conditions,test_conditions,training_bcs,testing_bcs,weighted=weighted)
    left_out_fits = left_out_fits[0]

    types = this_data[this_data['barcode'].isin(dataset['testing_bcs'])]['mutation_type'].values

    models = models[::-1]

    if weighted:
        full_oneD = np.asarray([tools.var_explained_weighted_by_type(both_new[:,i],guesses[0][:,i],types)[0] for i in range(both_new.shape[1])])
        full_models = [np.asarray([tools.var_explained_weighted_by_type(both_new[:,i],guesses[models[0]][:,i],types)[0] for i in range(both_new.shape[1])]),
                    np.asarray([tools.var_explained_weighted_by_type(both_new[:,i],guesses[models[1]][:,i],types)[0] for i in range(both_new.shape[1])])]

    else:
        full_oneD = np.asarray([tools.var_explained(both_new[:,i],guesses[0][:,i])[0] for i in range(both_new.shape[1])])
        full_models = [np.asarray([tools.var_explained(both_new[:,i],guesses[models[0]][:,i])[0] for i in range(both_new.shape[1])]),
                    np.asarray([tools.var_explained(both_new[:,i],guesses[models[1]][:,i])[0] for i in range(both_new.shape[1])])]
    # top_ax = fig.add_subplot(gs[1])
    top_ax = fig.add_subplot(gs[0])
    if labels:
        plt.text(s='A',x=-0.1,y=1.02,fontsize=15,fontweight='semibold',transform=top_ax.transAxes)

    plt.axhline(0,color='gray',linestyle=':')



    if weighted:
        plt.ylabel(r'Weighted Coefficient of Determination ($\widetilde R^2$)')
    else:
        plt.ylabel(r'Coefficient of Determination ($R^2$)')
    # plt.xlabel('Condition')

    # if include_subtle:
    #     plt.xticks(range(len(test_conditions)+len(train_conditions)),[renamed_conditions[col.replace('_fitness','')] for col in train_conditions + test_conditions],rotation=90)
    #     plt.xlim(-0.5,len(train_conditions)+len(test_conditions)-0.5)
    # else:
    #     plt.xticks(range(len(test_conditions)),[renamed_conditions[col.replace('_fitness','')] for col in test_conditions],rotation=90)
    #     plt.xlim(-0.5,len(test_conditions)-0.5)
    if include_subtle:
        plt.xticks(range(len(test_conditions)+len(train_conditions)),['' for col in train_conditions + test_conditions],rotation=90)
        plt.xlim(-0.5,len(train_conditions)+len(test_conditions)-0.5)
    else:
        plt.xticks(range(len(test_conditions)),['' for col in test_conditions],rotation=90)
        plt.xlim(-0.5,len(test_conditions)-0.5)




    

    if include_subtle:
        for left_out_index in range(len(train_conditions)):
                model0line, = plt.plot([left_out_index],[left_out_fits[train_conditions[left_out_index]][0][0]],marker='o',
                        markeredgecolor=model_color,markerfacecolor='None',linestyle='',alpha=0.8)

        if build_up != False:
            plt.savefig(f'{build_up}_subtle_1.pdf',bbox_inches='tight')

        for model in models:
            for left_out_index in range(len(train_conditions)):

                if model == models[0]:
                    model1line, = plt.plot([left_out_index],[left_out_fits[train_conditions[left_out_index]][model][0]],marker='o',
                            color='r',alpha=0.8)
                else:
                    model2line, = plt.plot([left_out_index],[left_out_fits[train_conditions[left_out_index]][model][0]],marker='.',
                            color=model_color,alpha=0.8)

            if build_up != False:
                plt.savefig(f'{build_up}_subtle_{model+1}.pdf',bbox_inches='tight')


    if all_shown:
        for left_out_index in range(len(train_conditions)):
            plt.plot(range(len(train_conditions),len(train_conditions)+len(test_conditions)),left_out_fits[train_conditions[left_out_index]][0][1:],marker='o',
                        markeredgecolor=model_color,markerfacecolor='None',linestyle='None',alpha=0.1)
            for model in models:
                if model == models[0]:
                    plt.plot(range(len(train_conditions),len(train_conditions)+len(test_conditions)),left_out_fits[train_conditions[left_out_index]][model][1:],marker='o',
                        color='r',linestyle='None',alpha=0.1)
                else:
                    plt.plot(range(len(train_conditions),len(train_conditions)+len(test_conditions)),left_out_fits[train_conditions[left_out_index]][model][1:],marker='.',
                        color=model_color,linestyle='None',alpha=0.1)
    else:
        arrayed = []
        for left_out_index in range(len(train_conditions)):
            arrayed.append(left_out_fits[train_conditions[left_out_index]][0][1:])

        arrayed = np.asarray(arrayed)

        # full_oneD = np.asarray([tools.var_explained_weighted_by_type(both_new[:,i],guesses[0][:,i],types)[0] for i in range(both_new.shape[1])])
        # plt.errorbar(range(len(train_conditions),len(train_conditions)+len(test_conditions)),np.median(arrayed,axis=0,yerr=np.asarray([np.mean(arrayed,axis=0)-np.min(arrayed,axis=0),np.max(arrayed,axis=0)-np.mean(arrayed,axis=0)]),
        #         marker='o',markeredgecolor=model_color,markerfacecolor='None',linestyle='None',alpha=0.8,ecolor='k')
        plt.errorbar(range(len(train_conditions),len(train_conditions)+len(test_conditions)),full_oneD,yerr=np.asarray([full_oneD-np.min(arrayed,axis=0),np.max(arrayed,axis=0)-full_oneD]),
                marker='o',markeredgecolor=model_color,markerfacecolor='None',linestyle='None',alpha=0.8,ecolor='k')

        if build_up != False:
            plt.savefig(f'{build_up}_strong_1.pdf',bbox_inches='tight')

        # plt.plot(range(len(train_conditions),len(train_conditions)+len(test_conditions)),left_out_fits[train_conditions[left_out_index]][0][1:],marker='o',
        #                 markeredgecolor='k',markerfacecolor='None',linestyle='None',alpha=0.1)
        for m,model in enumerate(models):
            arrayed = []

            for left_out_index in range(len(train_conditions)):
                arrayed.append(left_out_fits[train_conditions[left_out_index]][model][1:])


            full_thismodel = full_models[m]

            print(model)
            print(full_thismodel)

            arrayed = np.asarray(arrayed)
            if model == models[0]:
                # plt.errorbar([i-0.2 for i in range(arrayed.shape[1])],np.mean(arrayed,axis=0),yerr=2*np.std(arrayed,axis=0),marker='o',linestyle='None')
                plt.errorbar(range(len(train_conditions),len(train_conditions)+len(test_conditions)),full_thismodel,yerr=np.asarray([full_thismodel-np.min(arrayed,axis=0),np.max(arrayed,axis=0)-full_thismodel]),
                marker='o',color='r',linestyle='None',alpha=0.8)
            else:
                plt.errorbar(range(len(train_conditions),len(train_conditions)+len(test_conditions)),full_thismodel,yerr=np.asarray([full_thismodel-np.min(arrayed,axis=0),np.max(arrayed,axis=0)-full_thismodel]),
                marker='.',color=model_color,linestyle='None',alpha=0.8)

            if build_up != False:
                    plt.savefig(f'{build_up}_strong_{model+1}.pdf',bbox_inches='tight')


    lines = [Line2D([0], [0], marker='o',markeredgecolor=model_color,markerfacecolor='None',linestyle='None',alpha=0.8),
            Line2D([0], [0], marker='.',color=model_color,linestyle='None',alpha=0.8),
            Line2D([0], [0], marker='o',color='r',linestyle='None',alpha=0.8)]
    
    labels = ['1 component',f'{models[1]+1} components',f'{models[0]+1} components']
    top_ax.legend(lines, labels,loc='lower left')

    if build_up != False:
        plt.savefig(f'{build_up}_fulltop.pdf',bbox_inches='tight')

    # plt.legend(loc='lower left')
    plt.axvline(len(train_conditions)-0.5,color=model_color,lw=0.75)


    if ylim_adjust == False:
        plt.ylim(ymin,ymax)

    ymin,ymax = plt.ylim()

    ## eye guides
    if build_up != False:
        rect = matplotlib.patches.Rectangle((-0.5,ymin),this_fitness.shape[1]-len(test_conditions),ymax-ymin,
                                        linewidth=0,edgecolor='lightgray',facecolor='lightgray',alpha=0.2)
    
        top_ax.add_patch(rect)
    else:
        for i in range(int(np.ceil((len(train_conditions)+len(test_conditions))/3))):
            if (i % 2) == 0:
                # print(i)
                rect = matplotlib.patches.Rectangle((8+1+3*i-0.5,ymin),3,ymax-ymin,
                                                linewidth=0,edgecolor=guide_color,facecolor=guide_color,alpha=guide_alpha)
                top_ax.add_patch(rect)





    # bottom_ax = fig.add_subplot(gs[0])
    bottom_ax = fig.add_subplot(gs[1])

    # plt.ylabel(f'Percent of {models[0]+1} model\nexplained by {models[1]+1} model' )
    plt.ylabel(f'Percent improvement due\nto three minor components')
    plt.xlabel('Environment')

    plt.axhline(0.0,color='gray',linestyle=':')


    # plt.xticks(range(len(test_conditions)),[renamed_conditions[col.replace('_fitness','')] for col in test_conditions],rotation=90)
    # if include_subtle:
    #     plt.xticks(range(len(test_conditions)+len(train_conditions)),['' for col in train_conditions + test_conditions],rotation=90)
    #     plt.xlim(-0.5,len(train_conditions)+len(test_conditions)-0.5)
    # else:
    #     plt.xticks(range(len(test_conditions)),['' for col in test_conditions],rotation=90)
    #     plt.xlim(-0.5,len(test_conditions)-0.5)

    if include_subtle:
        plt.xticks(range(len(test_conditions)+len(train_conditions)),[renamed_conditions[col.replace('_fitness','')] for col in train_conditions + test_conditions],rotation=90)
        plt.xlim(-0.5,len(train_conditions)+len(test_conditions)-0.5)
    else:
        plt.xticks(range(len(test_conditions)),[renamed_conditions[col.replace('_fitness','')] for col in test_conditions],rotation=90)
        plt.xlim(-0.5,len(test_conditions)-0.5)

    n_m3_conditions = 9

    arrow_left = bottom_ax.transData.transform_point((0, 0))
    arrow_right = bottom_ax.transData.transform_point((n_m3_conditions+1, 0))

    arrow_width = (arrow_right[0]-arrow_left[0])/2
    # print(arrow_left,arrow_right,arrow_width)

    trans_arrow = matplotlib.transforms.blended_transform_factory(bottom_ax.transData, bottom_ax.transAxes)

    plt.annotate('Batches\nof the\nEvolution\nCondition', xy=(n_m3_conditions/2-0.5, -0.4), xytext=(n_m3_conditions/2-0.5, -0.55), 
        fontsize=10,ha='center', va='top',xycoords=trans_arrow,annotation_clip=False,
        arrowprops=dict(arrowstyle=f'-[, widthB={arrow_width}, lengthB=7.0', lw=1.0,mutation_scale=1.0))
    

    # plt.xlabel('Condition')
    
    model1 = models[0]
    model2 = models[1]

    if include_subtle:
        for left_out_index in range(len(train_conditions)):
            plt.plot([left_out_index],[(left_out_fits[train_conditions[left_out_index]][model1][0]-left_out_fits[train_conditions[left_out_index]][model2][0])/left_out_fits[train_conditions[left_out_index]][model1][0]],marker='o',
                        markeredgecolor=model_color,markerfacecolor=model_color)

    if all_shown:
        for left_out_index in range(len(train_conditions)):
            plt.plot(range(len(train_conditions),len(train_conditions)+len(test_conditions)),(np.asarray(left_out_fits[train_conditions[left_out_index]][model1][1:])-np.asarray(left_out_fits[train_conditions[left_out_index]][model2][1:]))/np.asarray(left_out_fits[subtle_conditions[left_out_index]][model1][1:]),marker='o',
                            markeredgecolor=model_color,markerfacecolor=model_color,linestyle='None',alpha=0.1)
    else:
        arrayed = []
        for left_out_index in range(len(train_conditions)):
            arrayed.append((np.asarray(left_out_fits[train_conditions[left_out_index]][model1][1:])-np.asarray(left_out_fits[train_conditions[left_out_index]][model2][1:]))/np.asarray(left_out_fits[train_conditions[left_out_index]][model1][1:]))
        
        full_difference = (full_models[0]-full_models[1])/full_models[0]
        print(full_difference)
        print(np.mean(full_difference))
        arrayed = np.asarray(arrayed)
        # plt.errorbar(range(len(train_conditions),len(train_conditions)+len(test_conditions)),np.median(arrayed,axis=0),yerr=np.asarray([np.mean(arrayed,axis=0)-np.min(arrayed,axis=0),np.max(arrayed,axis=0)-np.mean(arrayed,axis=0)]),
        #         marker='o',markeredgecolor=model_color,markerfacecolor=model_color,linestyle='None',ecolor=model_color)
        plt.errorbar(range(len(train_conditions),len(train_conditions)+len(test_conditions)),full_difference,yerr=np.asarray([full_difference-np.min(arrayed,axis=0),np.max(arrayed,axis=0)-full_difference]),
                marker='o',markeredgecolor=model_color,markerfacecolor=model_color,linestyle='None',ecolor=model_color)
    if ylim_adjust == False:
        plt.ylim(-0.1,0.8)
    ymin,ymax = plt.ylim()

    bottom_ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    

    if build_up != False:
        rect = matplotlib.patches.Rectangle((-0.5,ymin),this_fitness.shape[1]-len(test_conditions),ymax-ymin,
                                        linewidth=0,edgecolor='lightgray',facecolor='lightgray',alpha=0.2)
    
        bottom_ax.add_patch(rect)
        plt.savefig(f'{build_up}_bottom.pdf',bbox_inches='tight')
    else:
        for i in range(int(np.ceil((len(train_conditions)+len(test_conditions))/3))):
            if (i % 2) == 0:
                # print(i)
                rect = matplotlib.patches.Rectangle((8+1+3*i-0.5,ymin),3,ymax-ymin,
                                                linewidth=0,edgecolor=guide_color,facecolor=guide_color,alpha=guide_alpha)
            
                bottom_ax.add_patch(rect)

    plt.axvline(len(train_conditions)-0.5,color=model_color,lw=0.75)

    return gs

def Figure4_leaveout_w_examples(dataset,gene_list,this_data,example_conditions,models='default',weighted=True,style='default',permute=True,build_up=False,labels=False,subtle=True,ylim_adjust=False):
    all_guesses = dataset['CV_all_guesses']
    both_old = dataset['both_old']
    dhats = dataset['dhats']
    this_fitness  = dataset['this_fitness']
    train  = dataset['train']
    test = dataset['test']
    both_new = dataset['both_new']
    guesses = dataset['guesses']
    test_conditions  = dataset['test_conditions']
    train_conditions = dataset['train_conditions']
    if models == 'default':
        models = [dataset['CV_best_rank_index']]

    fig = plt.figure(figsize=(6+(2)*len(example_conditions),12))

    # fig = plt.figure(figsize=(4+1,4+1))
    outer_gs = gridspec.GridSpec(2, 2, width_ratios=[6, 2*len(example_conditions)], height_ratios=[4, 4])
    # gs = GridSpec(3, 3, width_ratios=[5, 4, 1], height_ratios=[5, 1, 4])

    

    # left_gs = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec = outer_gs[0,0],height_ratios=[1,3],hspace=0.05,wspace=0.25)
    left_gs = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec = outer_gs[0,0],height_ratios=[3,1],hspace=0.05,wspace=0.25)
    
    # ax1 = fig.add_subplot(left_gs[0])

    # if labels:
    #     plt.text(s='A',x=-0.1,y=1.02,fontsize=15,fontweight='semibold',transform=left_gs.transAxes)


    # largescale_predictions_graph(ax1,this_fitness,train,test,both_new,guesses,models,test_conditions,dataset,this_data,
    #                             weighted=weighted,style=style,permute=permute,build_up=build_up)


    largescale_predictions_with_improvement_leaveout(fig,left_gs,this_fitness,train,test,both_new,guesses,models,train_conditions,test_conditions,dataset,this_data,
                                            weighted=weighted,style=style,permute=permute,build_up=build_up,include_subtle=subtle,labels=labels,ylim_adjust=ylim_adjust)

    # ax2 = fig.add_subplot(left_gs[1])





    # plt.text(s='B',x=-0.1,y=1.02,fontsize=15,fontweight='semibold',transform=ax2.transAxes)
    inner_gs = gridspec.GridSpecFromSubplotSpec(1+len(models),len(example_conditions),subplot_spec = outer_gs[0,1],hspace=0.1,wspace=0.25)
    
    # plt.text(s='B',x=-0.1,y=1.02,fontsize=15,fontweight='semibold',transform=inner_gs.transAxes)

    prediction_examples(fig,inner_gs,dataset,this_data,example_conditions,models,weighted=weighted,label=labels,style=style)

    # ax2 = fig.add_subplot(outer_gs[0,1])
    # if labels:
    #     plt.text(s='C',x=1.1,y=1.02,fontsize=15,fontweight='semibold',transform=ax1.transAxes)



    return fig


def prediction_accuracy_by_type_subtle(dataset,this_data,model,testing_only_types,n_perms=1000,replacement_names = {60700:'SSK2'},uniques='Default',style='Default'):

    if style == 'Default':
        # constrast_color = 'k'
        line_color = 'k'
        zeroline_color = 'gray'
        dot_alpha = 0.3
    elif style == 'dark':
        # contrast_color = 'w'
        line_color = 'lightgray'
        dot_alpha=0.7

    all_jitters = [tools.jitter_point(0) for i in range(1000)]


    all_guesses = dataset['CV_all_guesses']
    both_old = dataset['both_old']
    dhats = dataset['dhats']
    this_fitness = dataset['this_fitness']
    this_error = dataset['this_error']
    train  = dataset['train']
    test = dataset['test']
    both_new = dataset['both_new']
    guesses = dataset['guesses']
    # model = dataset_examples['CV_best_rank_index']
    test_conditions  = dataset['test_conditions']
    train_conditions = dataset['train_conditions']

    leave_one_out_output = tools.leave_one_out_analysis(this_data,train_conditions,test_conditions,dataset['training_bcs'],dataset['testing_bcs'],weighted=True)

    subtle_guesses = leave_one_out_output[1]

    all_conditions = train_conditions + test_conditions

    all_bcs = sorted(list(dataset['training_bcs'])+list(dataset['testing_bcs']))
    testing_bcs = list(dataset['testing_bcs'])
    these_bcs = testing_bcs
    these_genes = this_data[this_data['barcode'].isin(these_bcs)]['mutation_type'].values
        
    for bc,gene in replacement_names.items():
        if bc in these_bcs:
            loc = np.where(np.asarray(these_bcs)==bc)[0][0]
            these_genes[loc] = gene

    # gene_bc = np.asarray([f'{gene} ({bc})' for bc,gene in zip(these_bcs,these_genes)])
    gene_bc = np.asarray([f'{gene}' for bc,gene in zip(these_bcs,these_genes)])
    
    m3_0_fitness = this_fitness[np.where(np.isin(all_bcs,these_bcs))[0]][:,0]
    
    gene_f_names = [f'{gene} ({f})' for gene,f in zip(these_genes,m3_0_fitness)]

    testing_only_accuracy = []
    perm_accuracies = []


    for c,condition in enumerate(train_conditions):

        fig = plt.figure()
        
        plt.title(f'{condition}')

        condition_loc = np.where(np.isin(all_conditions,condition))[0][0]

        this_f = this_fitness[np.where(np.isin(all_bcs,these_bcs))[0]][:,condition_loc] 
        this_e = this_error[np.where(np.isin(all_bcs,these_bcs))[0]][:,condition_loc]


        est_list = []

        # print(len(subtle_guesses[0][len(model)-1]))
        print(len(model)-1)
            
        estimates = [(np.abs(subtle_guesses[c][len(model)-1][bc,0]-this_f[bc]))/this_e[bc] for bc in range(len(these_bcs))]

        diff = np.asarray(estimates)

        sorting = range(len(these_genes))
        print(sorting)
        sorted_genes = copy.copy(these_genes[sorting])
        
        label_locs = []
        label_names = []
        verts = []
        gene1 = sorted_genes[0]
        

        unis = np.unique(these_genes)
        gene_means = []
        for gene in unis:
            here_locs = np.where(these_genes==gene)[0]
            gene_means.append(np.mean(diff[here_locs]))

        if uniques == 'Default': # include other adaptive here!!!
            unique_genes = [gene for gene in unis[np.argsort(gene_means)[::-1]] if gene not in ['other','NotSequenced','NotSequenced_adaptive','IRA1_other','ExpNeutral']]
            unique_names = [gene.replace('_',' ') for gene in unique_genes]
        else:
            unique_genes = uniques[0]
            unique_names = uniques[1]
        
        for g,gene in enumerate(unique_genes):
            if type(gene) != list:
                gene = [gene]
            
            locs = np.where(np.isin(sorted_genes,gene))[0]
            locs = np.asarray((locs))

            this_x = g
            
            colors = [mutant_colorset[this_g] for this_g in sorted_genes[locs]]
            if (len(locs) > 1) and (len(gene) ==1):
                plt.scatter([this_x+all_jitters[i] for i in range(len(locs))],(diff[sorting][locs]),color=colors,alpha=dot_alpha)
                plt.plot([this_x-0.2,this_x+0.2],[np.mean(diff[sorting][locs]),np.mean(diff[sorting][locs])],color=line_color,alpha=0.8)
            
            elif len(locs) > 1:
                plt.scatter([this_x+all_jitters[i] for i in range(len(locs))],(diff[sorting][locs]),color=colors,alpha=dot_alpha)

            else:
                plt.scatter([this_x],(diff[sorting][locs]),color=colors,alpha=dot_alpha)

        testing_only_locs = np.where(np.isin(sorted_genes,testing_only_types))[0]
        testing_only_locs = np.asarray((testing_only_locs))
        print(testing_only_locs)
        testing_only_accuracy.append(np.mean(diff[sorting][testing_only_locs]))

        perm_accuracies.append([])
        for perm in range(n_perms):

            locs = np.random.choice(range(len(diff[sorting])),len(testing_only_locs),replace=False)
            perm_accuracies[c].append(np.mean(diff[sorting][locs]))


        
        plt.axhline(0,linestyle=':',color=zeroline_color)

        # plt.ylim(-16,11)
#         plt.ylabel(f'Percent Improvement\nfrom\n{less_model}th component',rotation=90)
        plt.ylabel(f'Prediction Error per Mutant (units of measurement error)',rotation=90)
        
        for v in verts:
            plt.axvline(v,lw=0.5,color='lightgray')

        plt.xticks(range(len(unique_genes)),unique_names,rotation=90)
            
        plt.savefig(f'prediction_accuracy_by_type_{condition}.pdf',bbox_inches='tight')

    return testing_only_accuracy, perm_accuracies

def prediction_accuracy_by_type(dataset,this_data,model,testing_only_types,n_perms=1000,replacement_names = {60700:'SSK2'},uniques='Default',style='Default'):

    if style == 'Default':
        # constrast_color = 'k'
        line_color = 'k'
        zeroline_color = 'gray'
        dot_alpha = 0.3
    elif style == 'dark':
        # contrast_color = 'w'
        line_color = 'lightgray'
        dot_alpha=0.7

    all_jitters = [tools.jitter_point(0) for i in range(1000)]


    all_guesses = dataset['CV_all_guesses']
    both_old = dataset['both_old']
    dhats = dataset['dhats']
    this_fitness = dataset['this_fitness']
    this_error = dataset['this_error']
    train  = dataset['train']
    test = dataset['test']
    both_new = dataset['both_new']
    guesses = dataset['guesses']
    # model = dataset_examples['CV_best_rank_index']
    test_conditions  = dataset['test_conditions']
    train_conditions = dataset['train_conditions']

    all_conditions = train_conditions + test_conditions

    all_bcs = sorted(list(dataset['training_bcs'])+list(dataset['testing_bcs']))
    testing_bcs = list(dataset['testing_bcs'])
    these_bcs = testing_bcs
    these_genes = this_data[this_data['barcode'].isin(these_bcs)]['mutation_type'].values
        
    for bc,gene in replacement_names.items():
        if bc in these_bcs:
            loc = np.where(np.asarray(these_bcs)==bc)[0][0]
            these_genes[loc] = gene

    # gene_bc = np.asarray([f'{gene} ({bc})' for bc,gene in zip(these_bcs,these_genes)])
    gene_bc = np.asarray([f'{gene}' for bc,gene in zip(these_bcs,these_genes)])
    
    m3_0_fitness = this_fitness[np.where(np.isin(all_bcs,these_bcs))[0]][:,0]
    
    gene_f_names = [f'{gene} ({f})' for gene,f in zip(these_genes,m3_0_fitness)]

    testing_only_accuracy = []
    perm_accuracies = []


    for c,condition in enumerate(test_conditions):

        fig = plt.figure()
        
        plt.title(f'{condition}')

        condition_loc = np.where(np.isin(all_conditions,condition))[0][0]

        this_f = this_fitness[np.where(np.isin(all_bcs,these_bcs))[0]][:,condition_loc] 
        this_e = this_error[np.where(np.isin(all_bcs,these_bcs))[0]][:,condition_loc]


        est_list = []
            

        ve, old_mut_locs, new_mut_locs, old_cond_locs, new_cond_locs, these_sigmas = tools.SVD_mixnmatch_locations(this_fitness,train,test,model)
        these_sigmas = these_sigmas[:len(model),:len(model)]

        estimates = [np.abs(np.dot(new_mut_locs,np.dot(these_sigmas,new_cond_locs.T))[bc,condition_loc-old_cond_locs.shape[0]]-this_f[bc])/this_e[bc] for bc in range(len(these_bcs))]
        # est_list.append(copy.copy(np.asarray(estimates)))

        # diff = est_list[0]-est_list[1]
        # diff = np.asarray(diff)
        diff = np.asarray(estimates)

        # if c == 0:
            # return_diffs[mm] = diff

        sorting = range(len(these_genes))
        sorted_genes = copy.copy(these_genes[sorting])
        
        label_locs = []
        label_names = []
        verts = []
        gene1 = sorted_genes[0]
        

        unis = np.unique(these_genes)
        gene_means = []
        for gene in unis:
            here_locs = np.where(these_genes==gene)[0]
            gene_means.append(np.mean(diff[here_locs]))

        if uniques == 'Default': # include other adaptive here!!!
            unique_genes = [gene for gene in unis[np.argsort(gene_means)[::-1]] if gene not in ['other','NotSequenced','NotSequenced_adaptive','IRA1_other','ExpNeutral']]
            unique_names = [gene.replace('_',' ') for gene in unique_genes]
        else:
            unique_genes = uniques[0]
            unique_names = uniques[1]
        
        for g,gene in enumerate(unique_genes):
            if type(gene) != list:
                gene = [gene]
            
            locs = np.where(np.isin(sorted_genes,gene))[0]
            locs = np.asarray((locs))

            this_x = g
            
            colors = [mutant_colorset[this_g] for this_g in sorted_genes[locs]]
            if (len(locs) > 1) and (len(gene) ==1):
                plt.scatter([this_x+all_jitters[i] for i in range(len(locs))],(diff[sorting][locs]),color=colors,alpha=dot_alpha)
                plt.plot([this_x-0.2,this_x+0.2],[np.mean(diff[sorting][locs]),np.mean(diff[sorting][locs])],color=line_color,alpha=0.8)
            
            elif len(locs) > 1:
                plt.scatter([this_x+all_jitters[i] for i in range(len(locs))],(diff[sorting][locs]),color=colors,alpha=dot_alpha)

            else:
                plt.scatter([this_x],(diff[sorting][locs]),color=colors,alpha=dot_alpha)

        testing_only_locs = np.where(np.isin(sorted_genes,testing_only_types))[0]
        testing_only_locs = np.asarray((testing_only_locs))
        print(testing_only_locs)
        testing_only_accuracy.append(np.mean(diff[sorting][testing_only_locs]))

        perm_accuracies.append([])
        for perm in range(n_perms):

            locs = np.random.choice(range(len(diff[sorting])),len(testing_only_locs),replace=False)
            perm_accuracies[c].append(np.mean(diff[sorting][locs]))


        
        plt.axhline(0,linestyle=':',color=zeroline_color)

        # plt.ylim(-16,11)
#         plt.ylabel(f'Percent Improvement\nfrom\n{less_model}th component',rotation=90)
        plt.ylabel(f'Prediction Error per Mutant (units of measurement error)',rotation=90)
        
        for v in verts:
            plt.axvline(v,lw=0.5,color='lightgray')

        plt.xticks(range(len(unique_genes)),unique_names,rotation=90)
            
        plt.savefig(f'prediction_accuracy_by_type_{condition}.pdf',bbox_inches='tight')

    return testing_only_accuracy, perm_accuracies


def Figure5_new(dataset_overall,dataset_examples,this_data,models,focal_conditions,full_model=range(9),replacement_names = {60700:'SSK2'},uniques='Default',contrast_color='k',style='Default',box_annotation=True,
    gene_rename = {'TOR1':'TOR/Sch9 Pathway','SCH9':'TOR/Sch9 Pathway','KOG1':'TOR/Sch9 Pathway','Diploid_adaptive':'High-fitness Diploid'}):
        #         unique_genes = [['Diploid + Chr11Amp','Diploid + Chr12Amp'],
    #                         ['KOG1','TFS1','RAS2','SCH9','TOR1','SSK2'],
    #                         ['GPB2'],['IRA1_nonsense'],['Diploid'],['IRA1_missense'],['PDE2'],['Diploid_adaptive']]
    #         unique_names = ['Diploid + Chr. Amp','Unique\nAdaptive Haploids','GPB2','IRA1 nonsense','Diploid','IRA1 missense','PDE2','Adaptive Diploids']
         
    if style == 'Default':
        # constrast_color = 'k'
        line_color = 'k'
        zeroline_color = 'gray'
        dot_alpha = 0.3
    elif style == 'dark':
        # contrast_color = 'w'
        line_color = 'lightgray'
        zeroline_color = 'gray'
        dot_alpha=0.7

    gene_label_names = {'GPB2':'$\it{GPB2}$',
 'Diploid + Chr11Amp':'Diploid + Chr11Amp',
 'Diploid + IRA1':'Diploid + $\it{IRA1}$',
 'CYR1':'$\it{CYR1}$',
 'IRA1 missense':'$\it{IRA1}$ $\it{missense}$',
 'PDE2':'$\it{PDE2}$',
 'TOR/Sch9 Pathway':'TOR/Sch9 Pathway',
 'Diploid + Chr12Amp':'Diploid + Chr12Amp',
 'TFS1':'$\it{TFS1}$',
 'IRA1 nonsense':'$\it{IRA1}$ $\it{nonsense}$',
 'RAS2':'$\it{RAS2}$',
 'IRA2':'$\it{IRA2}$',
 'Diploid':'Diploid',
 'High-fitness Diploid':"Diploid w/ add'l mutation",
 'Diploid + IRA2':'Diploid + $\it{IRA2}$',
 'GPB1':'$\it{GPB1}$',
 'SSK2':'$\it{SSK2}$'}

    highlight_colors =  ["#e41a1c","#377eb8","#4daf4a","#984ea3"]

    all_jitters = [tools.jitter_point(0) for i in range(1000)]

    n_perms = 100
    weighted= True
    ymax = 1.0
    ymin = -1.0

    all_guesses = dataset_examples['CV_all_guesses']
    both_old = dataset_examples['both_old']
    dhats = dataset_examples['dhats']
    this_fitness = dataset_examples['this_fitness']
    this_error = dataset_examples['this_error']
    train  = dataset_examples['train']
    test = dataset_examples['test']
    both_new = dataset_examples['both_new']
    guesses = dataset_examples['guesses']
    model = dataset_examples['CV_best_rank_index']
    test_conditions  = dataset_examples['test_conditions']
    train_conditions = dataset_examples['train_conditions']

    all_conditions = train_conditions + test_conditions

    all_bcs = sorted(list(dataset_examples['training_bcs'])+list(dataset_examples['testing_bcs']))
    testing_bcs = list(dataset_examples['testing_bcs'])
    these_bcs = testing_bcs
    these_genes = this_data[this_data['barcode'].isin(these_bcs)]['mutation_type'].values
        
    for bc,gene in replacement_names.items():
        if bc in these_bcs:
            loc = np.where(np.asarray(these_bcs)==bc)[0][0]
            these_genes[loc] = gene

    for g,gene in enumerate(these_genes):
        if gene in gene_rename.keys():
            these_genes[g] = gene_rename[gene]

    # gene_bc = np.asarray([f'{gene} ({bc})' for bc,gene in zip(these_bcs,these_genes)])
    gene_bc = np.asarray([f'{gene}' for bc,gene in zip(these_bcs,these_genes)])
    
    m3_0_fitness = this_fitness[np.where(np.isin(all_bcs,these_bcs))[0]][:,0]
    
    gene_f_names = [f'{gene} ({f})' for gene,f in zip(these_genes,m3_0_fitness)]

    

    ve_improvements = tools.improvements_component_by_condition_leaveout(dataset_overall,this_data,full_model=full_model)

    # left_out_fits = tools.leave_one_out_analysis(this_data,train_conditions,test_conditions,training_bcs,testing_bcs,weighted=weighted)

    contrast_condition = {}
    contrast_color = 'k'

    fig = plt.figure(figsize=(10,10))
    # fig = plt.figure(figsize=(20,10))
    outer_gs = GridSpec(2, 1, height_ratios=[1, 2],hspace=0.2)

    ### TOP PANEL (overall improvements)
    top_gs = gridspec.GridSpecFromSubplotSpec(1, 2,subplot_spec = outer_gs[0],width_ratios=[7, 3],wspace=0.3)

    top_left = fig.add_subplot(top_gs[0])
    plt.text(s='A',x=-0.1,y=1.02,fontsize=15,fontweight='semibold',transform=top_left.transAxes)

    c_jitters = [tools.jitter_point(0,0.05) for i in range(1000)]

    improvement_delta(top_left,ve_improvements,0,train_conditions,test_conditions,focal_conditions,contrast_condition,contrast_color,jitters=c_jitters,style=style)

    top_right = fig.add_subplot(top_gs[1])
    # plt.text(s='B',x=-0.1,y=1.02,fontsize=15,fontweight='semibold',transform=bottom_left.transAxes)

    cutoff = 5
    improvement_delta(top_right,ve_improvements,cutoff,train_conditions,test_conditions,focal_conditions,contrast_condition,contrast_color,jitters=c_jitters,style=style,show_condition_names=True)
    right_ymin,right_ymax = plt.ylim()
    
    plt.title('Magnification')

    plt.tight_layout()


    end = len(ve_improvements[0,:])+1

    if box_annotation:
        rect = matplotlib.patches.Rectangle((cutoff-0.5,right_ymin),end-(1+cutoff),right_ymax-right_ymin,
                                            linewidth=2,edgecolor='#525252',facecolor='white',alpha=0.3)

        top_left.add_patch(rect)

        con1 = ConnectionPatch(xyB=(1+cutoff-0.5+end-(2+cutoff),right_ymax), xyA=(0,1), coordsA="axes fraction", coordsB="data",
                              axesA=top_right, axesB=top_left, color="#525252",alpha=0.3,linewidth=2)
        con2 = ConnectionPatch(xyB=(1+cutoff-0.5+end-(2+cutoff),right_ymin), xyA=(0,0), coordsA="axes fraction", coordsB="data",
                              axesA=top_right, axesB=top_left, color="#525252",alpha=0.3,linewidth=2)
        top_right.add_artist(con1)
        top_right.add_artist(con2)


    ### BOTTOM PANEL(examples of improvements)

    bottom_gs = gridspec.GridSpecFromSubplotSpec(len(models), 1,subplot_spec = outer_gs[1])

    return_diffs = {}

    for mm,model_set in enumerate(models):
        this_bot = fig.add_subplot(bottom_gs[mm])
        if mm == 0:
            plt.text(s='B',x=-0.1,y=1.02,fontsize=15,fontweight='semibold',transform=this_bot.transAxes)
        
        model_name = model_set[0]
        model_list = model_set[1]
        interesting_conditions = model_set[2]

        for c,condition in enumerate(interesting_conditions):

            condition_loc = np.where(np.isin(all_conditions,condition))[0][0]

            this_f = this_fitness[np.where(np.isin(all_bcs,these_bcs))[0]][:,condition_loc]
            this_e = this_error[np.where(np.isin(all_bcs,these_bcs))[0]][:,condition_loc]

            plt.title(f'{model_name}th component')

            est_list = []
            
            for m,model in enumerate(model_list):

                ve, old_mut_locs, new_mut_locs, old_cond_locs, new_cond_locs, these_sigmas = tools.SVD_mixnmatch_locations(this_fitness,train,test,model)
                these_sigmas = these_sigmas[:len(model),:len(model)]

                estimates = [np.abs(np.dot(new_mut_locs,np.dot(these_sigmas,new_cond_locs.T))[bc,condition_loc-old_cond_locs.shape[0]]-this_f[bc])/this_e[bc] for bc in range(len(these_bcs))]
                est_list.append(copy.copy(np.asarray(estimates)))

            diff = est_list[0]-est_list[1]
            diff = np.asarray(diff)

            if c == 0:
                return_diffs[mm] = diff

            sorting = range(len(these_genes))
            sorted_genes = copy.copy(these_genes[sorting])
            
            label_locs = []
            label_names = []
            verts = []
            gene1 = sorted_genes[0]
            
            if mm==0 and c==0:
                unis = np.unique(these_genes)
                gene_means = []
                for gene in unis:
                    here_locs = np.where(these_genes==gene)[0]
                    gene_means.append(np.mean(diff[here_locs]))

                if uniques == 'Default':
                    unique_genes = [gene for gene in unis[np.argsort(gene_means)[::-1]] if gene not in ['other','other_adaptive','NotSequenced','NotSequenced_adaptive','IRA1_other','ExpNeutral']]
                    unique_names = [gene.replace('_',' ') for gene in unique_genes]
                else:
                    unique_genes = uniques[0]
                    unique_nanes = uniques[1]
            
            for g,gene in enumerate(unique_genes):
                if type(gene) != list:
                    gene = [gene]
                
                locs = np.where(np.isin(sorted_genes,gene))[0]
                locs = np.asarray((locs))

                this_x = len(unique_genes)*c+g
                
                colors = [mutant_colorset[this_g] for this_g in sorted_genes[locs]]
                if (len(locs) > 1) and (len(gene) ==1):
                    plt.scatter([this_x+all_jitters[i] for i in range(len(locs))],(diff[sorting][locs]),color=colors,alpha=dot_alpha)
                    plt.plot([this_x-0.2,this_x+0.2],[np.mean(diff[sorting][locs]),np.mean(diff[sorting][locs])],color=line_color,alpha=0.8)
                
                elif len(locs) > 1:
                    plt.scatter([this_x+all_jitters[i] for i in range(len(locs))],(diff[sorting][locs]),color=colors,alpha=dot_alpha)

                else:
                    plt.scatter([this_x],(diff[sorting][locs]),color=colors,alpha=dot_alpha)
            
            
            plt.axhline(0,linestyle=':',color=zeroline_color)

            # plt.ylim(-16,11)
    #         plt.ylabel(f'Percent Improvement\nfrom\n{less_model}th component',rotation=90)
            plt.ylabel(f'Improvement\n(units of measurement error)',rotation=90)
            
            for v in verts:
                plt.axvline(v,lw=0.5,color='lightgray')
                
        
        plt.xlim(-0.5,len(interesting_conditions)*len(unique_genes)-0.5)
        print(unique_names)

        if mm ==len(models)-1:
            plt.xticks(range(len(interesting_conditions)*len(unique_genes)),[gene_label_names[name] for name in np.tile(unique_names,len(interesting_conditions))],rotation=90)
            plt.xlabel('Mutation Type')
        else:
            plt.xticks(range(len(interesting_conditions)*len(unique_genes)),[],rotation=90)

        plt.ylim(-20,15)
        ymin,ymax = plt.ylim()
        
        rect = matplotlib.patches.Rectangle((len(unique_genes)-0.5,ymin),len(unique_genes),ymax-ymin,
                                                linewidth=0,edgecolor='gray',facecolor='gray',alpha=0.03)

        plt.gca().add_patch(rect)
        
        trans = matplotlib.transforms.blended_transform_factory(plt.gca().transData, plt.gca().transAxes)
        
        for c,condition in enumerate(interesting_conditions):
            if c == 0:
                plt.text(s=f"{renamed_conditions[condition.replace('_fitness','')]}",
                 x=c*len(unique_genes)+len(unique_genes)/2-0.5,y=0.95,
                ha='center',va='center',transform=trans,
                color=focal_conditions[interesting_conditions[c]][1],weight='semibold') 
            else:
                plt.text(s=f"{renamed_conditions[condition.replace('_fitness','')]}",
                 x=c*len(unique_genes)+len(unique_genes)/2-0.5,y=0.95,
                ha='center',va='center',transform=trans,
                color=focal_conditions[interesting_conditions[c]][1],weight='semibold')
                
                plt.axvline(c*len(unique_genes)-0.5,color='k',lw=1)

    return fig, return_diffs


def smoothsegment(seg, Nsmooth=100):
    return np.concatenate([[seg[0]], np.linspace(seg[1], seg[2], Nsmooth), [seg[3]]])

def plot_dendrogram(icoord,dcoord,labels,leaves,figsize, polar=False,gap=0.05,coloring=False,thresh=0.22,cmap=tools.mutant_colorset,namemap=False,fontsize=5):
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
                                 ha='right',va='center',fontsize=fontsize)
                elif 180 <= deg_angle < 270: 
                    lab = ax.text(x,y,label,color=color,
                                  rotation=deg_angle-180,rotation_mode='anchor',
                                 ha='right',va='center',fontsize=fontsize)
                else:
                    lab = ax.text(x,y,label,color=color,
                                  rotation=deg_angle,rotation_mode='anchor',
                                 ha='left',va='center',fontsize=fontsize)

            ax.set_xticklabels([])
                

def svd_noise_comparison_figure(ax,this_f,err,n_pulls,yscale='linear',permutation=False,annotate=True,output=False,show_error=True,style='default',build_up=False):
    
    if style == 'dark':
        marker_color = 'w'
    elif style == 'default':
        marker_color = 'k'

    U, s, V = np.linalg.svd(this_f)
    
    max_d = min(this_f.shape)

    if type(err) != np.float:
        error = err.flatten()
    else:
        error = [err for i in range(len(this_f.flatten()))]

    plt.xlabel('Number of Components')
    plt.ylabel('Variance explained by component')
    
    plt.xticks(np.arange(0,max_d+1,1),np.arange(0,max_d+1,1))
    plt.xlim(0.5,max_d+0.5)

    plt.yscale(yscale)
#     plt.tight_layout()

    # SVD on error alone
    noise_s_list = []
    perm_s_list = []
    for i in range(n_pulls):
        # this_set = np.asarray([np.random.normal(0,np.sqrt(error[i])) for i in range(len(this_f.flatten()))]).reshape(this_f.shape[0],this_f.shape[1])
        this_set = np.asarray([np.random.normal(0,error[i]) for i in range(len(this_f.flatten()))]).reshape(this_f.shape[0],this_f.shape[1])

        U, noise_s, V = np.linalg.svd(this_set)
        noise_s_list.append(noise_s)
        if show_error:
            plt.plot(np.arange(1,max_d+1),np.square(noise_s)/np.sum(np.square(s)),color='gray',alpha=0.1)

    if permutation:
        for i in range(n_pulls):
            this_set = np.random.permutation(this_f.ravel()).reshape(this_f.shape[0],this_f.shape[1])
            U, perm_s, V = np.linalg.svd(this_set)
            perm_s_list.append(perm_s)
            plt.plot(np.arange(1,max_d+1),np.square(perm_s)/np.sum(np.square(s)),color='r',alpha=0.1)

    if build_up != False:
        plt.ylim(5*10**-6,1.5)
        plt.savefig(build_up+'_1.pdf',bbox_inches='tight')
    # Mean empirical noise max
    mean_noise_max = np.mean(noise_s_list,axis=0)[0] 
    print(mean_noise_max)
    print(mean_noise_max**2/np.sum(np.square(s)))
    # print(mean_noise_max**2/np.sum(np.square(s))

    if show_error:
        plt.axhline(mean_noise_max**2/np.sum(np.square(s)),linestyle=':',color = marker_color ,alpha=0.8)

    if build_up != False:
        plt.savefig(build_up+'_2.pdf',bbox_inches='tight')

    if build_up != False:
        for i in range(1,max_d+1):
            l, = plt.plot(np.arange(1,i+1),np.square(s[:i])/np.sum(np.square(s)),color=marker_color,marker='o',alpha=0.8)
            plt.savefig(f'{build_up}_3_{i}.pdf',bbox_inches='tight')
            l.remove()
    
    plt.plot(np.arange(1,max_d+1),np.square(s)/np.sum(np.square(s)),color=marker_color,marker='o',alpha=0.8)
    


    max_detected = np.where(s < mean_noise_max)[0][0]
    max_s = s[max_detected-1]**2/np.sum(np.square(s))
    next_s = s[max_detected]**2/np.sum(np.square(s))

    

    # label_xs = [1.3,2.3,3.3
    # label_ys = []
    if annotate:
        for c in range(max_detected):
            this_s = s[c]**2/np.sum(np.square(s))
            plt.annotate(s=f'{100*this_s:.2f}%',xy=(c+1+0.3+0.1*c,this_s),ha='left',va='center')
        # labels.append(plt.text(x=c+1+0.2,y=this_s+0.1*this_s,s=f'{100*this_s:.2f}%'))
        # plt.annotate(
    # adjust_text(labels,x=np.arange(1,max_d+1),y=np.square(s)/np.sum(np.square(s)))
 
    # print(np.where(s < mean_noise_max))
    print(max_s,next_s,next_s/max_s)
    print(max_detected)

    if annotate:
        plt.xlabel('Number of Components')
        plt.ylabel('Variance explained by component')
    
    
    plt.xticks(np.arange(0,max_d+1,1),np.arange(0,max_d+1,1))
    plt.xlim(0.5,max_d+0.5)
#     plt.tight_layout()
    
    

    if output:
      return ax, max_detected, s, np.square(s)/np.sum(np.square(s))
    else:
      return ax

