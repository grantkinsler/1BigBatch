import numpy as np
import pandas as p
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.spatial import distance
from scipy.stats.mstats import gmean
from sklearn.linear_model import LinearRegression
import itertools
import copy

mutant_colorset = {'CYR1':'#cab2d6', # light purple
                 'Diploid':'#fb9a99', # light red
                 'Diploid + Chr11Amp':'#e31a1c', # dark red for adaptive diploids
                 'Diploid + Chr12Amp':'#e31a1c',
                 'Diploid + IRA1':'#e31a1c',
                 'Diploid + IRA2':'#e31a1c',
                 'GPB1':'#b2df8a',  # light green
                 'GPB2':'#33a02c',  # dark green
                 'IRA1':'#1f78b4', # dark blue
                 'IRA2':'#a6cee3', # light blue
                 'NotSequenced':'gray',
                 'PDE2':'#ff7f00',  # dark orange
                 'RAS2':'#ffff99', # yellow
                 'SCH9':'#6a3d9a', # dark purple for TOR mutants
                 'TFS1':'#6a3d9a',
                 'TOR1':'#6a3d9a',
                 'KOG1':'#6a3d9a', 
                 'other':'k'}

# old_colorset = {condition:sns.color_palette('Accent',len(old_conditions.keys()))[i] for i,condition in enumerate(old_conditions.keys())}
# bigbatch_colorset = {condition:sns.color_palette('Paired',len(bigbatch_conditions.keys()))[i] for i,condition in enumerate(bigbatch_conditions.keys())}

condition_colorset = {'13': (0.9921568627450981, 0.7529411764705882, 0.5254901960784314),
 '18': (1.0, 1.0, 0.6),
 '1BB_0.2MKCl': (0.8901960784313725, 0.10196078431372549, 0.10980392156862745),
 '1BB_0.2MNaCl': (0.984313725490196, 0.6039215686274509, 0.6),
 '1BB_0.5%Raf': (1.0, 0.4980392156862745, 0.0),
 '1BB_0.5MKCl': (0.9921568627450981, 0.7490196078431373, 0.43529411764705883),
 '1BB_1%Gly': (0.792156862745098, 0.6980392156862745, 0.8392156862745098),
 '1BB_1.4%Gluc': (0.6980392156862745, 0.8745098039215686, 0.5411764705882353),
 '1BB_1.8%Gluc': (0.2, 0.6274509803921569, 0.17254901960784313),
 '1BB_Baffle': (0.12156862745098039, 0.47058823529411764, 0.7058823529411765),
 '1BB_M3': (0.6509803921568628, 0.807843137254902, 0.8901960784313725),
 '20': (0.2196078431372549, 0.4235294117647059, 0.6901960784313725),
 '21': (0.9411764705882353, 0.00784313725490196, 0.4980392156862745),
 '23': (0.7490196078431373, 0.3568627450980392, 0.09019607843137253),
 '3': (0.4980392156862745, 0.788235294117647, 0.4980392156862745),
 '6': (0.7450980392156863, 0.6823529411764706, 0.8313725490196079)}

def flatten(list2d):
    return list(itertools.chain.from_iterable(list2d))


def jitter_point(mean,std=0.15):
    return np.random.normal(mean,std)


def calculate_fitness(X,O,Ancestor):
    
    sqdistances = distance.cdist(X,O,metric='sqeuclidean')
    Anc_dist2 = np.broadcast_to(distance.cdist(O,[Ancestor],metric='sqeuclidean').T,(X.shape[0],O.shape[0]))
    est_fitness = np.exp((Anc_dist2 - sqdistances) / 2.0) - 1.0

    return est_fitness

def calculate_fitness_linear(X,O,Ancestor):
    
    sigma = np.eye(X.shape[1],O.shape[0])
    
#     est_fitness = np.dot(X,np.dot(sigma,O.T))
    est_fitness = np.dot(X,O.T)

    return est_fitness

def loglikelihood_function(est_fitness,fitness,error,metric='euclidean'):    
        
    log2pi_variance = np.log(np.sqrt(2.0 * np.pi * error))
    inverse_variance = 1.0/error
    log_likelihood = -log2pi_variance - 0.5 * inverse_variance * np.square(fitness - est_fitness)

    return log_likelihood.sum()

def bic(ll,n,k):

    return np.log(n)*k - 2 * ll

def bic1(data,fit,rank):

    """ 
    Bayesian Information Criterion 1 from Bai and Ng 2002 and tested by Owen and Perry 2009
    """
    
    k = float(rank)
    m = float(data.shape[0])
    n = float(data.shape[1])

    return np.log(np.linalg.norm(data-fit)**2)  + k*(m+n)/(m*n)*np.log(m*n/(m+n))

def bic2(data,fit,rank):

    """ 
    Bayesian Information Criterion 2 from Bai and Ng 2002 and tested by Owen and Perry 2009
    """
    
    k = float(rank)
    m = float(data.shape[0])
    n = float(data.shape[1])
    C = np.min([np.sqrt(m),np.sqrt(n)])

    return np.log(np.linalg.norm(data-fit)**2)  + k*(m+n)/(m*n)*np.log(C**2)

def bic3(data,fit,rank):

    """ 
    Bayesian Information Criterion 3 from Bai and Ng 2002 and tested by Owen and Perry 2009
    """
    
    k = float(rank)
    m = float(data.shape[0])
    n = float(data.shape[1])
    C = np.min([np.sqrt(m),np.sqrt(n)])

    return np.log(np.linalg.norm(data-fit)**2)  + k*np.log(C**2)/C**2

def aic(ll,k):

    return 2*k - 2 * ll

def var_explained(data,model):
    
    ss_res = np.sum((data-model)**2)
    ss_tot = np.sum((data-np.mean(data))**2)
    
    return 1 - ss_res/ss_tot, ss_res, ss_tot

def SVD_condition_predictions(data,old_c,new_c,n_mutants,permuted_mutants=False,permuted_conditions=False,mse=False):
    
    """ Predicting new conditions from SVD fit on old conditions using ALL mutants """

    old_m = range(n_mutants)
    max_rank = len(old_c)

    if permuted_mutants and permuted_conditions:
        this_data = copy.copy(data)
        this_data[old_m,old_c] = np.random.permutation(this_data[old_m,old_c].ravel()).reshape(len(old_m),len(old_c))
        subset = this_data[np.repeat(old_m,len(old_c)),np.tile(old_c,len(old_m))].ravel()

    elif permuted_mutants: ### I think this is named wrong/confusingly - this is holding mutants in place and permuting the conditions
        this_data = copy.copy(data)
        for mut in old_m:
            this_data[mut,old_c] = np.random.permutation(this_data[mut,old_c])

    elif permuted_conditions: ### I think this is named wrong/confusingly - this is holding conditions in place and permuting the mutants

        this_data = np.swapaxes(copy.copy(data),0,1)
        for cond in old_c:
            this_data[cond,old_m] = np.random.permutation(this_data[cond,old_m])
        this_data = np.swapaxes(this_data,0,1)

    else:
        this_data = copy.copy(data)


    old_fitness = this_data[np.repeat(old_m,len(old_c)),np.tile(old_c,len(old_m))].reshape(len(old_m),len(old_c))
    new_fitness = this_data[np.repeat(old_m,len(new_c)),np.tile(new_c,len(old_m))].reshape(len(old_m),len(new_c))
    
    U, s2, V2 = np.linalg.svd(old_fitness)

    # old_condition_weights = np.dot(np.diag(s2),V2)

    reg = LinearRegression(fit_intercept=False).fit(U[:,:s2.shape[0]],new_fitness)

    new_condition_weights = reg.coef_.swapaxes(0,1)

    rank_fit = []
    rank_fit_by_condition = []

    for rank in range(1,max_rank+1):
        rank_fit_by_condition.append([])

        predicted_new = np.dot(U[:,:rank],new_condition_weights[:rank,:])
        if mse:
            rank_fit.append(np.sum(np.square(new_fitness-predicted_new)))
        else:
            rank_fit.append(var_explained(new_fitness,predicted_new)[0])

        for k in range(len(new_c)):
            if mse:
                rank_fit_by_condition[rank-1].append(np.sum(np.square(new_fitness[:,k]-predicted_new[:,k])))
            else:
                rank_fit_by_condition[rank-1].append(var_explained(new_fitness[:,k],predicted_new[:,k])[0])



    return rank_fit, rank_fit_by_condition


def SVD_predictions(data,folds,n_mutants,n_conditions,n_folds,permuted_mutants=False,permuted_conditions=False,mse=False):
    
    """ 
    Bi-cross validation using multiple folds of data matrix. 

    Method from Owen and Perry 2009.

    For each fold, we have the following data matrix:

                        "new conditions"  "old conditions"
    "new mutants"              A                  B
    "old mutants"              C                  D

    We first perform SVD on the D sub-matrix (using only old mutants and old conditions).
    For every pseudo inverse rank k approximation of D (denoted by D_k^+), we matrix multiply B * D_k^+ * C which gives the best estimate for A from the D_k approximation.

    We then evaluate prediction ability use the residual (eqn 3.3 from Owen and Perry 2009):

        A - B * D_k^+ * C 

    """


    max_rank = int((n_folds-1)*n_conditions/n_folds)
    all_folds = np.zeros(max_rank)

    fold_fits = []
    fold_fits_by_condition = []
    fold_fits_by_mutant = []
    for f,fold in enumerate(folds):

        

        fold_fits_by_condition.append([])
        fold_fits_by_mutant.append([])

        new_m = fold[0]
        new_c = fold[1]
        old_m = sorted([i for i in range(n_mutants) if i not in new_m])
        old_c = sorted([i for i in range(n_conditions) if i not in new_c])
        rank_fit = []
        true_fit = []

        if permuted_mutants and permuted_conditions:
            this_data = copy.copy(data)
            this_data[old_m,old_c] = np.random.permutation(this_data[old_m,old_c].ravel()).reshape(len(old_m),len(old_c))
            subset = this_data[np.repeat(old_m,len(old_c)),np.tile(old_c,len(old_m))].ravel()

        elif permuted_mutants:
            this_data = copy.copy(data)
            for mut in old_m:
                this_data[mut,old_c] = np.random.permutation(this_data[mut,old_c])

        elif permuted_conditions:

            this_data = np.swapaxes(copy.copy(data))
            for cond in old_c:
                this_data[cond,old_m] = np.random.permutation(this_data[cond,old_m])
            this_data = np.swapaxes(this_data,0,1)

        else:
            this_data = copy.copy(data)


        both_old = this_data[np.repeat(old_m,len(old_c)),np.tile(old_c,len(old_m))].reshape(len(old_m),len(old_c))

        U2, s2, V2 = np.linalg.svd(both_old)
        mut_new = this_data[np.repeat(new_m,len(old_c)),np.tile(old_c,len(new_m))].reshape(len(new_m),len(old_c))                    
        cond_new = this_data[np.repeat(old_m,len(new_c)),np.tile(new_c,len(old_m))].reshape(len(old_m),len(new_c))
        both_new = this_data[np.repeat(new_m,len(new_c)),np.tile(new_c,len(new_m))].reshape(len(new_m),len(new_c))

        for rank in range(1,max_rank+1):

            new_s = np.asarray(list(s2[:rank]) + list(np.zeros(s2[rank:].shape)))
            S2 = np.zeros((U2.shape[0],V2.shape[0]))
            S2[:min([U2.shape[0],V2.shape[0]]),:min([U2.shape[0],V2.shape[0]])] = np.diag(new_s)

            D_hat = np.dot(U2[:,:rank],np.dot(S2,V2)[:rank,:])
            A_hat = np.dot(mut_new,np.dot(np.linalg.pinv(D_hat),cond_new))
            if mse:
                rank_fit.append(np.sum(np.square(both_new-A_hat)))

            else:
                rank_fit.append(var_explained(both_new,A_hat)[0])

            fold_fits_by_condition[f].append([])
            fold_fits_by_mutant[f].append([])
            
            for k in range(len(new_c)):
                if mse:
                    fold_fits_by_condition[f][rank-1].append(np.sum(np.square(both_new[:,k]-A_hat[:,k])))
                else:
                    fold_fits_by_condition[f][rank-1].append(var_explained(both_new[:,k],A_hat[:,k])[0])
            for j in range(len(new_m)):
                if mse:
                    fold_fits_by_mutant[f][rank-1].append(np.sum(np.square(both_new[j,:]-A_hat[j,:])))
                else:
                    fold_fits_by_mutant[f][rank-1].append(var_explained(both_new[j,:],A_hat[j,:])[0])


        all_folds = all_folds + rank_fit
        fold_fits.append(rank_fit)
        
    return all_folds, fold_fits, fold_fits_by_condition, fold_fits_by_mutant

def SVD_fits(data,mse=False):
    U, s, V = np.linalg.svd(data)
    fits = []
    model_vars = []
    bic_1 = []
    bic_2 = []
    bic_3 = []

    for rank in range(1,data.shape[1]+1):
#             print(err,rep,rank)
        this_s = np.asarray(list(s[:rank]) + list(np.zeros(s[rank:].shape)))
        S = np.diag(this_s)
        this_rank = np.dot(U[:,:rank],np.dot(S,V)[:rank,:])
        if mse:
            fits.append(np.sum(np.square(data-this_rank)))
            model_vars.append(np.sum(np.square(data-this_rank)))
        else:
            var_explain, model_var, total_var = var_explained(data,this_rank)
            fits.append(var_explain)
            model_vars.append(model_var)
        bic_1.append(bic1(data,this_rank,rank))
        bic_2.append(bic2(data,this_rank,rank))
        bic_3.append(bic3(data,this_rank,rank))
      
    return fits, model_vars, bic_1, bic_2, bic_3

def svd_noise_comparison_figure(ax,this_f,err,n_pulls,yscale='linear',permutation=False):
    
    U, s, V = np.linalg.svd(this_f)
    
    max_d = this_f.shape[1]

    if type(err) != np.float:
        error = err.flatten()
    else:
        error = [err for i in range(len(this_f.flatten()))]

    # SVD on error alone
    noise_s_list = []
    perm_s_list = []
    for i in range(n_pulls):
        this_set = np.asarray([np.random.normal(0,np.sqrt(error[i])) for i in range(len(this_f.flatten()))]).reshape(this_f.shape[0],this_f.shape[1])
        U, noise_s, V = np.linalg.svd(this_set)
        noise_s_list.append(noise_s)
        plt.plot(np.arange(1,max_d+1),np.square(noise_s)/np.sum(np.square(s)),color='gray',alpha=0.1)

    if permutation:
        for i in range(n_pulls):
            this_set = np.random.permutation(this_f.ravel()).reshape(this_f.shape[0],this_f.shape[1])
            U, perm_s, V = np.linalg.svd(this_set)
            perm_s_list.append(perm_s)
            plt.plot(np.arange(1,max_d+1),np.square(perm_s)/np.sum(np.square(s)),color='r',alpha=0.1)


    plt.plot(np.arange(1,max_d+1),np.square(s)/np.sum(np.square(s)),color='k',marker='o',alpha=0.8)
    
    # Mean empirical noise max
    mean_noise_max = np.mean(noise_s_list,axis=0)[0] 
    plt.axhline(mean_noise_max**2/np.sum(np.square(s)),linestyle=':',color = 'k',alpha=0.8)

    max_detected = np.where(s < noise_s[0])[0][0]
    max_s = s[max_detected-1]**2/np.sum(np.square(s))
    next_s = s[max_detected]**2/np.sum(np.square(s))
    print(max_s,next_s,next_s/max_s)
    
    plt.xlabel('Number of Components')
    plt.ylabel('Variance explained by component')
    
    
    plt.xticks(np.arange(0,max_d+1,1),np.arange(0,max_d+1,1))
    plt.xlim(0.5,max_d+0.5)
#     plt.tight_layout()
    
    plt.yscale(yscale)
        
#         ax.annotate(s='Detection\nThreshold',
#                     xy=(4.5,mean_noise_max**2/np.sum(np.square(s))),
#                     xytext=(1,0.002),
#                    arrowprops=dict(
#                        arrowstyle="-|>",
#                             fc="0.0", ec="none",linewidth=1.5,
# #                             patchB=el,
#                             connectionstyle="angle3,angleA=0,angleB=-90"))
        
#         ax.annotate(s='6th/5th',
#                     xy=(6.2,s[5]**2/np.sum(np.square(s))),
#                     xytext=(6.2,s[4]**2/np.sum(np.square(s))),
#                     arrowprops=dict(
#                        arrowstyle="|-|",
#                             fc="0.0", ec="none",linewidth=1.5,
# #                             patchB=el,
# #                             connectionstyle="angle3,angleA=0,angleB=-90")
#                    ))
    return ax
        
def make_folds(n_mutants,n_conditions,n_folds):

    mutant_permutation = np.random.permutation(n_mutants)
    new_mutants = [sorted(mutant_permutation[(i)*int(n_mutants/n_folds):(i+1)*int(n_mutants/n_folds)]) for i in range(n_folds)]
    condition_permtutation = np.random.permutation(n_conditions)
    new_conditions = [sorted(condition_permtutation[(i)*int(n_conditions/n_folds):(i+1)*int(n_conditions/n_folds)]) for i in range(n_folds)]

    folds = [(new_mutants[fold],new_conditions[fold]) for fold in range(n_folds)]
    
    return folds
        
        
def svd_cross_validation_figure(ax,this_f,err,folds,n_permutations=0,mse=False,show_folds=True):
    
    n_mutants = this_f.shape[0]
    n_conditions = this_f.shape[1]
    
    n_folds = len(folds)

    real_fits, all_fold_fits, by_condition = SVD_predictions(this_f,folds,n_mutants,n_conditions,n_folds,mse=mse)
    real_fits = real_fits/n_folds
    
    max_rank = len(real_fits)
    if show_folds:
        for fold in range(n_folds):
            plt.plot(range(1,max_rank+1),all_fold_fits[fold][0],color='gray',alpha=0.3)

    for perm in range(n_permutations):
        permuted_mutants = copy.copy(this_f)

        perm_fits = SVD_predictions(permuted_mutants,folds,n_mutants,n_conditions,n_folds,permuted_mutants=True,mse=mse)[0]
        perm_fits = perm_fits/n_folds

        plt.plot(range(1,max_rank+1),perm_fits,color='r',alpha=0.1)
#         plt.scatter(range(1,max_rank+1)[np.where(perm_fits==np.min(perm_fits))[0][0]],np.min(perm_fits),color='gray',alpha=0.5,linestyle='--')

    plt.plot(range(1,max_rank+1),real_fits,color='k',alpha=1.0)
    plt.scatter(range(1,max_rank+1)[np.where(real_fits==np.max(real_fits))[0][0]],np.max(real_fits),color='k',alpha=1.0)

#     plt.axvline(x=d_true,color='k',linestyle=':')
    plt.axhline(np.max(real_fits),color='k',linestyle=':')

    print(np.max(real_fits))
    
#     plt.title(f'{err_names[e]} ({err:.1})')
    plt.xticks(np.arange(0,max_rank+1,1),np.arange(0,max_rank+1,1))
    plt.xlim(0.5,n_conditions+0.5)

    return ax

def plot_mutant_components(ax,U,this_data,x_component,y_component,mutant_colorset):

    plt.title('Mutant Components')
    already_plotted = []
    for mut in range(U.shape[0]):
        plt.scatter(U[mut,x_component],U[mut,y_component],color=mutant_colorset[mut],alpha=0.3,
                    label=f"{this_data['gene'].values[mut]}" if this_data['gene'].values[mut] not in already_plotted else '_nolegend_')
        already_plotted.append(this_data['gene'].values[mut])
    plt.xlabel(f'Component {x_component+1}')
    plt.ylabel(f'Component {y+component+1}')
    plt.legend(bbox_to_anchor=(1.0,1.0),ncol=2)
    ax.set_aspect('equal','box')
    ax.grid(True, which='both')
    sns.despine(ax=ax, offset=0) 

    return ax

