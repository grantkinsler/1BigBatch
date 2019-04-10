import numpy as np
import pandas as p
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.spatial import distance
from scipy.stats.mstats import gmean
import copy



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
    
    k = float(rank)
    m = float(data.shape[0])
    n = float(data.shape[1])

    return np.log(np.linalg.norm(data-fit)**2)  + k*(m+n)/(m*n)*np.log(m*n/(m+n))

def bic2(data,fit,rank):
    
    k = float(rank)
    m = float(data.shape[0])
    n = float(data.shape[1])
    C = np.min([np.sqrt(m),np.sqrt(n)])

    return np.log(np.linalg.norm(data-fit)**2)  + k*(m+n)/(m*n)*np.log(C**2)

def bic3(data,fit,rank):
    
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

def SVD_condition_predictions(data,old_c,new_c,n_mutants,n_conditions,permuted_mutants=False,permuted_conditions=False,mse=False):
    
    """ Predicting new conditions from SVD fit on old conditions using ALL mutants """

    max_rank = int(len(old_c))

    mutants = [i for i in range(n_mutants)]


    both_old = this_data[np.repeat(n_mutants,len(old_c)),np.tile(old_c,n_mutants)].reshape(len(old_m),n_mutants)

    U2, s2, V2 = np.linalg.svd(both_old)
                   
    cond_new = this_data[np.repeat(mutants,len(new_c)),np.tile(new_c,n_mutants)].reshape(n_mutants,len(new_c))



    for fold in folds:
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
#         both_true = truth[np.repeat(new_m,len(new_c)),np.tile(new_c,len(new_m))].reshape(len(new_m),len(new_c))

        for rank in range(1,max_rank+1):

            new_s = np.asarray(list(s2[:rank]) + list(np.zeros(s2[rank:].shape)))
            S2 = np.zeros((U2.shape[0],V2.shape[0]))
            S2[:min([U2.shape[0],V2.shape[0]]),:min([U2.shape[0],V2.shape[0]])] = np.diag(new_s)

            D_hat = np.dot(U2[:,:rank],np.dot(S2,V2)[:rank,:])
            A_hat = np.dot(mut_new,np.dot(np.linalg.pinv(D_hat),cond_new))
            if mse:
                rank_fit.append(np.sum(np.square(both_new-A_hat)))
#                 true_fit.append(np.sum(np.square(both_true-A_hat)))
            else:
                rank_fit.append(var_explained(both_new,A_hat)[0])
#                 true_fit.append(var_explained(both_true,A_hat)[0])
        all_folds = all_folds + rank_fit
#         true_folds = true_folds + true_fit
        
    return all_folds


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
#     true_folds = np.zeros(max_rank)
    for fold in folds:
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
#         both_true = truth[np.repeat(new_m,len(new_c)),np.tile(new_c,len(new_m))].reshape(len(new_m),len(new_c))

        for rank in range(1,max_rank+1):

            new_s = np.asarray(list(s2[:rank]) + list(np.zeros(s2[rank:].shape)))
            S2 = np.zeros((U2.shape[0],V2.shape[0]))
            S2[:min([U2.shape[0],V2.shape[0]]),:min([U2.shape[0],V2.shape[0]])] = np.diag(new_s)

            D_hat = np.dot(U2[:,:rank],np.dot(S2,V2)[:rank,:])
            A_hat = np.dot(mut_new,np.dot(np.linalg.pinv(D_hat),cond_new))
            if mse:
                rank_fit.append(np.sum(np.square(both_new-A_hat)))
#                 true_fit.append(np.sum(np.square(both_true-A_hat)))
            else:
                rank_fit.append(var_explained(both_new,A_hat)[0])
#                 true_fit.append(var_explained(both_true,A_hat)[0])
        all_folds = all_folds + rank_fit
#         true_folds = true_folds + true_fit
        
    return all_folds


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
        
        
def svd_cross_validation_figure(ax,this_f,err,folds,n_permutations=0,mse=False):
    
    n_mutants = this_f.shape[0]
    n_conditions = this_f.shape[1]
    
    n_folds = len(folds)

    real_fits = SVD_predictions(this_f,folds,n_mutants,n_conditions,n_folds,mse=mse)
    real_fits = real_fits/n_folds
    
    max_rank = len(real_fits)
#         true_fits = true_fits/n_folds

    for perm in range(n_permutations):
        permuted_mutants = copy.copy(this_f)

        perm_fits = SVD_predictions(permuted_mutants,folds,n_mutants,n_conditions,n_folds,permuted_mutants=True,mse=mse)
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