import numpy as np
import pandas as p
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.spatial import distance
from scipy.stats.mstats import gmean
from multiprocessing import Pool
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import cdist, euclidean
from itertools import combinations
import itertools
import copy


renamed_conditions = {
                  # '19':'M3 NoAnc',
                  # '19':'M3 Batch 9',
                  # 'M3_Batch_3':'M3 Batch 1',
                  # 'M3_Batch_6':'M3 Batch 2',
                  # 'M3_Batch_13':'M3 Batch 3',
                  # 'M3_Batch_18':'M3 Batch 4',
                  # 'M3_Batch_20':'M3 Batch 5',
                  # 'M3_Batch_21':'M3 Batch 6',
                  # 'M3_Batch_23':'M3 Batch 7',
                  # '1BB_M3':'M3 Batch 8',
                  #  '19':'M3',
                  # 'M3_Batch_3':'M3',
                  # 'M3_Batch_6':'M3',
                  # 'M3_Batch_13':'M3',
                  # 'M3_Batch_18':'M3',
                  # 'M3_Batch_20':'M3',
                  # 'M3_Batch_21':'M3',
                  # 'M3_Batch_23':'M3',
                  # '1BB_M3':'M3',

                   '19':'EC',
                  'M3_Batch_3':'EC',
                  'M3_Batch_6':'EC',
                  'M3_Batch_13':'EC',
                  'M3_Batch_18':'EC',
                  'M3_Batch_20':'EC',
                  'M3_Batch_21':'EC',
                  'M3_Batch_23':'EC',
                  '1BB_M3':'EC',


                  # '1BB_Baffle':'Baffle(8)',
                    '1BB_Baffle':'Baffle',
                  '1BB_1.4%Gluc' :'1.4% Gluc',
                  '1BB_1.8%Gluc' :'1.8% Gluc',
                  '1BB_0.2MNaCl' :'0.2M NaCl',
                  '1BB_0.5MNaCl' :'0.5M NaCl',
                  '1BB_0.2MKCl' :'0.2M KCl',
                  '1BB_0.5MKCl' :'0.5M KCl',
                  # '1BB_8.5uMGdA' :'+ 8.5uM GdA(8)',
                    '1BB_8.5uMGdA' :'8.5uM GdA',
                  '1BB_17uMGdA' :'17uM GdA',
                  '1BB_2ugFlu' :'2ug Flu',
                  '1BB_0.5ugFlu' :'0.5ug Flu',
                  '1BB_1%Raf' :'1% Raf',
                  '1BB_0.5%Raf' :'0.5% Raf',
                  '1BB_1%Gly' :'1% Gly',
                  '1BB_1%EtOH' :'1% EtOH',
                  '1BB_SucRaf' :'1% Suc,Raf',
                  'Ferm_44hr_Transfer':'12hr Ferm',
                  'Ferm_40hr_Transfer':'8hr Ferm',
                  'Ferm_54hr_Transfer':'22hr Ferm',
                  'Ferm_50hr_Transfer':'18hr Ferm',
                  'Resp_24hr_Transfer':'1 Day',
                  'Resp_3Day_Transfer':'3 Day',
                  'Resp_4Day_Transfer':'4 Day',
                  'Resp_5Day_Transfer':'5 Day',
                  'Resp_6Day_Transfer':'6 Day',
                  'Resp_7Day_Transfer':'7 Day',
                  'DMSO':'DMSO',
                  # 'Geldanamycin8.5uM':'+ 8.5uM GdA(1)',
                  'Geldanamycin8.5uM':'8.5uM GdA',
                  '1.4%':'Baffle, 1.4% Gluc',
                  # '1.5%':'Baffle(10)',
                    '1.5%':'Baffle',
                  '1.6%':'Baffle, 1.6% Gluc',
                  '1.7%':'Baffle, 1.7% Gluc',
                  '1.8%':'Baffle, 1.8% Gluc',
                  '2.5%':'Baffle, 2.5% Gluc',
                  'Ben0.4':'Baffle, 0.4 Benomyl',
                  'Ben2':'Baffle, 2 Benomyl',
                  }

mutant_colorset = {'CYR1':'#cab2d6', # light purple
                 # 'Diploid':'#fb9a99', # light red
                 # 'Diploid + Chr11Amp':'#e31a1c', # dark red for adaptive diploids
                 # 'Diploid + Chr12Amp':'#e31a1c',
                 # 'Diploid + IRA1':'#e31a1c',
                 # 'Diploid + IRA2':'#e31a1c',
                 'Diploid':'#e31a1c',
                 'Diploid + Chr11Amp':'#fb9a99', # light red for adaptive diploids
                 'Diploid + Chr12Amp':'#fb9a99',
                 'Diploid + IRA1':'#fb9a99',
                 'Diploid + IRA2':'#fb9a99',
                 'Diploid_adaptive':'#a50f15',

                 # 'Diploid':'#e31a1c',
                 # 'Diploid + Chr11Amp':'#fb9a99', # light red for adaptive diploids
                 # 'Diploid + Chr12Amp':'#fb9a99',
                 # 'Diploid + IRA1':'#fb9a99',
                 # 'Diploid + IRA2':'#fb9a99',
                 # 'Diploid_adaptive':'#fb9a99',

                 'GPB1':'#b2df8a',  # light green
                 'GPB2':'#33a02c',  # dark green
                 # 'IRA1':'#1f78b4', # dark blue
                 # 'IRA2':'#a6cee3', # light blue
                 'IRA1_nonsense':'#1f78b4', # dark blue
                 'IRA1_missense':'#a6cee3', # dark blue
                 'IRA2':'gray', # light blue
                 'IRA1_other':'gray', # light blue
                 'NotSequenced':'gray',
                 'NotSequenced_adaptive':'gray',
                 'PDE2':'#ff7f00',  # dark orange
                 'RAS2':'#b15928', # brown
                 'TFS1':'#fdbf6f', # light orange
                 'SSK2':'gray',
                 'SCH9':'#6a3d9a', # dark purple for TOR mutants
                 'TOR1':'#6a3d9a',
                 'KOG1':'#6a3d9a', 
                 'other':'lightgray',
                 'other_adaptive':'darkgray',
                 'ExpNeutral':'k',}

# old_colorset = {condition:sns.color_palette('Accent',len(old_conditions.keys()))[i] for i,condition in enumerate(old_conditions.keys())}
# bigbatch_colorset = {condition:sns.color_palette('Paired',len(bigbatch_conditions.keys()))[i] for i,condition in enumerate(bigbatch_conditions.keys())}

condition_categories = {
                
                  '19':'M3',
                  'M3_Batch_3':'M3',
                  'M3_Batch_6':'M3',
                  'M3_Batch_13':'M3',
                  'M3_Batch_18':'M3',
                  'M3_Batch_20':'M3',
                  'M3_Batch_21':'M3',
                  'M3_Batch_23':'M3',
                  '1BB_M3':'M3',

                  '1BB_Baffle':'glucose',
                  '1.4%':'glucose',
                  '1.5%':'glucose',
                  '1.6%':'glucose',
                  '1.7%':'glucose',
                  '1.8%':'glucose',
                  '2.5%':'glucose',
                  '1BB_1.4%Gluc' :'glucose',
                  '1BB_1.8%Gluc' :'glucose',

                  '1BB_0.2MNaCl' :'salt',
                  '1BB_0.5MNaCl' :'salt',
                  '1BB_0.2MKCl' :'salt',
                  '1BB_0.5MKCl' :'salt',

                  '1BB_8.5uMGdA' :'drugs',
                  '1BB_17uMGdA' :'drugs',
                  '1BB_2ugFlu' :'drugs',
                  '1BB_0.5ugFlu' :'drugs',
                  'DMSO':'drugs',
                  'Geldanamycin8.5uM':'drugs',
                  'Ben0.4':'drugs',
                  'Ben2':'drugs',

                  '1BB_1%Raf' :'carbons',
                  '1BB_0.5%Raf' :'carbons',
                  '1BB_1%Gly' :'carbons',
                  '1BB_1%EtOH' :'carbons',
                  '1BB_SucRaf' :'carbons',

                  'Ferm_44hr_Transfer':'ferm',
                  'Ferm_40hr_Transfer':'ferm',
                  'Ferm_54hr_Transfer':'ferm',
                  'Ferm_50hr_Transfer':'ferm',

                  'Resp_24hr_Transfer':'resp/stat',
                  'Resp_3Day_Transfer':'resp/stat',
                  'Resp_4Day_Transfer':'resp/stat',
                  'Resp_5Day_Transfer':'resp/stat',
                  'Resp_6Day_Transfer':'resp/stat',
                  'Resp_7Day_Transfer':'resp/stat',

                  }


# condition_colorset = {'13': (0.9921568627450981, 0.7529411764705882, 0.5254901960784314),
#  '18': (1.0, 1.0, 0.6),
#  '1BB_0.2MKCl': (0.8901960784313725, 0.10196078431372549, 0.10980392156862745),
#  '1BB_0.2MNaCl': (0.984313725490196, 0.6039215686274509, 0.6),
#  '1BB_0.5%Raf': (1.0, 0.4980392156862745, 0.0),
#  '1BB_1%Raf': (0.0, 0.0, 0.0),
#  '1BB_0.5MKCl': (0.9921568627450981, 0.7490196078431373, 0.43529411764705883),
#  '1BB_1%Gly': (0.792156862745098, 0.6980392156862745, 0.8392156862745098),
#  '1BB_1.4%Gluc': (0.6980392156862745, 0.8745098039215686, 0.5411764705882353),
#  '1BB_1.8%Gluc': (0.2, 0.6274509803921569, 0.17254901960784313),
#  '1BB_Baffle': (0.12156862745098039, 0.47058823529411764, 0.7058823529411765),
#  '1BB_M3': (0.6509803921568628, 0.807843137254902, 0.8901960784313725),
#  '20': (0.2196078431372549, 0.4235294117647059, 0.6901960784313725),
#  '21': (0.9411764705882353, 0.00784313725490196, 0.4980392156862745),
#  '23': (0.7490196078431373, 0.3568627450980392, 0.09019607843137253),
#  '3': (0.4980392156862745, 0.788235294117647, 0.4980392156862745),
#  '6': (0.7450980392156863, 0.6823529411764706, 0.8313725490196079)}

condition_colorset = {
 '1.4%': (0.8503344867358708, 0.14686658977316416, 0.13633217993079583),
 '1.5%': (0.9874509803921568, 0.5411764705882353, 0.41568627450980394),
 '1.6%': (0.9466666666666667, 0.26823529411764707, 0.19607843137254902),
 '1.7%': (0.9835755478662053, 0.4127950788158401, 0.28835063437139563),
 '1.8%': (0.7364705882352941, 0.08, 0.10117647058823528),
 '19': (0.8870588235294118, 0.8870588235294118, 0.8870588235294118),
 '1BB_0.2MKCl': (0.9921568627450981, 0.6564705882352941, 0.3827450980392157),
 '1BB_0.2MNaCl': (0.9937254901960785, 0.8501960784313726, 0.7043137254901961),
 '1BB_0.5%Raf': (0.681045751633987, 0.8718954248366013, 0.6562091503267974),
 '1BB_0.5MKCl': (0.9545098039215686, 0.44, 0.10666666666666666),
 '1BB_0.5MNaCl': (0.7709803921568628,0.2541176470588235,0.007058823529411764),
 '1BB_0.5ugFlu': (0.21568627450980393, 0.5294117647058824, 0.7542483660130719),
 '1BB_1%EtOH': (0.21568627450980393, 0.6287581699346405, 0.3333333333333333),
 '1BB_1%Gly': (0.8606689734717416, 0.9458362168396771, 0.8385697808535179),
 '1BB_1%Raf': (0.45176470588235296, 0.7670895809304115, 0.4612072279892349),
 '1BB_1.4%Gluc': (0.9969242599000384, 0.8961937716262975, 0.8489042675893886),
 '1BB_1.8%Gluc': (0.9913725490196079, 0.7913725490196079, 0.7082352941176471),
 '1BB_17uMGdA': (0.5105882352941177, 0.7323029603998462, 0.8588389081122645),
 '1BB_2ugFlu': (0.6718954248366014, 0.8143790849673203, 0.9006535947712418),
 '1BB_8.5uMGdA': (0.34646674356016915, 0.632402921953095, 0.8106728181468666),
 '1BB_Baffle': (0.9882352941176471, 0.6715417147251057, 0.5605382545174933),
 '1BB_M3': (0.25098039215686274, 0.25098039215686274, 0.25098039215686274),
 '1BB_SucRaf': (0.04359861591695502, 0.4648212226066897, 0.20369088811995384),
 '2.5%': (0.5946174548250673, 0.04613610149942329, 0.07558631295655516),
 'Ben0.4': (0.10557477893118032, 0.41262591311034214, 0.6859669357939254),
 'Ben2': (0.03137254901960784, 0.301914648212226, 0.588404459823145),
 'DMSO': (0.7993540945790081, 0.8740792003075739, 0.944882737408689),
 'Ferm_40hr_Transfer': (0.7019607843137254,0.803921568627451,0.8901960784313725),
 'Ferm_44hr_Transfer': (0.984313725490196,0.7058823529411765,0.6823529411764706),
 'Ferm_50hr_Transfer': (0.8, 0.9215686274509803, 0.7725490196078432),
 'Ferm_54hr_Transfer': (0.8705882352941177,0.796078431372549,0.8941176470588236),
 'Geldanamycin8.5uM': (0.8825067281814687,0.929196462898885,0.9724413687043445),
 'M3_Batch_13': (0.47843137254901963,0.47843137254901963,0.47843137254901963),
 'M3_Batch_18': (0.8087812379853903, 0.8087812379853903, 0.8087812379853903),
 'M3_Batch_20': (0.7105882352941176, 0.7105882352941176, 0.7105882352941176),
 'M3_Batch_21': (0.11380238369857748,0.11380238369857748,0.11380238369857748),
 'M3_Batch_23': (0.9538638985005767, 0.9538638985005767, 0.9538638985005767),
 'M3_Batch_3': (0.586082276047674, 0.586082276047674, 0.586082276047674),
 'M3_Batch_6': (0.3713033448673587, 0.3713033448673587, 0.3713033448673587),
 'Resp_24hr_Transfer': (0.8207612456747405,0.8218992695117262,0.9044982698961938),
 'Resp_3Day_Transfer': (0.9265974625144175,0.919769319492503,0.9557093425605536),
 'Resp_4Day_Transfer': (0.6878892733564014,0.6835832372164552,0.829834678969627),
 'Resp_5Day_Transfer': (0.5513264129181085,0.537916186082276,0.7524490580545944),
 'Resp_6Day_Transfer': (0.440722798923491,0.36772010765090346,0.6653902345251825),
 'Resp_7Day_Transfer': (0.3405767012687428,0.1742560553633218,0.5709342560553633)}



def flatten(list2d):
    return list(itertools.chain.from_iterable(list2d))


def jitter_point(mean,std=0.15):
    return np.random.normal(mean,std)

def inverse_variance_mean(means,standard_devs):

    variances = standard_devs

    mean = np.sum(means/variances)/(np.sum(1/variances))

    variance = (np.sum(1/variances))**(-1)

    return mean, variance


def downsample_single(set1,target):
    
    down = np.zeros((set1.shape[0],set1.shape[1]))
    for col in range(set1.shape[1]):
    
        total1 = np.sum(set1[:,col])
    
        if total1 < target:
            down[:,col] = set1[:,col]
        else:
            down[:,col] = np.random.multinomial(target,set1[:,col]/total1,1)

    return down
    
def downsample(set1,set2,target='default'):
    
    total1 = np.sum(set1)
    total2 = np.sum(set2)
    
    if target == 'default':
        target = min([total1,total2])
    
    if total1 == target:
        down1 = set1
    else:
        down1 = np.random.multinomial(target,set1/total1,1)
    
    if total2 == target:
        down2 = set2
    else:
        down2 = np.random.multinomial(target,set2/total2,1)
    
    return down1, down2

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

    error = error**2 # the error passed in is the standard deviation
        
    log2pi_variance = np.log(np.sqrt(2.0 * np.pi * error))
    inverse_variance = 1.0/error
    log_likelihood = -log2pi_variance - 0.5 * inverse_variance * np.square(fitness - est_fitness)

    return log_likelihood.sum()

def log_likelihood(est_fitness,fitness,error,metric='euclidean'):    

    error = error**2 # the error passed in is the standard deviation
        
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


def count_matrix(types,data,exceptions={'adaptive_other':1,'Diploid_adaptive':1}):
    
    counts = np.unique(np.asarray(types),return_counts=True)

    like_type_count_dict = {mut_type:count for mut_type,count in zip(*counts)}

    like_type_counts = [like_type_count_dict[mut_type] if mut_type not in exceptions.keys() else exceptions[mut_type] for mut_type in types]
    
    inv_like_type_counts = np.reciprocal(like_type_counts,dtype=np.float)

    if len(data.shape) > 1:
        like_type_counts = np.repeat(like_type_counts,data.shape[1]).reshape(len(like_type_counts),data.shape[1])
        inv_like_type_counts = np.repeat(inv_like_type_counts,data.shape[1]).reshape(len(inv_like_type_counts),data.shape[1])
    
    return like_type_counts,inv_like_type_counts

def var_explained_weighted_by_type(data,model,types,exceptions={'adaptive_other':1,'Diploid_adaptive':1},weighted=True):

    if weighted:
        counts,inv_counts = count_matrix(types,data,exceptions)

        ss_res = np.sum(((data-model)**2)/counts)
        ss_tot = np.sum(((data-np.mean(data))**2)/counts)

        return 1 - ss_res/ss_tot, ss_res, ss_tot
    else:
        return var_explained(data,model)


def sum_squared_error(data,model):

    return np.sum(np.square(data-model))

def max_variance_explainable(data,test_cols,n_samples=1000,r2=False):

    test_errs = [col.replace('fitness','error') for col in test_cols]

    resampled_data = [np.random.normal(data[test_cols].values,data[test_errs].values) for s in range(n_samples)]

    if r2 == True:
        r2s = [var_explained(data[test_cols].values,resampled_data[s])[0] for s in range(n_samples)]

        return np.mean(r2s)
    else:
        sum_squares = [sum_squared_error(data[test_cols].values,resampled_data[s]) for s in range(n_samples)]
        
        return np.mean(sum_squares)



def geometric_median(X, eps=1e-5):
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if euclidean(y, y1) < eps:
            return y1

        y = y1

def centroid(arr):
    
    return np.mean(arr,axis=0)


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

def SVD_predictions_train_test(data,train,test,permuted_mutants=False,permuted_conditions=False,mse=False,by_condition=False,by_mutant=False,error=None,likelihood=False):
    
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

    train_c = train[0]
    train_m = train[1]

    test_c = test[0]
    test_m = test[1]

    this_data = data



    assert len(train_m) + len(test_m) == this_data.shape[0]
    assert len(train_c) + len(test_c) == this_data.shape[1]

    max_rank = min([len(train_c),len(train_m)])

    fits_by_condition = []
    fits_by_mutant = []
    mean_fits =[]

    if permuted_mutants and permuted_conditions:
        this_data = copy.copy(data)
        this_data[train_m,train_c] = np.random.permutation(this_data[train_m,train_c].ravel()).reshape(len(train_m),len(train_c))
        subset = this_data[np.repeat(train_m,len(train_c)),np.tile(train_c,len(train_m))].ravel()

    elif permuted_mutants:
        this_data = copy.copy(data)
        for mut in train_m:
            this_data[mut,train_c] = np.random.permutation(this_data[mut,train_c])

    elif permuted_conditions:

        this_data = np.swapaxes(copy.copy(data),0,1)
        for cond in train_c:
            this_data[cond,train_m] = np.random.permutation(this_data[cond,train_m])
        this_data = np.swapaxes(this_data,0,1)

    else:
        this_data = copy.copy(data)


    both_old = this_data[np.repeat(train_m,len(train_c)),np.tile(train_c,len(train_m))].reshape(len(train_m),len(train_c))

    U2, s2, V2 = np.linalg.svd(both_old)

    mut_new = this_data[np.repeat(test_m,len(train_c)),np.tile(train_c,len(test_m))].reshape(len(test_m),len(train_c))  
    cond_new = this_data[np.repeat(train_m,len(test_c)),np.tile(test_c,len(train_m))].reshape(len(train_m),len(test_c))
    both_new = this_data[np.repeat(test_m,len(test_c)),np.tile(test_c,len(test_m))].reshape(len(test_m),len(test_c))

    if likelihood:
        both_new_error = error[np.repeat(test_m,len(test_c)),np.tile(test_c,len(test_m))].reshape(len(test_m),len(test_c))

    mean_mutant_prediction = np.repeat(np.mean(mut_new,axis=1),len(test_c)).reshape(len(test_m),len(test_c))

    if mse:
        mean_fits = np.sum(np.square(both_new-mean_mutant_prediction))
    elif likelihood:
        mean_fits = log_likelihood(mean_mutant_prediction,both_new,both_new_error)
    else: 
        mean_fits = var_explained(both_new,mean_mutant_prediction)[0]

    mean_fits_by_condition = []

    if by_condition:
        for k in range(len(test_c)):
            if mse:
                mean_fits_by_condition.append(np.sum(np.square(both_new[:,k]-mean_mutant_prediction[:,k])))
            elif likelihood:
                mean_fits_by_condition.append(log_likelihood(mean_mutant_prediction[:,k],both_new[:,k],both_new_error[:,k]))
            else:
                mean_fits_by_condition.append(var_explained(both_new[:,k],mean_mutant_prediction[:,k])[0])


    fit_by_rank = []
    guesses = []
    dhats = []
    for rank in range(1,max_rank+1):

        new_s = np.asarray(list(s2[:rank]) + list(np.zeros(s2[rank:].shape)))
        S2 = np.zeros((U2.shape[0],V2.shape[0]))
        S2[:min([U2.shape[0],V2.shape[0]]),:min([U2.shape[0],V2.shape[0]])] = np.diag(new_s)

        D_hat = np.dot(U2[:,:rank],np.dot(S2,V2)[:rank,:])
        A_hat = np.dot(mut_new,np.dot(np.linalg.pinv(D_hat),cond_new))

        dhats.append(D_hat)
        # print(rank,var_explained(both_old,D_hat)[0])

        guesses.append(A_hat)
        if mse:
            fit_by_rank.append(np.sum(np.square(both_new-A_hat)))

        else:
            fit_by_rank.append(var_explained(both_new,A_hat)[0])

        fits_by_condition.append([])
        fits_by_mutant.append([])

        if by_condition:
            for k in range(len(test_c)):
                if mse:
                    fits_by_condition[rank-1].append(np.sum(np.square(both_new[:,k]-A_hat[:,k])))
                elif likelihood:
                    fits_by_condition[rank-1].append(log_likelihood(mean_mutant_prediction[:,k],both_new[:,k],both_new_error[:,k]))
                else:
                    fits_by_condition[rank-1].append(var_explained(both_new[:,k],A_hat[:,k])[0])
        if by_mutant: 
            for j in range(len(test_m)):
                if mse:
                    fits_by_mutant[rank-1].append(np.sum(np.square(both_new[j,:]-A_hat[j,:])))
                elif likelihood:
                    fits_by_mutant[rank-1].append(log_likelihood(mean_mutant_prediction[j,:],both_new[j,:],both_new_error[j,:]))
                else:
                    fits_by_mutant[rank-1].append(var_explained(both_new[j,:],A_hat[j,:])[0])

        
    return fit_by_rank, fits_by_condition, fits_by_mutant, mean_fits, mean_fits_by_condition, guesses, dhats, both_old

def SVD_predictions_train_test_mixnmatch(data,train,test,n_components_per_set,by_condition=False,permuted_mutants=False,permuted_conditions=False,mse=False,by_mutant=False,likelihood=False,error=None):
    
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

    train_c = train[0]
    train_m = train[1]

    test_c = test[0]
    test_m = test[1]

    this_data = data

    if likelihood:
        both_new_error = error[np.repeat(test_m,len(test_c)),np.tile(test_c,len(test_m))].reshape(len(test_m),len(test_c))

    assert len(train_m) + len(test_m) == this_data.shape[0]
    assert len(train_c) + len(test_c) == this_data.shape[1]

    max_rank = min([len(train_c),len(train_m)])

    assert n_components_per_set <= max_rank

    fits_by_condition = []
    fits_by_mutant = []
    mean_fits =[]

    if permuted_mutants and permuted_conditions:
        this_data = copy.copy(data)
        this_data[train_m,train_c] = np.random.permutation(this_data[train_m,train_c].ravel()).reshape(len(train_m),len(train_c))
        subset = this_data[np.repeat(train_m,len(train_c)),np.tile(train_c,len(train_m))].ravel()

    elif permuted_mutants:
        this_data = copy.copy(data)
        for mut in train_m:
            this_data[mut,train_c] = np.random.permutation(this_data[mut,train_c])

    elif permuted_conditions:

        this_data = np.swapaxes(copy.copy(data))
        for cond in old_c:
            this_data[cond,train_m] = np.random.permutation(this_data[cond,train_m])
        this_data = np.swapaxes(this_data,0,1)

    else:
        this_data = copy.copy(data)


    both_old = this_data[np.repeat(train_m,len(train_c)),np.tile(train_c,len(train_m))].reshape(len(train_m),len(train_c))

    U2, s2, V2 = np.linalg.svd(both_old)

    mut_new = this_data[np.repeat(test_m,len(train_c)),np.tile(train_c,len(test_m))].reshape(len(test_m),len(train_c))  
    cond_new = this_data[np.repeat(train_m,len(test_c)),np.tile(test_c,len(train_m))].reshape(len(train_m),len(test_c))
    both_new = this_data[np.repeat(test_m,len(test_c)),np.tile(test_c,len(test_m))].reshape(len(test_m),len(test_c))

    mean_mutant_prediction = np.repeat(np.mean(mut_new,axis=1),len(test_c)).reshape(len(test_m),len(test_c))

    if mse:
        mean_fits = np.sum(np.square(both_new-mean_mutant_prediction))
    elif likelihood:
        mean_fits = log_likelihood(mean_mutant_prediction,both_new,both_new_error)
    else: 
        mean_fits = var_explained(both_new,mean_mutant_prediction)[0]

    fit_by_rank = []

    component_sets = []

    guesses = []


    for cs,component_set in enumerate(combinations(range(max_rank),n_components_per_set)):
        component_set = list(component_set)

        component_sets.append(component_set)

        # new_s = np.asarray(list(s2[component_set]) + list(np.zeros(len(s2)-len(component_set))))
        new_s = np.asarray([s2[s] if s in component_set else 0 for s in range(len(s2))]  )
        # print(new_s)
        S2 = np.zeros((U2.shape[0],V2.shape[0]))
        S2[:min([U2.shape[0],V2.shape[0]]),:min([U2.shape[0],V2.shape[0]])] = np.diag(new_s)

        D_hat = np.dot(U2[:,component_set],np.dot(S2,V2)[component_set,:])
        A_hat = np.dot(mut_new,np.dot(np.linalg.pinv(D_hat),cond_new))

        guesses.append(A_hat)
        if mse:
            fit_by_rank.append(np.sum(np.square(both_new-A_hat)))
        elif likelihood:
            fit_by_rank.append(log_likelihood(A_hat,both_new,both_new_error))
        else:
            fit_by_rank.append(var_explained(both_new,A_hat)[0])

        fits_by_condition.append([])
        fits_by_mutant.append([])

        if by_condition:
            for k in range(len(test_c)):
                if mse:
                    fits_by_condition[cs].append(np.sum(np.square(both_new[:,k]-A_hat[:,k])))
                else:
                    fits_by_condition[cs].append(var_explained(both_new[:,k],A_hat[:,k])[0])
        if by_mutant:
            for j in range(len(test_m)):
                if mse:
                    fits_by_mutant[cs].append(np.sum(np.square(both_new[j,:]-A_hat[j,:])))
                else:
                    fits_by_mutant[cs].append(var_explained(both_new[j,:],A_hat[j,:])[0])

        
    return fit_by_rank, fits_by_condition, fits_by_mutant, mean_fits, component_sets, guesses

def SVD_locations(this_data,train_cols,test_cols,training_bcs,testing_bcs,model,by_condition=False,permuted_mutants=False,permuted_conditions=False,mse=False,by_mutant=False):
    
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

    fitness_cols = np.concatenate((train_cols,test_cols))

    fitness = this_data[fitness_cols].values

    n_mutants = fitness.shape[0]
    n_conditions = fitness.shape[1]

    train_conditions = train_cols
    train_locs = np.where(np.isin(fitness_cols,train_conditions))[0]

    test_conditions = test_cols
    test_locs = np.where(np.isin(fitness_cols,test_conditions))[0]

    all_locs = sorted(list(train_locs)+list(test_locs))

    used_mutants = [bc for bc in this_data['barcode'].values if bc in (list(training_bcs) + list(testing_bcs))]

    all_mut_locs =sorted(list(np.where(np.isin(this_data['barcode'].values,used_mutants))[0]))

    this_data = fitness[all_mut_locs,:][:,all_locs]
    new_train_locs = np.where(np.isin(all_locs,train_locs))[0]
    new_test_locs = np.where(np.isin(all_locs,test_locs))[0]  

    train = [new_train_locs,np.where(np.isin(used_mutants,training_bcs))[0]]
    test = [new_test_locs,np.where(np.isin(used_mutants,testing_bcs))[0]]

    train_c = train[0]
    train_m = train[1]

    test_c = test[0]
    test_m = test[1]

    component_set = range(model)

    assert len(train_m) + len(test_m) == this_data.shape[0]
    assert len(train_c) + len(test_c) == this_data.shape[1]

    max_rank = min([len(train_c),len(train_m)])

    fits_by_condition = []
    fits_by_mutant = []
    mean_fits =[]

    both_old = this_data[np.repeat(train_m,len(train_c)),np.tile(train_c,len(train_m))].reshape(len(train_m),len(train_c))

    U2, s2, V2 = np.linalg.svd(both_old)

    mut_new = this_data[np.repeat(test_m,len(train_c)),np.tile(train_c,len(test_m))].reshape(len(test_m),len(train_c))  
    cond_new = this_data[np.repeat(train_m,len(test_c)),np.tile(test_c,len(train_m))].reshape(len(train_m),len(test_c))
    both_new = this_data[np.repeat(test_m,len(test_c)),np.tile(test_c,len(test_m))].reshape(len(test_m),len(test_c))

    mean_mutant_prediction = np.repeat(np.mean(mut_new,axis=1),len(test_c)).reshape(len(test_m),len(test_c))

    if mse:
        mean_fits = np.sum(np.square(both_new-mean_mutant_prediction))
    else: 
        mean_fits = var_explained(both_new,mean_mutant_prediction)[0]

    fit_by_rank = []

    guesses = []

    # new_s = np.asarray(list(s2[component_set]) + list(np.zeros(len(s2)-len(component_set))))
    new_s = np.asarray([s2[s] if s in component_set else 0 for s in range(len(s2))]  )
    # print(new_s)
    S2 = np.zeros((U2.shape[0],V2.shape[0]))
    S2[:min([U2.shape[0],V2.shape[0]]),:min([U2.shape[0],V2.shape[0]])] = np.diag(new_s)

    train_mutant_locs = U2[:,component_set]
    train_condition_locs = np.dot(S2,V2)[component_set,:]

    D_hat = np.dot(U2[:,component_set],np.dot(S2,V2)[component_set,:])

    ## do least squares linear regression explicitly to get the locations.G

    reg_test_condition = LinearRegression(fit_intercept=False).fit(train_mutant_locs,cond_new)
    test_condition_locs = reg_test_condition.coef_

    reg_test_mutant = LinearRegression(fit_intercept=False).fit(train_condition_locs.swapaxes(0,1),mut_new.swapaxes(0,1))
    test_mutant_locs = reg_test_mutant.coef_.swapaxes(0,1)

    A_hat = np.dot(mut_new,np.dot(np.linalg.pinv(D_hat),cond_new))

    lin_reg_A_hat = np.dot(test_mutant_locs.swapaxes(0,1),test_condition_locs.swapaxes(0,1))

    print(len(component_set),var_explained(both_new,A_hat)[0])

    assert np.all(np.isclose(A_hat,lin_reg_A_hat)) # verify that the prediction we get is the same both ways

        
    return var_explained(both_new,A_hat)[0], train_mutant_locs, test_mutant_locs.swapaxes(0,1), train_condition_locs.swapaxes(0,1), test_condition_locs


def SVD_predictions_train_test_weighted(data,train,test,
                                        permuted_mutants=False,permuted_conditions=False,
                                        mse=False,by_condition=False,by_mutant=False,error=None,likelihood=False,
                                        weighted_by_type=True,types=None):
    
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

    train_c = train[0]
    train_m = train[1]

    test_c = test[0]
    test_m = test[1]

    this_data = data
    
    assert len(train_m) + len(test_m) == this_data.shape[0]
    assert len(train_c) + len(test_c) == this_data.shape[1]

    max_rank = min([len(train_c),len(train_m)])

    fits_by_condition = []
    fits_by_mutant = []
    mean_fits =[]

    if permuted_mutants and permuted_conditions:
        this_data = copy.copy(data)
        this_data[train_m,train_c] = np.random.permutation(this_data[train_m,train_c].ravel()).reshape(len(train_m),len(train_c))
        subset = this_data[np.repeat(train_m,len(train_c)),np.tile(train_c,len(train_m))].ravel()

    elif permuted_mutants:
        this_data = copy.copy(data)
        for mut in train_m:
            this_data[mut,train_c] = np.random.permutation(this_data[mut,train_c])

    elif permuted_conditions:

        this_data = np.swapaxes(copy.copy(data),0,1)
        for cond in train_c:
            this_data[cond,train_m] = np.random.permutation(this_data[cond,train_m])
        this_data = np.swapaxes(this_data,0,1)

    else:
        this_data = copy.copy(data)
        
    if weighted_by_type:
        train_count,train_inv_count = count_matrix(types[train_m],this_data[train_m,:])
        test_count,test_inv_count = count_matrix(types[test_m],this_data[test_m,:])
        
        order = np.argsort(np.concatenate((train_m,test_m)))
        
        all_count = np.concatenate((train_count,test_count))[order]
        all_inv_count = np.concatenate((train_inv_count,test_inv_count))[order]
        
        weighted_data = np.multiply(all_inv_count,this_data)
    else:
        train_count = np.ones(this_data[train_m,:].shape)
        train_inv_count = np.ones(this_data[train_m,:].shape)
        test_count = np.ones(this_data[test_m,:].shape)
        test_inv_count = np.ones(this_data[test_m,:].shape)
        
        order = np.argsort(np.concatenate((train_m,test_m)))
        
        all_count = np.concatenate((train_count,test_count))[order]
        all_inv_count = np.concatenate((train_inv_count,test_inv_count))[order]
        
        weighted_data = copy.copy(this_data)
        
    both_old = weighted_data[np.repeat(train_m,len(train_c)),np.tile(train_c,len(train_m))].reshape(len(train_m),len(train_c))
    both_old_unweighted = this_data[np.repeat(train_m,len(train_c)),np.tile(train_c,len(train_m))].reshape(len(train_m),len(train_c))

    U2, s2, V2 = np.linalg.svd(both_old)
    
    mut_new = weighted_data[np.repeat(test_m,len(train_c)),np.tile(train_c,len(test_m))].reshape(len(test_m),len(train_c))  
    cond_new = weighted_data[np.repeat(train_m,len(test_c)),np.tile(test_c,len(train_m))].reshape(len(train_m),len(test_c))
    both_new = weighted_data[np.repeat(test_m,len(test_c)),np.tile(test_c,len(test_m))].reshape(len(test_m),len(test_c))
    
    both_new_unweighted = this_data[np.repeat(test_m,len(test_c)),np.tile(test_c,len(test_m))].reshape(len(test_m),len(test_c))

    if likelihood:
        both_new_error = error[np.repeat(test_m,len(test_c)),np.tile(test_c,len(test_m))].reshape(len(test_m),len(test_c))

    mean_mutant_prediction = np.repeat(np.mean(mut_new,axis=1),len(test_c)).reshape(len(test_m),len(test_c))

    if mse:
        mean_fits = np.sum(np.square(both_new-mean_mutant_prediction))
    elif likelihood:
        mean_fits = log_likelihood(mean_mutant_prediction,both_new,both_new_error)
    else: 
        mean_fits = tools.var_explained(both_new,mean_mutant_prediction)[0]

    mean_fits_by_condition = []

    if by_condition:
        for k in range(len(test_c)):
            if mse:
                mean_fits_by_condition.append(np.sum(np.square(both_new[:,k]-mean_mutant_prediction[:,k])))
            elif likelihood:
                mean_fits_by_condition.append(log_likelihood(mean_mutant_prediction[:,k],both_new[:,k],both_new_error[:,k]))
            else:
                mean_fits_by_condition.append(tools.var_explained(both_new[:,k],mean_mutant_prediction[:,k])[0])


    fit_by_rank = []
    guesses = []
    dhats = []
    for rank in range(1,max_rank+1):

        new_s = np.asarray(list(s2[:rank]) + list(np.zeros(s2[rank:].shape)))
        S2 = np.zeros((U2.shape[0],V2.shape[0]))
        S2[:min([U2.shape[0],V2.shape[0]]),:min([U2.shape[0],V2.shape[0]])] = np.diag(new_s)

        D_hat = train_count[:,:len(train_c)]*np.dot(U2[:,:rank],np.dot(S2,V2)[:rank,:])
        A_hat = test_count[:,:len(test_c)]*np.dot(mut_new,np.dot(np.linalg.pinv(D_hat),cond_new))

        dhats.append(D_hat)

        guesses.append(A_hat)
        if mse:
            fit_by_rank.append(np.sum(np.square(both_new_unweighted-A_hat)))

        else:
            fit_by_rank.append(tools.var_explained(both_new_unweighted,A_hat)[0])

        fits_by_condition.append([])
        fits_by_mutant.append([])

        if by_condition:
            for k in range(len(test_c)):
                if mse:
                    fits_by_condition[rank-1].append(np.sum(np.square(both_new[:,k]-A_hat[:,k])))
                elif likelihood:
                    fits_by_condition[rank-1].append(log_likelihood(mean_mutant_prediction[:,k],both_new[:,k],both_new_error[:,k]))
                else:
                    fits_by_condition[rank-1].append(tools.var_explained(both_new_unweighted[:,k],A_hat[:,k])[0])
        if by_mutant: 
            for j in range(len(test_m)):
                if mse:
                    fits_by_mutant[rank-1].append(np.sum(np.square(both_new[j,:]-A_hat[j,:])))
                elif likelihood:
                    fits_by_mutant[rank-1].append(log_likelihood(mean_mutant_prediction[j,:],both_new[j,:],both_new_error[j,:]))
                else:
                    fits_by_mutant[rank-1].append(tools.var_explained(both_new_unweighted[j,:],A_hat[j,:])[0])

        
    return fit_by_rank, fits_by_condition, fits_by_mutant, mean_fits, mean_fits_by_condition, guesses, dhats, both_old

def SVD_mixnmatch_locations(data,train,test,component_set,by_condition=False,permuted_mutants=False,permuted_conditions=False,mse=False,by_mutant=False):
    
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

    train_c = train[0]
    train_m = train[1]

    test_c = test[0]
    test_m = test[1]

    this_data = data

    assert len(train_m) + len(test_m) == this_data.shape[0]
    assert len(train_c) + len(test_c) == this_data.shape[1]

    max_rank = min([len(train_c),len(train_m)])

    fits_by_condition = []
    fits_by_mutant = []
    mean_fits =[]

    both_old = this_data[np.repeat(train_m,len(train_c)),np.tile(train_c,len(train_m))].reshape(len(train_m),len(train_c))

    U2, s2, V2 = np.linalg.svd(both_old)

    mut_new = this_data[np.repeat(test_m,len(train_c)),np.tile(train_c,len(test_m))].reshape(len(test_m),len(train_c))  
    cond_new = this_data[np.repeat(train_m,len(test_c)),np.tile(test_c,len(train_m))].reshape(len(train_m),len(test_c))
    both_new = this_data[np.repeat(test_m,len(test_c)),np.tile(test_c,len(test_m))].reshape(len(test_m),len(test_c))

    mean_mutant_prediction = np.repeat(np.mean(mut_new,axis=1),len(test_c)).reshape(len(test_m),len(test_c))

    if mse:
        mean_fits = np.sum(np.square(both_new-mean_mutant_prediction))
    else: 
        mean_fits = var_explained(both_new,mean_mutant_prediction)[0]

    fit_by_rank = []

    guesses = []

    # new_s = np.asarray(list(s2[component_set]) + list(np.zeros(len(s2)-len(component_set))))
    new_s = np.asarray([s2[s] if s in component_set else 0 for s in range(len(s2))]  )
    # print(new_s)
    S2 = np.zeros((U2.shape[0],V2.shape[0]))
    S2[:min([U2.shape[0],V2.shape[0]]),:min([U2.shape[0],V2.shape[0]])] = np.diag(new_s)

    train_mutant_locs = U2[:,component_set]
    # train_condition_locs = np.dot(S2,V2)[component_set,:]
    train_condition_locs = V2[component_set,:]

    D_hat = np.dot(U2[:,component_set],np.dot(S2,V2)[component_set,:])

    ## do least squares linear regression explicitly to get the locations.

    # to find best test condition locations, we fold the Sigma matrix into the training mutant locations
    reg_test_condition = LinearRegression(fit_intercept=False).fit(np.dot(U2,S2)[:,component_set],cond_new)
    test_condition_locs = reg_test_condition.coef_

    # to find best test mutant locations, we fold the Sigma matrix into the training condition locations
    reg_test_mutant = LinearRegression(fit_intercept=False).fit(np.dot(S2,V2)[component_set,:].swapaxes(0,1),mut_new.swapaxes(0,1))
    test_mutant_locs = reg_test_mutant.coef_.swapaxes(0,1)

    A_hat = np.dot(mut_new,np.dot(np.linalg.pinv(D_hat),cond_new))

    lin_reg_A_hat = np.dot(test_mutant_locs.swapaxes(0,1),np.dot(S2[:,component_set],test_condition_locs.swapaxes(0,1))[component_set,:])

    print(len(component_set),var_explained(both_new,A_hat)[0])

    assert np.all(np.isclose(A_hat,lin_reg_A_hat)) # verify that the prediction we get is the same both ways

        
    return var_explained(both_new,A_hat)[0], train_mutant_locs, test_mutant_locs.swapaxes(0,1), train_condition_locs.swapaxes(0,1), test_condition_locs, S2


def SVD_predictions(data,folds,n_mutants,n_conditions,n_folds,permuted_mutants=False,permuted_conditions=False,mse=False,by_condition=False,by_mutant=False):
    
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
    mean_fits =[]
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

        mean_mutant_prediction = np.repeat(np.mean(mut_new,axis=1),len(new_c)).reshape(len(new_m),len(new_c))

        if mse:
            mean_fits.append(np.sum(np.square(both_new-mean_mutant_prediction)))
        else: 
            mean_fits.append(var_explained(both_new,mean_mutant_prediction)[0])


        for rank in range(1,max_rank+1):

            new_s = np.asarray(list(s2[:rank]) + list(np.zeros(s2[rank:].shape)))
            S2 = np.zeros((U2.shape[0],V2.shape[0]))
            S2[:min([U2.shape[0],V2.shape[0]]),:min([U2.shape[0],V2.shape[0]])] = np.diag(new_s)

            D_hat = np.dot(U2[:,:rank],np.dot(S2,V2)[:rank,:])

            print(rank,var_explained(both_old,D_hat)[0])
            A_hat = np.dot(mut_new,np.dot(np.linalg.pinv(D_hat),cond_new))
            if mse:
                rank_fit.append(np.sum(np.square(both_new-A_hat)))

            else:
                rank_fit.append(var_explained(both_new,A_hat)[0])

            fold_fits_by_condition[f].append([])
            fold_fits_by_mutant[f].append([])
            if by_condition:
                for k in range(len(new_c)):
                    if mse:
                        fold_fits_by_condition[f][rank-1].append(np.sum(np.square(both_new[:,k]-A_hat[:,k])))
                    else:
                        fold_fits_by_condition[f][rank-1].append(var_explained(both_new[:,k],A_hat[:,k])[0])
            if by_mutant: 
                for j in range(len(new_m)):
                    if mse:
                        fold_fits_by_mutant[f][rank-1].append(np.sum(np.square(both_new[j,:]-A_hat[j,:])))
                    else:
                        fold_fits_by_mutant[f][rank-1].append(var_explained(both_new[j,:],A_hat[j,:])[0])


        all_folds = all_folds + rank_fit
        fold_fits.append(rank_fit)
        
    return all_folds, fold_fits, fold_fits_by_condition, fold_fits_by_mutant, mean_fits

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


def svd_noise_comparison_detection(this_f,err,n_pulls,permutation=False):
    
    U, s, V = np.linalg.svd(this_f)
    
    max_d = min(this_f.shape)

    if type(err) != np.float:
        error = err.flatten()
    else:
        error = [err for i in range(len(this_f.flatten()))]

    # SVD on error alone
    noise_s_list = []
    perm_s_list = []
    for i in range(n_pulls):
        # this_set = np.asarray([np.random.normal(0,np.sqrt(error[i])) for i in range(len(this_f.flatten()))]).reshape(this_f.shape[0],this_f.shape[1])
        this_set = np.asarray([np.random.normal(0,error[i]) for i in range(len(this_f.flatten()))]).reshape(this_f.shape[0],this_f.shape[1])

        U, noise_s, V = np.linalg.svd(this_set)
        noise_s_list.append(noise_s)

    if permutation:
        for i in range(n_pulls):
            this_set = np.random.permutation(this_f.ravel()).reshape(this_f.shape[0],this_f.shape[1])
            U, perm_s, V = np.linalg.svd(this_set)
            perm_s_list.append(perm_s)

    # print(np.mean(noise_s_list,axis=0))
    # print(s)
    # print(s**2/np.sum(np.square(s)))
    
    # Mean empirical noise max
    mean_noise_max = np.mean(noise_s_list,axis=0)[0] 

    max_detected = np.where(s < mean_noise_max)[0][0]
    # print(np.where(s < mean_noise_max))
    max_s = s[max_detected-1]**2/np.sum(np.square(s))
    next_s = s[max_detected]**2/np.sum(np.square(s))

    return max_detected

        
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

    real_fits, all_fold_fits, by_condition, fold_fits_by_mutant, mean_fits = SVD_predictions(this_f,folds,n_mutants,n_conditions,n_folds,mse=mse)
    real_fits = real_fits/n_folds
    
    max_rank = len(real_fits)

    if show_folds:
        for fold in range(n_folds):
            plt.plot(range(1,max_rank+1),all_fold_fits[fold],color='gray',alpha=0.3)

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
    plt.ylabel(f'Component {y_component+1}')
    plt.legend(bbox_to_anchor=(1.0,1.0),ncol=2)
    # ax.set_aspect('equal','box')
    ax.grid(True, which='both')
    sns.despine(ax=ax, offset=0) 

    return ax

def SVD_train_test_folds(this_data,cols_avail,mutants_avail,n_folds,fixed_mutant_sets=False,full_output=False,mse=True):

    best_guesses = {}
    all_guesses = []
    
    for iteration in range(n_folds):
        first_set =  sorted(np.random.choice(range(len(cols_avail)),int(len(cols_avail)/2),replace=False))
        second_set = [i for i in range(len(cols_avail)) if i not in first_set]

        if not fixed_mutant_sets:

          training_mutants = sorted(np.random.choice(mutants_avail,int(len(mutants_avail)/2),replace=False))
          testing_mutants = [i for i in mutants_avail if i not in training_mutants]
        else:
          training_mutants = mutants_avail[0]
          testing_mutants = mutants_avail[1]


        cols_avail = np.asarray(cols_avail)

        train_cols = list(cols_avail[first_set])
        test_cols = list(cols_avail[second_set])

        fitness_cols = train_cols + test_cols

        fitness = this_data[fitness_cols].values
        error = this_data[[col.replace('fitness','error') for col in fitness_cols]].values

        n_mutants = fitness.shape[0]
        n_conditions = fitness.shape[1]

        train_conditions = train_cols
        train_locs = np.where(np.isin(fitness_cols,train_conditions))[0]

        test_conditions = test_cols
        test_locs = np.where(np.isin(fitness_cols,test_conditions))[0]

        all_locs = sorted(list(train_locs)+list(test_locs))

        used_mutants = [bc for bc in this_data['barcode'].values if bc in (list(training_mutants) + list(testing_mutants))]

        all_mut_locs =sorted(list(np.where(np.isin(this_data['barcode'].values,used_mutants))[0]))

        this_fitness = fitness[all_mut_locs,:][:,all_locs]
        this_error = error[all_mut_locs,:][:,all_locs]
        new_train_locs = np.where(np.isin(all_locs,train_locs))[0]
        new_test_locs = np.where(np.isin(all_locs,test_locs))[0]  

        train = [new_train_locs,np.where(np.isin(used_mutants,training_mutants))[0]]
        test = [new_test_locs,np.where(np.isin(used_mutants,testing_mutants))[0]]

        max_rank = min([len(train[0]),len(train[1])])

        # output = SVD_predictions_train_test(this_fitness,train,test,by_condition=True,mse=True)
        output = SVD_predictions_train_test(this_fitness,train,test,by_condition=True,mse=mse,error=this_error)

        best_by_rank = output[0]
        all_guesses.append(best_by_rank)
        best_guesses[f'{iteration}'] = (min(best_by_rank),np.where(best_by_rank==min(best_by_rank))[0][0] +1)
    if not full_output:
      return best_guesses, all_guesses
    else:
      return best_guesses, all_guesses, output


def situate_data(this_data,train_cols,test_cols,training_bcs,testing_bcs,gene_list,fixed_mutant_sets=False):

    data_situation = {}

    data_situation['training_bcs'] = training_bcs
    data_situation['testing_bcs'] = testing_bcs
    data_situation['train_cols'] = train_cols
    data_situation['test_cols'] = test_cols

    ## CROSS VALIDATION STAGE ON TRAINING DATA
    cols_avail = train_cols
    if not fixed_mutant_sets:
        CV_best_guesses, CV_all_guesses = SVD_train_test_folds(this_data,cols_avail,training_bcs,1000)
    else:
        CV_best_guesses, CV_all_guesses = SVD_train_test_folds(this_data,cols_avail,[training_bcs,testing_bcs],1000,fixed_mutant_sets=True)

    CV_best_rank_index =  np.where(np.mean(CV_all_guesses,axis=0)==min(np.mean(CV_all_guesses,axis=0)))[0][0]

    data_situation['CV_best_guesses'] = CV_best_guesses
    data_situation['CV_all_guesses'] = CV_all_guesses
    data_situation['CV_best_rank_index'] = CV_best_rank_index


    ## PREDICTION STAGE 

    fitness_cols = train_cols + test_cols

    fitness = this_data[fitness_cols].values

    n_mutants = fitness.shape[0]
    n_conditions = fitness.shape[1]

    train_conditions = train_cols
    train_locs = np.where(np.isin(fitness_cols,train_conditions))[0]

    test_conditions = test_cols
    test_locs = np.where(np.isin(fitness_cols,test_conditions))[0]

    all_locs = sorted(list(train_locs)+list(test_locs))

    used_mutants = [bc for bc in this_data['barcode'].values if bc in (list(training_bcs) + list(testing_bcs))]

    all_mut_locs =sorted(list(np.where(np.isin(this_data['barcode'].values,used_mutants))[0]))

    this_fitness = fitness[all_mut_locs,:][:,all_locs]
    new_train_locs = np.where(np.isin(all_locs,train_locs))[0]
    new_test_locs = np.where(np.isin(all_locs,test_locs))[0]  

    train = [new_train_locs,np.where(np.isin(used_mutants,training_bcs))[0]]
    test = [new_test_locs,np.where(np.isin(used_mutants,testing_bcs))[0]]

    train_c = train[0]
    train_m = train[1]

    test_c = test[0]
    test_m = test[1]

    both_new = this_fitness[np.repeat(test_m,len(test_c)),np.tile(test_c,len(test_m))].reshape(len(test_m),len(test_c))

    all_m = sorted(list(train_m) + list(test_m))
    new_cond = this_fitness[np.repeat(all_m,len(test_c)),np.tile(test_c,len(all_m))].reshape(len(all_m),len(test_c))

    data_situation['this_fitness'] = this_fitness
    data_situation['this_error'] = this_data[[col.replace('fitness','error') for col in fitness_cols]].values[all_mut_locs,:][:,all_locs]
    data_situation['train'] = train
    data_situation['test'] = test
    data_situation['both_new'] = both_new

    fit_by_rank, fits_by_condition, fits_by_mutant, mean_fits,mean_fits_by_condition, guesses, dhats, both_old = SVD_predictions_train_test(this_fitness,train,test,by_condition=True,by_mutant=True)


    data_situation['fit_by_rank'] = fit_by_rank
    data_situation['fits_by_condition'] = fits_by_condition
    data_situation['fits_by_mutant'] = fits_by_mutant
    data_situation['mean_fits'] = mean_fits
    data_situation['mean_fits_by_condition']= mean_fits_by_condition
    data_situation['guesses'] = guesses
    data_situation['dhats'] = dhats
    data_situation['both_old'] = both_old
    data_situation['train_conditions'] = train_conditions
    data_situation['test_conditions'] = test_conditions

    max_rank = min([len(train[0]),len(train[1])])

    data_situation['mut_locs'] = []
    data_situation['cond_locs'] = []
    data_situation['distances'] = []
    data_situation['geom_medians'] = []
    data_situation['centroids'] = []
    data_situation['avg_pairwise'] = []
    data_situation['distance_from_median'] = []
    data_situation['distance_from_centroid'] = []
    data_situation['dimension_weights'] = []

    
    ### LOCATIONS
    for model in range(1,max_rank+1):
        fit, train_muts, test_muts, train_cond, test_cond, weights = SVD_mixnmatch_locations(this_fitness,train,test,range(model))
        all_muts = np.concatenate((train_muts,test_muts))
        all_muts = all_muts[np.argsort(np.concatenate((train[1],test[1])))]

        data_situation['dimension_weights'].append(weights)


        all_cond = np.concatenate((train_cond,test_cond))
        data_situation['cond_locs'].append(all_cond)

        data_situation['mut_locs'].append(all_muts)
        data_situation['distances'].append(distance.pdist(all_muts))

        geom_medians = {}
        centroids = {}
        avg_pairwise = {}
        distances_from_median = {}
        distances_from_centroid = {}

        for gene in gene_list:
            these_locs = np.where(np.isin(used_mutants,this_data[this_data['mutation_type']==gene]['barcode'].values))

            geom_medians[gene] = geometric_median(all_muts[these_locs,:][0])

            centroids[gene] = centroid(all_muts[these_locs,:][0])
            
            avg_pairwise[gene] = np.mean(distance.pdist(all_muts[these_locs,:][0]))



            distances_from_median[gene] = distance.cdist(all_muts[these_locs,:][0],[geom_medians[gene]]).flatten()
            distances_from_centroid[gene] = distance.cdist(all_muts[these_locs,:][0],[centroids[gene]]).flatten()
            # distances_from_median[gene] = []
            # for loc in these_locs[0]:
                # distances_from_median[gene].append(distance.euclidean(all_muts[loc,:][0],geom_medians[gene]))


        data_situation['geom_medians'].append(geom_medians)
        data_situation['centroids'].append(centroids)
        data_situation['avg_pairwise'].append(avg_pairwise)
        data_situation['distance_from_median'].append(distances_from_median)
        data_situation['distance_from_centroid'].append(distances_from_centroid)


    return data_situation


class NullObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        pass

def tick_base_calculator(bigmin,bigmax,targets=[0.1,0.2,0.3,0.4,0.5,1.0,2.0,5.0,10.0],num_ticks=3):

  farthest = np.max([abs(bigmin),abs(bigmax)])

  for target in targets:
    if int(farthest/target)+1 == num_ticks:
      return target

  return 1.0

def text_color_legend(ax, visible_handles=False, legend_prop={'weight':'semibold'}, bbox_to_anchor=(0,0), **kargs):
    """text_color_legend() -> eliminates legend key and simply colors labels with the color of the lines."""
    handles, labels = ax.get_legend_handles_labels()
    handles = [handle[0] if type(handle) == list else handle for handle in handles]
    if not visible_handles: 
        kargs['handler_map'] = {handle:NullObjectHandler() for handle in handles}
    L = ax.legend(handles, labels, prop=legend_prop, bbox_to_anchor=bbox_to_anchor, **kargs)
    for handle, text in zip(handles, L.get_texts()):
        text.set_color(handle.get_facecolor() if handle.get_fill() else handle.get_edgecolor())
    return L


def select_train_test_mutants(fitness_data,max_train=4,max_test=10,exclusion_list=['other','NotSequenced','NotSequenced_adaptive','ExpNeutral'],include_all_neutrals=True):

    mutation_types = np.unique(fitness_data['mutation_type'].values)

    train_list = []

    for mut_type in mutation_types:
        if not mut_type in exclusion_list:
            this_mut_type = fitness_data[(fitness_data['mutation_type'].isin([mut_type]))]
            n_samples =  min([int(np.floor(len(this_mut_type.index)/2)),max_train])
            # print(mut_type,len(this_mut_type.index),n_samples)
        
            train_list = sorted(np.unique(train_list + list(np.random.choice(this_mut_type['barcode'].values,n_samples,replace=False))))

    # print(len(train_list))
    test_list = []

    for mut_type in mutation_types:
        if not mut_type in exclusion_list:
            this_mut_type = fitness_data[(fitness_data['mutation_type'].isin([mut_type]))]
            options = [bc for bc in this_mut_type['barcode'].values if bc not in train_list]
            n_samples = min([int(np.floor(len(options))),max_test])
            # print(mut_type,len(this_mut_type.index),n_samples)
            test_list = sorted(np.unique(test_list + list(np.random.choice(options,n_samples,replace=False))))

    if include_all_neutrals:
        exp_neutral =  fitness_data[fitness_data['mutation_type'].isin(['ExpNeutral'])]
        test_list = sorted(np.unique(test_list + [bc for bc in list(exp_neutral['barcode'].values) if bc not in test_list]))
        # print(len(exp_neutral['barcode'].values))

    test_list = [bc for bc in test_list if bc not in train_list]
    # print(len(test_list))

    return train_list, test_list


def fits_by_condition(dataset,this_data,max_model='default',bc_removal=False):

    all_guesses = dataset['CV_all_guesses']
    both_old = dataset['both_old']
    dhats = dataset['dhats']
    test = dataset['test']
    both_new = dataset['both_new']
    guesses = dataset['guesses']
    this_fitness = dataset['this_fitness']


    test_conditions  = dataset['test_conditions']

    if max_model == 'default':
        max_model = dataset['CV_best_rank_index']+1

    train  = copy.copy(dataset['train'])
    test = copy.copy(dataset['test'])

    training_bcs = copy.copy(dataset['training_bcs'])
    testing_bcs = copy.copy(dataset['testing_bcs'])


    if bc_removal != False:
        train_where = np.where(np.isin(training_bcs,bc_removal))[0]
        test_where = np.where(np.isin(testing_bcs,bc_removal))[0]
        
        train[1] = [x for i,x in enumerate(train[1]) if i not in train_where]
        test[1] = [x for i,x in enumerate(test[1]) if i not in test_where]
        
        training_bcs = [x for i,x in enumerate(training_bcs) if i not in test_where]
        testing_bcs = [x for i,x in enumerate(testing_bcs) if i not in test_where]


    nonsubtle_fits = []
    subtle_fits = []
    overall_subtle = []
    overall_nonsubtle = []
    for model in range(max_model):

        types = this_data[this_data['barcode'].isin(testing_bcs)]['mutation_type'].values
        
        new_mut_locs = dataset['mut_locs'][model][test[1]]
        
        old_cond_locs = dataset['cond_locs'][model][train[0]]

        new_cond_locs = dataset['cond_locs'][model][test[0]]
        
        these_sigmas = dataset['dimension_weights'][model]
        these_sigmas = these_sigmas[:model+1,:model+1]

        


        overall_nonsubtle.append(var_explained_weighted_by_type(this_fitness[test[1],:][:,test[0]],
                                                                      np.dot(new_mut_locs,np.dot(these_sigmas,new_cond_locs.T)),types)[0])

        this_sse = np.asarray([var_explained_weighted_by_type(this_fitness[test[1],test[0][i]],
                                                                      np.dot(new_mut_locs,np.dot(these_sigmas,new_cond_locs.T))[:,i],types)[0] for i in range(new_cond_locs.shape[0])])
        nonsubtle_fits.append(this_sse)
        
        overall_subtle.append(var_explained_weighted_by_type(this_fitness[test[1],:][:,train[0]],
                                                                      np.dot(new_mut_locs,np.dot(these_sigmas,old_cond_locs.T)),types)[0])


        subtle_sse = np.asarray([var_explained_weighted_by_type(this_fitness[test[1],train[0][i]],
                                                                      np.dot(new_mut_locs,np.dot(these_sigmas,old_cond_locs.T))[:,i],types)[0] for i in range(old_cond_locs.shape[0])])

        subtle_fits.append(subtle_sse)

    return overall_subtle, subtle_fits, overall_nonsubtle, nonsubtle_fits
        






