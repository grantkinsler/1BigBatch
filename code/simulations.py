import sys
import getopt
import numpy as np
import pandas as p
import random
import scipy
from scipy.spatial import distance
# import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from ast import literal_eval
import uuid
import datetime
import csv



selection = True
anc_max = 1.0
max_jump = 0.5

optimum_deviation = 0.1  # pick such that ancestor's fitness does not change by more than some percentage

n_dimensions = 3
n_mutants = 250
n_conditions = 50
n_replicates = 1
fixed_ancestor = False
radius_distribution = False
radius_mean = 2.0
radius_std = 1.0

dry_run = False
plot = False

default_Sigma2 = True
default_weights = True
weirduncle = 0

batch_name = ''


def usage():

    print("\nUsage has not yet been written for this script.")
    print("Contact the author Grant Kinsler at grantkinsler@gmail.com\n")

def write_simulation_information_file(batch_name,Sigma2,weights,replicate_identifier,n_dimensions,n_mutants,n_conditions,max_jump,anc_max,selection,fixed_ancestor,optimum_deviation):
    
    information_dictionary = {'batch_name':[batch_name],'simulation_identifier':[replicate_identifier],'datetime':[datetime.datetime.now().isoformat(' ')],
    'simulation_script':['FGM_simulation.py'],'fitness_function':['Gaussian'],'n_dimensions':[n_dimensions],'n_mutants':[n_mutants],
    'n_conditions':[n_conditions],'max_jump':[max_jump],'anc_max':[anc_max],'selection':[selection],'fixed_ancestor':[fixed_ancestor],'optimum_deviation':[optimum_deviation],
    'Sigma2':[str(np.diag(Sigma2))],'weights':[str(weights)]}

    df = p.DataFrame.from_dict(information_dictionary)
    df.to_csv('InformationFiles/{}_information.txt'.format(replicate_identifier),index=False)

def write_location_files(replicate_identifier,AncestorLocation,MutantLocations,OptimaLocations,Sigma2):
    AncestorLocation.to_csv('LocationFiles/{}_ancestorlocations.csv'.format(replicate_identifier),index=False)
    MutantLocations.to_csv('LocationFiles/{}_mutantlocations.csv'.format(replicate_identifier),index=False)
    OptimaLocations.to_csv('LocationFiles/{}_optimalocations.csv'.format(replicate_identifier),index=False)
    Sigma2.to_csv('LocationFiles/{}_sigma2.csv'.format(replicate_identifier),index=True,index_label='Condition')


def write_fitness_file(replicate_identifier,FitnessandError,n_conditions,n_mutants):
    with open('FitnessFiles/{}_gaussianfitness.csv'.format(replicate_identifier),'w') as csvfile:
        datawriter = csv.writer(csvfile, delimiter=',')
        datawriter.writerow(["Mutant"]+["Condition" + str(k) + "_fitness" if a==0 else  "Condition" + str(k) + "_error"  for a,k in zip(np.tile(range(2),2*n_conditions),np.insert(range(n_conditions),np.arange(n_conditions),range(n_conditions))) ])
        for j in range(n_mutants):
            datawriter.writerow(np.insert(FitnessandError[j,:],0,j))

# try:
#     opts, args = getopt.getopt(sys.argv[1:], "h", ["help","batch_name=","dimensions=","mutants=","conditions=",
#         "max_jump=","replicates=","selection","fixed_ancestor","anc_max=","optimum_deviation=",
#         "weights=","sigma2=","weirduncle=","dry_run","plot"])
# except getopt.GetoptError, error:
#     sys.stderr.write(str(error)+"\n")
#     usage()
#     sys.exit(2)
# for opt, arg in opts:
#     if opt in ("-h", "--help"):
#         usage()
#         sys.exit()
#     elif opt == "--batch_name":
#         batch_name = arg
#     elif opt == "--dimensions":
#         n_dimensions = int(arg)
#     elif opt == "--mutants":
#         n_mutants = int(arg)
#     elif opt == "--conditions":
#         n_conditions = int(arg)
#     elif opt == "--max_jump":
#         max_jump = float(arg)
#     elif opt == "--replicates":
#         n_replicates = int(arg)
#     elif opt == "--selection":
#         selection = True
#     elif opt == "--anc_max":
#         anc_max = float(arg)
#     elif opt == "--fixed_ancestor":
#         fixed_ancestor = True
#     elif opt == "--optimum_deviation":
#         optimum_deviation = float(arg)
#     elif opt == "--weights":
#         weights = np.asarray(literal_eval(arg))
#         default_weights = False
#     elif opt == "--sigma2":
#         Sigma2 = np.diag(np.asarray(literal_eval(arg)))
#         default_Sigma2 = False
#     elif opt == "--dry_run":
#         dry_run = True
#     elif opt == "--plot":
#         plot = True
#     elif opt == "--weirduncle":
#         weirduncle = int(arg)
#     else:
#         sys.stderr.write("Unknown option %s\n" %opt)
#         usage()
#         sys.exit(2)


def radius_pull_normal(mean,std):

    r = np.random.normal(mean,std)

    if r < 0:
        r = 0

    return r

def radius_pull_expo(lam):

    r = np.random.exponential(lam)

    if r < 0:
        r = 0

    return r

def normal_pull(n_dimensions,std,weights):

    X = np.random.normal(0,std,n_dimensions)*weights

    return X


def gaussian_fitness(Mutants,Optima,Ancestor,Sigma2):
    
    if type(Sigma2) != float:
        squared_distances = np.square(distance.cdist(Mutants,Optima,VI=Sigma2,metric='mahalanobis'))
        Anc_dist_squared = np.broadcast_to(np.square(distance.cdist(Optima,Ancestor,VI=Sigma2,metric='mahalanobis')).T,(Mutants.shape[0],Optima.shape[0]))
        relative_fitness = np.exp((Anc_dist_squared - squared_distances) / (2.0 * 1.0)) - 1.0

    else:
        squared_distances = distance.cdist(Mutants, Optima, metric='sqeuclidean')
        Anc_dist_squared = np.broadcast_to(distance.cdist(Optima,Ancestor,metric='sqeuclidean').T,(Mutants.shape[0],Optima.shape[0]))
        relative_fitness = np.exp((Anc_dist_squared - squared_distances) / (2.0 * Sigma2)) - 1.0

    return relative_fitness

def dfe_pdf(fitness,shape=2.0,scale=0.05):

    # want mode to be 0:

    x = fitness + (shape - 1 )*scale

    probability = x**(shape-1) * np.exp(-x/scale) / scale**(shape -1) / scipy.special.gamma(shape)

    return probability


def nball_pull(n_dimensions,max_radius,weights,fixed_radius=False):
    """ Algorithm from Marsaglia 1972 - gives a uniform distribution in the n-ball """
    
    if fixed_radius:
        radius  =  max_radius # fix this radius as the radius for this draw
    else: 
        unit_radius = np.random.uniform(0,1)**(1/float(n_dimensions))  # sample uniformly in ball with radius 1
        radius = max_radius * unit_radius # scale this out to desired max radius

    X = np.random.normal(0,1,n_dimensions)
    random_point = X/np.linalg.norm(X)*radius*weights

    return random_point

def simulation(n_dimensions,n_mutants,n_conditions,called=True,n_replicates=1,max_jump=0.5,anc_max=1.0,optimum_deviation=1.0,weights=1,weirduncle=False,default_Sigma2=True,Sigma2=1,
            default_weights=True,dry_run=True,fixed_ancestor=True,selection=True):

    if default_weights:
        weights = np.ones(n_dimensions)

    if weirduncle:
        n_dimensions = n_dimensions + 1
        uncleweights = np.asarray(np.append(weights,1.0))
        weights = np.asarray(np.append(weights,0.0))
    

    if default_Sigma2:
        Sigma2 = np.eye(n_dimensions)


    for rep in range(n_replicates):

        replicate_identifier = uuid.uuid4().hex # Generate a unique identifier for this replicate (and use hexidecimal representation since it's shorter)
        
        Ancestor = nball_pull(n_dimensions,anc_max,np.ones(n_dimensions),fixed_radius=fixed_ancestor)
        Ancestor = Ancestor.reshape((1,n_dimensions))

        Mutants = np.zeros((n_mutants,n_dimensions))
        mutant_count = 0
        fitness_list = []

        ancestor_distance = np.linalg.norm(Ancestor)
        Original_Peak = np.zeros((1,n_dimensions))

        while mutant_count < n_mutants:
            if mutant_count < weirduncle:
                cartesian_vector = normal_pull(n_dimensions,max_jump,uncleweights)

            else:
                cartesian_vector = normal_pull(n_dimensions,max_jump,weights)

            new_loc = Ancestor + cartesian_vector
            # print new_loc

            fitness = gaussian_fitness(new_loc,Original_Peak,Ancestor,Sigma2)[0][0]


            if selection:
                # p_fix = dfe_pdf(fitness)
                p_fix = fitness # probability of establishment
            else:
                p_fix = 1

            if  np.random.uniform(0,1) < p_fix:
                Mutants[mutant_count,:] = new_loc
                fitness_list.append(fitness)
                mutant_count += 1

        Optima = np.zeros((n_conditions,n_dimensions))
        condition_count = 0

        while condition_count < n_conditions:

            cartesian_vector = nball_pull(n_dimensions,optimum_deviation,np.ones(n_dimensions))

            new_loc = np.zeros((1,n_dimensions)) + cartesian_vector

            Optima[condition_count,:] = new_loc
            condition_count += 1


        fitness_matrix = gaussian_fitness(Mutants,Optima,Ancestor,Sigma2)
        error = np.ones((n_mutants,n_conditions)) # no error in this simulation, will add in another script
        fitness_and_error = np.insert(error,np.arange(fitness_matrix.shape[1]),fitness_matrix,axis=1)

        mutantlocations = p.DataFrame(Mutants)
        mutantlocations.columns = ['Coordinate' + str(i) for i in range(n_dimensions)]
        mutantlocations['Mutant'] = range(n_mutants)

        optimalocations = p.DataFrame(Optima)
        optimalocations.columns = ['Coordinate' + str(i) for i in range(n_dimensions)]
        optimalocations['Condition'] = range(n_conditions)

        if type(Sigma2) == float:
            sigma2output = p.DataFrame(np.ones((n_conditions,n_dimensions))*Sigma2)
        else:
            sigma2output = p.DataFrame(Sigma2)

        ancestorlocs = p.DataFrame(Ancestor)
        ancestorlocs.columns = ['Coordinate' + str(i) for i in range(n_dimensions)]

        
        """ Write the output files """
        if not dry_run:
            write_simulation_information_file(batch_name,Sigma2,weights,replicate_identifier,n_dimensions,n_mutants,n_conditions,max_jump,anc_max,selection,fixed_ancestor,optimum_deviation)
            write_location_files(replicate_identifier,ancestorlocs,mutantlocations,optimalocations,sigma2output)
            write_fitness_file(replicate_identifier,fitness_and_error,n_conditions,n_mutants)

    if called:
        return fitness_matrix, replicate_identifier, n_dimensions, n_mutants, n_conditions

# simulation()
