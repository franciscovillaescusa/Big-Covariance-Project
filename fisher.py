# This script reads the covariance matrix and the derivatives and computes the Fisher
# matrix for either P(k), HMF and VSF, and any combinations of them
import numpy as np
import sys,os
sys.path.append('/simons/scratch/fvillaescusa/pdf_information/library')
import BCP_library as BCPL


root      = '/home/fvillaescusa/data/pdf_information/'
root_der1 = '/home/fvillaescusa/data/pdf_information/derivatives/Pk_m/'
root_der2 = '/home/fvillaescusa/data/pdf_information/derivatives/voids/'
##################################### INPUT ###########################################
parameters       = ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']
realizations_Cov = 15000  #number of realizations to use for the covariance
realizations_der = 250    #number of realizations to use for the derivatives
z                = 0      #redshift

# parameters of the Pk
kmax  = 0.2 #h/Mpc
do_Pk = True

# parameters of the VSF
VSF_bins = 23
do_VSF   = True

# parameters of the HMF
Nmin     = 100.0    #minimum number of CDM particles in a halo
Nmax     = 10000.0  #maximum number of CDM particles in a halo
HMF_bins = 15       #number of bins in the HMF
do_HMF   = True
#######################################################################################


########################## Inverse of Covariance #######################
# read the covariance. This file should include the covariance of all observables
# even if only a subset of them is considered
f_Cov = '../covariance/Cov_Pk+HMF+VSF_%d_z=%s.txt'%(realizations_Cov,z)
kMR1, kMR2, Cov = np.loadtxt(f_Cov, unpack=True)  #read covariance
bins            = int(round(np.sqrt(len(kMR1))))  #find the number of bins
Cov             = np.reshape(Cov, (bins,bins))    #reshape covariance
kMR2            = kMR2[:bins]                     #get the k, M and R bins

# define the array with the indexes of the elements belonging to the covariance
ids = np.array([], dtype=np.int32);  suffix = '_'
if do_Pk:
    k          = kMR2[:-(HMF_bins+VSF_bins)] #all k-bins in the covariance
    indexes_Pk = np.where(k<kmax)[0]         #find k-bins where k<kmax
    k          = k[indexes_Pk]               #only consider k<kmax bins
    ids        = np.hstack([ids, indexes_Pk])
    suffix     += 'Pk_'

if do_HMF:
    M           = kMR2[-HMF_bins-VSF_bins:-VSF_bins]       #all M-bins
    indexes_HMF = np.arange(-HMF_bins-VSF_bins, -VSF_bins)
    ids         = np.hstack([ids, indexes_HMF])
    suffix += 'HMF_'

if do_VSF:
    R           = kMR2[-VSF_bins:]              #all R-bins
    indexes_VSF = np.arange(-VSF_bins, 0)
    ids         = np.hstack([ids, indexes_VSF])
    suffix      += 'VSF_'

# name of output file
fout = 'Fisher%skmax=%.1f_%dder_%dcov_z=%s.npy'\
       %(suffix, kmax, realizations_der, realizations_Cov, z)

# construct the new covariance matrix
all_bins = len(ids) #size of the new covariance matrix
new_Cov  = np.zeros((all_bins,all_bins), dtype=np.float64)
for i,id1 in enumerate(ids):
    for j,id2 in enumerate(ids):
        new_Cov[i,j] = Cov[id1,id2]

# compute the inverse of the covariance
ICov = BCPL.Inv_Cov(new_Cov) 
########################################################################

############################## Derivatives #############################
# find the number of parameters and define the matrix containing the derivatives
params_num = len(parameters)                                    #number of cosmo params
derivative = np.zeros((params_num, all_bins), dtype=np.float64) #array with derivatives

# do a loop over all the cosmological parameters
for i in xrange(params_num):
        
    # define a temporary array that host the derivative vector of the parameter
    derivat = np.array([], dtype=np.float64)    

    # read the Pk derivatives
    if do_Pk:
        f = '%s/derivative_%s_Pk_m_%d_z=%d.txt'\
            %(root_der1,parameters[i],realizations_der,z)
        k_der, der_Pk = np.loadtxt(f, unpack=True)
        indexes = np.where(k_der<kmax)[0] #select on k-bins where k<kmax
        k_der, der_Pk = k_der[indexes], der_Pk[indexes] 
        if not(np.allclose(k_der, k, rtol=1e-8, atol=1e-10)):  #check k-values 
            raise Exception('k-values differ in the Pk derivatives!!!')
        derivat = np.hstack([derivat, der_Pk])

    # read the HMF derivatives
    if do_HMF:
        f = '../../HMF/derivatives/derivative_%s_%d_%.1e_%.1e_%d_z=%s.txt'\
            %(parameters[i], realizations_der, Nmin, Nmax, HMF_bins, z)
        M_der, der_HMF, dder_HMF = np.loadtxt(f, unpack=True)
        der_HMF *= 1e12
        if not(np.allclose(M_der, M, rtol=1e-8, atol=1e-10)):  #check M-bins
            raise Exception('M-values differ in the HMF derivatives!!!')
        derivat = np.hstack([derivat, der_HMF])

    # read the VSF derivatives
    if do_VSF:
        f = '%s/derivative_%s_%d_rebin_z=%d.txt'\
            %(root_der2,parameters[i],realizations_der,z)
        R_der, der_VSF = np.loadtxt(f, unpack=True)
        der_VSF *= 1e9
        if not(np.allclose(R_der, R, rtol=1e-8, atol=1e-10)):  #check R-bins
            raise Exception('k-values differ in the Pk derivatives!!!')
        derivat = np.hstack([derivat, der_VSF])
        
    # move content of temporary array to derivative matrix
    derivative[i] = derivat
########################################################################

############################ Fisher matrix #############################
Fisher = np.zeros((params_num, params_num), dtype=np.float64)
for i in xrange(params_num):
    for j in xrange(i, params_num):
        if i==j:
            Fisher[i,j] = np.dot(derivative[i], np.dot(ICov, derivative[i]))
        else:
            Fisher[i,j] = 0.5*(np.dot(derivative[i], np.dot(ICov, derivative[j])) + \
                               np.dot(derivative[j], np.dot(ICov, derivative[i])))
            Fisher[j,i] = Fisher[i,j]

# save results to file
np.save(fout, Fisher)

# compute the marginalized error on the parameters
IFisher = np.linalg.inv(Fisher)
for i in xrange(params_num):
    print 'Error on %03s = %.5f'%(parameters[i], np.sqrt(IFisher[i,i]))
########################################################################
