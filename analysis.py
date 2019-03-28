# This scripts computes:
# 1) The covariance matrix for all probes (Pk+HMF+VSF)
# 2) The mean of all statistics for all cosmologies
# 3) The derivatives of the statistics with respect to the cosmological parameters
# 4) The Fisher matrix for the considered probes
from mpi4py import MPI
import numpy as np
import sys,os
sys.path.append('/home/fvillaescusa/data/pdf_information/analysis')
import analysis_library as AL

###### MPI DEFINITIONS ######
comm   = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()

##################################### INPUT ###########################################
# folders with the data and output files
root_data    = '/home/fvillaescusa/data/pdf_information/'
root_results = '/home/fvillaescusa/data/pdf_information/analysis/git_repo/'

# general parameters
parameters       = ['Om', 'Ob2', 'h', 'ns', 's8', 'Mnu']
BoxSize          = 1000.0  #Mpc/h
snapnum          = 4       #z=0
realizations_Cov = 15000   #number of realizations for the covariance
realizations_der = 250     #number of realizations for the derivatives

# parameters of the Pk
kmax  = 0.2 #h/Mpc
do_Pk = True

# parameters of the VSF
grid        = 1024
VSF_bins    = 29
Rmin        = 4.0   #Mpc/h
Rmax        = 33.0  #Mpc/h
delete_bins = [1,2,3,5,7,9]
do_VSF      = True

# parameters of the HMF
Nmin     = 100.0    #minimum number of CDM particles in a halo
Nmax     = 10000.0  #maximum number of CDM particles in a halo
HMF_bins = 15       #number of bins in the HMF
do_HMF   = True
#######################################################################################

# find the corresponding redshift
z_dict = {4:0, 3:0.5, 2:1, 1:2, 0:3};  z = z_dict[snapnum] 

######################## COMPUTE/READ FULL COVARIANCE #############################
# read/compute the covariance of all the probes (Pk+HMF+VSF)
# bins is an array with the number of bins in each statistics
# X is an array with the value of the statistics in each bin
# Cov is the covariance matrix with size bins x bins
bins, X, Cov = AL.covariance(realizations_Cov, BoxSize, snapnum, root_data, root_results, 
                             kmax, grid, Rmin, Rmax, VSF_bins, delete_bins,
                             HMF_bins, Nmin, Nmax)
###################################################################################

########################## COMPUTE ALL DERIVATIVES ################################
# compute the mean values of the different statistics for all cosmologies and 
# compute the derivatives of the statistics with respect to the parameters
AL.derivatives(realizations_der, BoxSize, snapnum, root_data, root_results, 
               kmax, grid, Rmin, Rmax, VSF_bins, delete_bins, HMF_bins, Nmin, Nmax)
###################################################################################

if myrank>0:  sys.exit() #here finishes the parallelism 

######################## COMPUTE INVERSE OF (SUB)-COVARIANCE ######################
# find the (sub)covariance of the considered observables; invert it
Cov  = AL.subcovariance(Cov, bins, do_Pk, do_HMF, do_VSF);  ICov = AL.Inv_Cov(Cov)
###################################################################################

################################# GENERAL THINGS ##################################
# find the k-bins, M-bins and R-bins and the number of cosmo parameter
k = X[np.arange(0,                np.sum(bins[:1]))]
N = X[np.arange(np.sum(bins[:1]), np.sum(bins[:2]))]
R = X[np.arange(np.sum(bins[:2]), np.sum(bins[:3]))]
all_bins   = Cov.shape[0]
params_num = len(parameters)

# find the different suffixes
suffix_Pk  = 'Pk_%d_%.2f_z=%s.txt'%(realizations_der, kmax, z)
suffix_HMF = 'HMF_%d_%.1e_%.1e_%d_z=%s.txt'%(realizations_der, Nmin, Nmax, HMF_bins, z)
suffix_VSF = 'VSF_%d_%.1e_%.1e_%d_%s_z=%s.txt'\
             %(realizations_der, Rmin, Rmax, VSF_bins, delete_bins, z)
###################################################################################

############################## READ DERIVATIVES ###################################
# define the matrix containing the derivatives
derivative = np.zeros((params_num, all_bins), dtype=np.float64)

# do a loop over all the parameters
for i in xrange(params_num):

    # temporary array storing the derivatives
    derivat = np.array([], dtype=np.float64)
        
    if do_Pk:  #read the Pk derivatives
        f = '%s/derivatives/derivative_%s_%s'%(root_results,parameters[i],suffix_Pk)
        k_der, der_Pk = np.loadtxt(f, unpack=True)
        if not(np.allclose(k_der, k, rtol=1e-8, atol=1e-10)):  
            raise Exception('k-values differ in the Pk derivatives!!!')
        derivat = np.hstack([derivat, der_Pk])

    if do_HMF:  #read the HMF derivatives
        f = '%s/derivatives/derivative_%s_%s'%(root_results,parameters[i],suffix_HMF)
        N_der, der_HMF = np.loadtxt(f, unpack=True)
        if not(np.allclose(N_der, N, rtol=1e-8, atol=1e-10)):  
            raise Exception('N-values differ in the HMF derivatives!!!')
        derivat = np.hstack([derivat, der_HMF])

    if do_VSF:  #read the VSF derivatives
        f = '%s/derivatives/derivative_%s_%s'%(root_results,parameters[i],suffix_VSF)
        R_der, der_VSF = np.loadtxt(f, unpack=True)
        if not(np.allclose(R_der, R, rtol=1e-8, atol=1e-10)):  
            raise Exception('k-values differ in the Pk derivatives!!!')
        derivat = np.hstack([derivat, der_VSF])

    derivative[i] = derivat
###################################################################################

#################################### FISHER #######################################
# compute the Fisher matrix
Fisher = np.zeros((params_num, params_num), dtype=np.float64)
for i in xrange(params_num):
    for j in xrange(i, params_num):
        if i==j:
            Fisher[i,j] = np.dot(derivative[i], np.dot(ICov, derivative[i]))
        else:
            Fisher[i,j] = 0.5*(np.dot(derivative[i], np.dot(ICov, derivative[j])) + \
                               np.dot(derivative[j], np.dot(ICov, derivative[i])))
            Fisher[j,i] = Fisher[i,j]

# compute the marginalized error on the parameters
IFisher = np.linalg.inv(Fisher)
for i in xrange(params_num):
    print 'Error on %03s = %.5f'%(parameters[i], np.sqrt(IFisher[i,i]))

# save results to file
fout = 'Fisher_%d_%d'%(realizations_der, realizations_Cov)
if do_Pk:   fout += '_Pk_%.2f'%kmax
if do_HMF:  fout += '_HMF_%.1e_%.1e_%d'%(Nmin, Nmax, HMF_bins)
if do_VSF:  fout += '_VSF_%.1e_%.1e_%d_%s'%(Rmin, Rmax, VSF_bins, delete_bins)
fout += '_z=%s.npy'%z
np.save(fout, Fisher)
###################################################################################


