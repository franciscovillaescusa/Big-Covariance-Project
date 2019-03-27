# For each realization of the fiducial cosmology, this script reads the power spectrum
# and computes the HMF and VSF. It then computes the covariance of Pk, HMF & VSF
from mpi4py import MPI
import numpy as np
import sys,os,h5py
import readfof

###### MPI DEFINITIONS ######
comm   = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()

root = '/home/fvillaescusa/data/pdf_information/fiducial/'
#################################### INPUT ###########################################
# general parameters
realizations = 15000
BoxSize      = 1000.0 #Mpc/h
snapnum      = 4
suffix       = 'Pk+HMF+VSF' #label to use in the output files

# parameters for the Pk subcovariance part
kmax = 1.0 #h/Mpc

# parameters for the VSF subcovariance part
VSF_bins    = 29   #number of bins in the VSF
grid        = 1024
Rmax        = 33.0 #Mpc/h 
Rmin        = 4.0  #Mpc/h 
delete_bins = [1,2,3,5,7,9] #-> only for rebinning

# parameters for the HMF subcovariance part
Nmin     = 100.0    #minimum number of CDM particles in a halo
Nmax     = 10000.0  #maximum number of CDM particles in a halo
HMF_bins = 15       #number of bins in the HMF
######################################################################################

# find the redshift
z_dict = {4:0, 3:0.5, 2:1, 1:2, 0:3}
z      = z_dict[snapnum]

# compute the new bins of the VSF, their mean and size
bins_VSF = np.logspace(np.log10(Rmin), np.log10(Rmax), VSF_bins+1)
bins_VSF = np.delete(bins_VSF, delete_bins) #remove the unwanted bins
dR       = bins_VSF[1:]-bins_VSF[:-1]       #size of the bin (Mpc/h)
VSF_bins = VSF_bins - len(delete_bins)      #number of VSF bins
Rmean    = bins_VSF[1:]                     #Mpc/h

# compute the HMF bins, their mean and size
bins_HMF = np.logspace(np.log10(Nmin), np.log10(Nmax), HMF_bins+1)
dM       = bins_HMF[1:] - bins_HMF[:-1]       #size of the bin
Mmean    = 0.5*(bins_HMF[1:] + bins_HMF[:-1]) #mean of the bin

############# read data to compute the covariance ############
if myrank==0:  print 'Reading data from the realizations at z=%s...'%z

# find the indexes and number of bins to use in the P(k)
f1 = '%s/0/Pk_m_z=%s.txt'%(root,z)  #read the Pk of one file
k_real, Pk_real, Nmodes = np.loadtxt(f1, unpack=True) 
indexes = np.where(k_real<=kmax)[0] #find the indexes of the k<kmax modes
Pk_bins = len(indexes)              #find the number of bins

# find the total number of bins
all_bins = Pk_bins + VSF_bins + HMF_bins

# define the array hosting all the different Pk+HMF+VSF
data_p = np.zeros((realizations,all_bins), dtype=np.float64) #data each cpu reads
data   = np.zeros((realizations,all_bins), dtype=np.float64) #Matrix with all data

# each cpu will only read a fraction of all realizations
numbers = np.where(np.arange(realizations)%nprocs==myrank)[0]

# each cpu reads its realizations
count, count_p = 0, 0
for i in numbers:
    if i%1000==0:  print i

    # read the Pk file
    f1 = '%s/%d/Pk_m_z=%s.txt'%(root,i,z)
    k_real, Pk_real, Nmodes = np.loadtxt(f1, unpack=True) 
    k_real, Pk_real = k_real[indexes], Pk_real[indexes]

    # compute the HMF
    snapdir  = '%s/%d/'%(root,i)
    FoF      = readfof.FoF_catalog(snapdir,snapnum,long_ids=False,
                                   swap=False,SFR=False,read_IDs=False)
    mass     = FoF.GroupMass*1e10               #Msun/h  
    part     = FoF.GroupLen                     #number of particles in the halo
    p_mass   = mass[0]/part[0]                  #mass of a single particle in Msun/h
    mass     = p_mass*(part*(1.0-part**(-0.6))) #correct FoF masses
    HMF_real = np.histogram(part, bins=bins_HMF)[0]/(dM*BoxSize**3)
    HMF_real *= 1e12

    # compute the VSF
    f = h5py.File('%s/%d/voids/void_catalogue_z=%s.hdf5'%(root,i,z), 'r')
    radius   = f['radius'][:]*BoxSize/grid-1e-5 
    f.close()   
    VSF_real = np.histogram(radius, bins=bins_VSF)[0]/(dR*BoxSize**3)
    VSF_real *= 1e9

    # join Pk, HMF and VSF
    data_p[i] = np.hstack([Pk_real, HMF_real, VSF_real])
    count_p += 1

# join the data of the different cpus into a single matrix (only for master)
comm.Reduce(data_p, data, root=0);  del data_p
count = np.array(count);  count_p = np.array(count_p)
comm.Reduce(count_p, count, root=0)
if myrank==0:  print 'Number of realizations worked on: %d'%count

# compute the meand and std of the data
data_mean = np.mean(data, axis=0)
data_std  = np.std(data,  axis=0)
vector    = np.hstack([k_real, Mmean, Rmean]) #corresponding quantities

# master computes mean and standard deviation
if myrank==0:
    np.savetxt('mean_%s_%d_z=%s.txt'%(suffix,realizations,z), 
               np.transpose([vector, data_mean, data_std]))


########## compute the covariance ##########
if myrank>0:  sys.exit()

# define the arrays containing the covariance matrices
print 'Computing the covariance at z=%s...'%z
Cov      = np.zeros((all_bins,all_bins), dtype=np.float64)
Cov_norm = np.zeros((all_bins,all_bins), dtype=np.float64)

# compute the covariance matrix
for i in xrange(all_bins):
    for j in xrange(i,all_bins):
        Cov[i,j] = np.sum((data[:,i]-data_mean[i])*(data[:,j]-data_mean[j])) 
        if j>i:  Cov[j,i] = Cov[i,j]
Cov /= (realizations-1.0)

# compute the normalized covariance matrix
f = open('Cov_norm_%s_%d_z=%s.txt'%(suffix,realizations,z), 'w')
g = open('Cov_%s_%d_z=%s.txt'%(suffix,realizations,z), 'w')
for i in xrange(all_bins):
    for j in xrange(all_bins):
        Cov_norm[i,j] = Cov[i,j]/np.sqrt(Cov[i,i]*Cov[j,j])
        f.write(str(vector[i])+' '+str(vector[j])+' '+str(Cov_norm[i,j])+'\n')
        g.write(str(vector[i])+' '+str(vector[j])+' '+str(Cov[i,j])+'\n')
f.close();  g.close()
    

