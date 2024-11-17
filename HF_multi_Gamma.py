# -*- coding: utf-8 -*-
# @File : GW_planewave.py
# @Author : Zhongqing Guo
# @Time : 2023/03/09 03:33:58
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
from mpi4py import MPI
import time
import sys
import os
import warnings
from numpy import (
    load, savez, savez_compressed,
    array, linspace, arange, zeros, ones, eye, full, identity,
    hstack, vstack, stack, concatenate, sort, argsort, where, all, any, expand_dims,
    tensordot, einsum, dot, vdot, inner, kron, cross,
    trace, transpose, conj, real, imag, diag, sum, prod, diagonal, fill_diagonal, roll,
    around, abs, angle, pi, sqrt, exp, log, sin, cos, tan, heaviside,
    min, max, nonzero,
)
from numpy.linalg import eigh, eigvalsh, det, inv, norm, eig, eigvals
# from scipy.sparse import coo_array
from matplotlib.pyplot import subplots, figure, plot, imshow, scatter
from numba import njit

# Constants
lattice=2 # 1=rectangular 2=triangular
Ls0=float(sys.argv[1])
num_k1 = int(sys.argv[2]) # 
nD = int(sys.argv[3]) # 
Dfield = float(sys.argv[4])
epsilon_d = float(sys.argv[5])
N1=int(sys.argv[6])
# filling_e=1/9
filling_e=1
init_type=int(sys.argv[7])
vF0=2.1354*2.46#5.253084
tperp=0.34
vF = vF0
# nD=int((Ls/Ls_n(N1)))+1
# if nD%2==0:
#     nD+=1
# num_k1=(round(63/nD/6))*6
# if abs(filling_e-1.)<1e-9:
#     Ls/=sqrt(2)
Uonsite=zeros(2,'float') # in eV
Uonsite[0]=-Dfield/2.
Uonsite[1]=Dfield/2.

Ls=Ls0
num_k0=num_k1
num_kpt=num_k1*num_k0
num_G=nD**2
order_kin=float(N1) # order of kinetic energy
# filling=2  ## For Wigner crystal, filling is usually 1 or 2, corresponding to spin polarized or unpolarized case.
meff=1.3 ## this is the effective mass at the band edge of the insulating substrate, in untis of electron mass.

delta=.0 # infinitesimal in Green's function

# d0=3.35
# Eext = Dfield/epsilon_d
# Uonsite[0]=-Eext*(N1-1)*d0/2
# Uonsite[1]=Eext*(N1-1)*d0/2


if lattice==1:
### Rectangular lattice ###
    r_xy=1.
    # r_xy=3.89/3.20
    Lx=Ls
    Ly=r_xy*Ls
    lattice_R=array([
        [Lx,0.,0.],
        [0.,Ly,0.],
        [0.,0.,1.],
    ])
elif lattice==2:
### Triangular lattice ###
    lattice_R=array([
        [sqrt(3.)*Ls/2.,-Ls/2.,0.],
        [sqrt(3.)*Ls/2., Ls/2.,0.],
        [            0.,    0.,1.],
    ])

lattice_K=inv(lattice_R.T)*2.*pi
Ad=det(lattice_R)
Uvalue=14.4*2.*pi/Ad # the volume of cell \Omega_0 in bare Coulomb potential has been divided by here!!!!!!!!!!
ds=400. # A

bvec0=lattice_K[0,:2]
bvec1=lattice_K[1,:2]
# kpG_cutoff=norm(bvec0)*sqrt(3)/4*nD
kgrid=arange(num_k1)-num_k1//2 # even num_k1
kvec=((kgrid[:,None]*bvec0[None,:]/num_k0)[:,None,:]+(kgrid[:,None]*bvec1[None,:]/num_k1)[None,:,:]).reshape([-1,2])
Ggrid=arange(nD)-nD//2
Gvec=((Ggrid[:,None]*bvec0[None,:])[:,None,:]+(Ggrid[:,None]*bvec1[None,:])[None,:,:]).reshape([-1,2])

sq3=sqrt(3)
k_GK=[]
k_GK_ind=[]
k_GM=[]
k_GM_ind=[]
k_MK=[]
k_MK_ind=[]
k_GKp=[]
k_GKp_ind=[]
k_GMp=[]
k_GMp_ind=[]
k_MpKp=[]
k_MpKp_ind=[]
for ik,k in enumerate(kvec):
    kx,ky=k
    if abs(kx)<1e-9:
        k_GK.append(k)
        k_GK_ind.append(ik)
    if abs(3*kx+sq3*ky)<1e-9:
        k_GM.append(k)
        k_GM_ind.append(ik)
    if abs(-sq3*kx+3*ky-3*abs(bvec0[1]*2/3))<1e-9:
        k_MK.append(k)
        k_MK_ind.append(ik)
    if abs(kx)<1e-9:
        k_GKp.append(k)
        k_GKp_ind.append(ik)
    if abs(3*kx-sq3*ky)<1e-9:
        k_GMp.append(k)
        k_GMp_ind.append(ik)
    if abs(sq3*kx+3*ky+3*abs(bvec0[1]*2/3))<1e-9:
        k_MpKp.append(k)
        k_MpKp_ind.append(ik)
k_GK=array(k_GK)
k_GK_ind=array(k_GK_ind)
k_GK_ind=k_GK_ind[argsort(k_GK[:,1])]
k_GM=array(k_GM)
k_GM_ind=array(k_GM_ind)
k_GM_ind=k_GM_ind[argsort(k_GM[:,0])]
k_MK=array(k_MK)
k_MK_ind=array(k_MK_ind)
k_MK_ind=k_MK_ind[argsort(k_MK[:,1])]
k_GKp=array(k_GKp)
k_GKp_ind=array(k_GKp_ind)
k_GKp_ind=k_GKp_ind[argsort(k_GKp[:,1])]
k_GMp=array(k_GMp)
k_GMp_ind=array(k_GMp_ind)
k_GMp_ind=k_GMp_ind[argsort(k_GM[:,0])]
k_MpKp=array(k_MpKp)
k_MpKp_ind=array(k_MpKp_ind)
k_MpKp_ind=k_MpKp_ind[argsort(k_MpKp[:,1])]
klenGK=norm(kvec[k_GK_ind[1]]-kvec[k_GK_ind[0]])
klenGM=norm(kvec[k_GM_ind[1]]-kvec[k_GM_ind[0]])
klenMK=norm(kvec[k_MK_ind[1]]-kvec[k_MK_ind[0]])
klenGKp=norm(kvec[k_GKp_ind[1]]-kvec[k_GKp_ind[0]])
klenGMp=norm(kvec[k_GMp_ind[1]]-kvec[k_GMp_ind[0]])
klenMpKp=norm(kvec[k_MpKp_ind[1]]-kvec[k_MpKp_ind[0]])

band_index=[]
kline=[0]
for ik in range((k_GK_ind.shape[0]+1)//3):
    band_index.append(k_GK_ind[k_GK_ind.shape[0]-k_GK_ind.shape[0]//6-1-ik])
    kline.append(klenGK)
for ik in range(k_GM_ind.shape[0]//2):
    band_index.append(k_GM_ind[k_GM_ind.shape[0]//2-ik])
    kline.append(klenGM)
for ik in range(num_k1//6+1):
    band_index.append(k_MK_ind[ik])
    kline.append(klenMK)
kline=np.cumsum(array(kline))[:-1]
xticks=[kline[0], kline[(k_GK_ind.shape[0]+1)//3], kline[(k_GK_ind.shape[0]+1)//3+k_GM_ind.shape[0]//2],kline[-1]]
xlabels=['$K_s$','$\Gamma_s$', '$M_s$','$K_s$']
band_index=array(band_index)

band_index_p=[]
kline=[0]
for ik in range((k_GKp_ind.shape[0]+1)//3):
    band_index_p.append(k_GKp_ind[k_GKp_ind.shape[0]//6+ik])
    kline.append(klenGK)
for ik in range(k_GM_ind.shape[0]//2):
    band_index_p.append(k_GMp_ind[k_GMp_ind.shape[0]//2-ik])
    kline.append(klenGM)
for ik in range(num_k1//6+1):
    band_index_p.append(k_MpKp_ind[-ik-1])
    kline.append(klenMK)
kline=np.cumsum(array(kline))[:-1]
xticks=[kline[0], kline[(k_GKp_ind.shape[0]+1)//3], kline[(k_GKp_ind.shape[0]+1)//3+k_GMp_ind.shape[0]//2],kline[-1]]
xlabels_p=['$K_s\'$','$\Gamma_s$', '$M_s\'$','$K_s\'$']
band_index_p=array(band_index_p)

comm=MPI.COMM_WORLD
size=comm.Get_size()
rank=comm.Get_rank()

pwd=os.getcwd()
folder=f'data_nD{nD}'
path=f'{pwd}/{folder}'
if rank==0:
    if not os.path.exists(f'{pwd}/{folder}'):
        os.mkdir(path)
        print(f'{path} is created')
    else:
        print(f'{path} has been created')
     
nsub=2
filling=num_G+filling_e ## For Wigner crystal, filling is usually 1 or 2, corresponding to spin polarized or unpolarized case.
def make_hamk0(kvec,Gvec,Uonsite,n=N1):
    kpG=kvec[:,None,:]+Gvec[None,:,:]
    kpG_mat=zeros((kvec.shape[0],num_G*nsub,num_G*nsub),'complex')
    for iG in range(num_G):
        kpG_mat[:,iG*nsub+0,iG*nsub+1]=(kpG[:,iG,0]+kpG[:,iG,1]*1.j)**n
        kpG_mat[:,iG*nsub+1,iG*nsub+0]=(kpG[:,iG,0]-kpG[:,iG,1]*1.j)**n
        # kpG_mat[:,iG*nsub+0,iG*nsub+1]=(-kpG[:,iG,0]+kpG[:,iG,1]*1.j)**n
        # kpG_mat[:,iG*nsub+1,iG*nsub+0]=(-kpG[:,iG,0]-kpG[:,iG,1]*1.j)**n
    H0=(-1/tperp)**(n-1)*vF**(n)*kpG_mat
    for iG in range(num_G):
        H0[:,iG*nsub+0,iG*nsub+0]=Uonsite[0]
        H0[:,iG*nsub+1,iG*nsub+1]=Uonsite[1]
    return H0#+diag(linspace(0,1,num_G*nsub)*1e-9)[None,:,:]
nband=num_G*nsub

def chemical_potential(Ek,filling,output=False):
    Emin=Ek.min()
    Emax=Ek.max()
    num_loop=100
    for i in range(num_loop):
        mu = (Emin + Emax)/2.
        occ = sum(heaviside(mu-Ek,1.))/num_kpt
        error = abs(occ-filling)
        if error<1e-7:
            # if output:
            #     print('The filling factor is ', occ)
            break            
        if occ>filling:
            Emax = mu
        else:
            Emin = mu
        if i == num_loop-1 and output:
            print("The occupation number can't be reached!!!!!!")
            print('The filling factor is', occ)
    return mu

@njit
def find_GQindex():
    iGpQ=zeros((num_G,num_G),'int')
    iGmQ=zeros((num_G,num_G),'int')
    for iQ in range(num_G):
        for iG in range(num_G):
            iGpQ[iG,iQ]=1000
            iGmQ[iG,iQ]=1000
            GpQvec=Gvec[iG,:]+Gvec[iQ,:]
            GmQvec=Gvec[iG,:]-Gvec[iQ,:]
            for iGp in range(num_G):
                dGpQ=GpQvec-Gvec[iGp,:]
                if (dGpQ[0]**2+dGpQ[1]**2)<1e-9:
                    iGpQ[iG,iQ]=iGp
                dGmQ=GmQvec-Gvec[iGp,:]
                if (dGmQ[0]**2+dGmQ[1]**2)<1e-9:
                    iGmQ[iG,iQ]=iGp 
    return iGpQ,iGmQ
iGpQ,iGmQ=find_GQindex()

@njit
def VQ(q): 
    norm_q=norm(q)
    if norm_q>1e-9:
    # if norm_q>1e-9 and norm_q<kpG_cutoff*2:
        return np.tanh(norm_q*ds)/norm_q
    else:
        return 0.

@njit
def VQ_mat(kvec,k_subset,iQ):
    Gvec0=[]
    Gvec1=[]
    Qvec=Gvec[iQ]
    for iG0 in range(num_G):
        for iG1 in range(num_G):
            GGQ=Gvec[iG0]-Gvec[iG1]+Gvec[iQ]
            if (GGQ[0]**2+GGQ[1]**2)<1e-9:
                Gvec0.append(iG0)
                Gvec1.append(iG1)
    VQ_mat=zeros((kvec.shape[0],k_subset.shape[0],len(Gvec0),len(Gvec1)),'float')
    for i0,k0 in enumerate(kvec):
        for i1,k1 in enumerate(k_subset):
            for i2,iG0 in enumerate(Gvec0):
                for i3,iG1 in enumerate(Gvec1):
                    VQ_mat[i0,i1,i2,i3]=VQ(k0-k1+Gvec[iG0]-Gvec[iG1]+Qvec)
    return VQ_mat

@njit
def mean_field_iQ(iQ,k_subset,rhoH,rho_up,rho_up_subset,VQ_H,VQ_F):
    # Hartree energy and potential
    EHartree=0.
    paramH=zeros((num_G*nsub,num_G*nsub),'complex')
    rhoQ=0.j
    rhoQ_T=0.j
    for iG in range(num_G):
        if iGpQ[iG,iQ]<500:
            for isub0 in range(nsub):
                rhoQ+=rhoH[iGpQ[iG,iQ]*nsub+isub0,iG*nsub+isub0]
        if iGmQ[iG,iQ]<500:
            for isub0 in range(nsub):
                rhoQ_T+=rhoH[iGmQ[iG,iQ]*nsub+isub0,iG*nsub+isub0]
    # print(rhoQ[0])
    rhoQ_VQ=rhoQ*VQ_H/epsilon_d*Uvalue
    for iG in range(num_G):
        if iGpQ[iG,iQ]<500:
            for isub0 in range(nsub):
                paramH[iGpQ[iG,iQ]*nsub+isub0,iG*nsub+isub0]=rhoQ_VQ
    EHartree+=rhoQ_VQ*rhoQ_T

    # Fock energy and potential
    EFock_up=0.
    paramF_up=zeros((k_subset.shape[0],num_G*nsub,num_G*nsub),'complex')
    # if rank==0:
    #     print(f'number of Q left: {num_G-iQ}')
    VQ_F_shape=VQ_F.shape
    rhoQ_up=zeros((VQ_F_shape[0],VQ_F_shape[2],nsub,nsub),'complex')
    rhoQ_up_T=zeros((VQ_F_shape[1],VQ_F_shape[3],nsub,nsub),'complex')
    rhoQ_up_VQ=zeros((VQ_F_shape[1],VQ_F_shape[3],nsub,nsub),'complex')
    count=0
    for iG in range(num_G):
        if iGpQ[iG,iQ]<500:
            for isub0 in range(nsub):
                for isub1 in range(nsub):
                    rhoQ_up[:,count,isub0,isub1]=rho_up[:,iGpQ[iG,iQ]*nsub+isub0,iG*nsub+isub1]
            count+=1
    for i1 in range(VQ_F_shape[1]):
        for i3 in range(VQ_F_shape[3]):
            for i0 in range(VQ_F_shape[0]):
                for i2 in range(VQ_F_shape[2]):
                    rhoQ_up_VQ[i1,i3]+=rhoQ_up[i0,i2]*VQ_F[i0,i1,i2,i3]
    rhoQ_up_VQ=rhoQ_up_VQ/epsilon_d*Uvalue
    count=0
    for iG in range(num_G):
        if iGmQ[iG,iQ]<500:
            for isub0 in range(nsub):
                for isub1 in range(nsub):
                    paramF_up[:,iGmQ[iG,iQ]*nsub+isub1,iG*nsub+isub0]=rhoQ_up_VQ[:,count,isub0,isub1]
            count+=1
    count=0
    for iG in range(num_G):
        if iGmQ[iG,iQ]<500:
            for isub0 in range(nsub):
                for isub1 in range(nsub):
                    rhoQ_up_T[:,count,isub1,isub0]=rho_up_subset[:,iGmQ[iG,iQ]*nsub+isub0,iG*nsub+isub1]
            count+=1
    for i1 in range(VQ_F_shape[1]):
        for i3 in range(VQ_F_shape[3]):
            for isub0 in range(nsub):
                for isub1 in range(nsub):
                    EFock_up+=rhoQ_up_VQ[i1,i3,isub0,isub1]*rhoQ_up_T[i1,i3,isub0,isub1]
    EFock=EFock_up
    
    return EHartree,EFock,paramH,paramF_up

def mean_field_mpi(kvec,Gvec,paramH0,paramF0_up,VQ_H,VQ_F,n):

    ########################### MPI ###########################
    # If the size of array to be parallelized *can not be divided* by the number of cores,
    # the array will be diveded into subsets with 2 types of size:
    # {num_more} subsets have {subset_size+1} elements, lefted are the subsets with {subset_size} elements
    subset_size,num_more=divmod(kvec.shape[0],size)
    k_subsets=[kvec[i*(subset_size+1):(i+1)*(subset_size+1)] if i < num_more else kvec[i*subset_size+num_more:(i+1)*subset_size+num_more] for i in range(size)] # divide kvec into size subsets
    paramH0_subsets=[paramH0[i*(subset_size+1):(i+1)*(subset_size+1)] if i < num_more else paramH0[i*subset_size+num_more:(i+1)*subset_size+num_more] for i in range(size)] # divide paramH0 into size subsets
    paramF0_up_subsets=[paramF0_up[i*(subset_size+1):(i+1)*(subset_size+1)] if i < num_more else paramF0_up[i*subset_size+num_more:(i+1)*subset_size+num_more] for i in range(size)] # divide paramF0_up into size subsets
    k_subset=comm.scatter(k_subsets,root=0)
    paramH0_subset=comm.scatter(paramH0_subsets,root=0)
    paramF0_up_subset=comm.scatter(paramF0_up_subsets,root=0)
    ###########################################################

    # bare Hamiltonian
    Hk0=make_hamk0(k_subset,Gvec,Uonsite,n)
    Hk0_up=Hk0+paramH0_subset+paramF0_up_subset
    Eup,Vup=eigh(Hk0_up)
    Ek=Eup

    ########################### MPI ###########################
    Ek_gather=comm.gather(Ek,root=0)
    comm.barrier()
    if rank==0:
        Ek=concatenate(Ek_gather)
        Ef=chemical_potential(Ek,filling,output=False)
    else:
        Ef=None
    Ef=comm.bcast(Ef,root=0)
    # print(Ef)
    ###########################################################

    # occupied states
    Vup*=expand_dims(heaviside(Ef-Eup,1.),1)
    Eg0_up=sum(trace(transpose(conj(Vup),[0,2,1])@Hk0@Vup,axis1=-2,axis2=-1))/num_kpt
    Eg0=Eg0_up

    # # round cutoff
    # kpG=k_subset[:,None,:]+Gvec[None,:,:]
    # kpG_cutoff_index=where(norm(kpG,axis=-1)>norm(bvec0)*sqrt(3)/4*nD)
    # k_index=kpG_cutoff_index[0]
    # for isub in range(nsub):
    #     G_index=kpG_cutoff_index[1]*nsub+isub
    #     Vup[k_index,G_index]=0.
    #     Vdn[k_index,G_index]=0.
    ########################### MPI ###########################
    Eg0_gather=comm.gather(Eg0,root=0)
    Vup_gather=comm.gather(Vup,root=0)
    comm.barrier()
    if rank==0:
        Eg0=sum(Eg0_gather)
        Vup=concatenate(Vup_gather)
        # density matrices
        rho_up=conj(Vup)@transpose(Vup,[0,2,1])
        rhoH=sum(rho_up,axis=0)/num_kpt
    else:
        rho_up=None
        rhoH=None
    rho_up=comm.bcast(rho_up,root=0)
    # if init_type==0:
    #     rho_up=kill_sigma_z(rho_up)
    rhoH=comm.bcast(rhoH,root=0)
    rho_up_subsets=[rho_up[i*(subset_size+1):(i+1)*(subset_size+1)] if i < num_more else rho_up[i*subset_size+num_more:(i+1)*subset_size+num_more] for i in range(size)] # divide rho_up into size subsets
    rho_up_subset=comm.scatter(rho_up_subsets,root=0)
    ###########################################################

    EHartree=0.
    EFock=0.
    paramH=zeros((num_G*nsub,num_G*nsub),'complex')
    paramF_up=zeros((k_subset.shape[0],num_G*nsub,num_G*nsub),'complex')
    for iQ in range(num_G):
        EHartree_iQ,EFock_iQ,paramH_iQ,paramF_up_iQ=mean_field_iQ(iQ,k_subset,rhoH,rho_up,rho_up_subset,VQ_H[iQ],VQ_F[f'{iQ}'])
        EHartree+=EHartree_iQ
        EFock+=EFock_iQ
        paramH+=paramH_iQ
        paramF_up+=paramF_up_iQ

    ########################### MPI ###########################
    paramF_up_gather=comm.gather(paramF_up,root=0)
    EFock_gather=comm.gather(EFock,root=0)
    comm.barrier()
    if rank==0:
        paramF_up=concatenate(paramF_up_gather,axis=0)
        EFock=sum(EFock_gather)
        paramH=expand_dims(paramH,0)
        paramF_up/=-num_kpt
        EHartree/=2.
        EFock/=-2.*num_kpt**2
        # total energy
        Etot=Eg0+EHartree+EFock
    else:
        paramH=None
        paramF_up=None
        EHartree=None
        EFock=None
        Eg0=None
        Etot=None
        Ek=None
    paramH=comm.bcast(paramH,root=0)
    paramF_up=comm.bcast(paramF_up,root=0)
    EHartree=comm.bcast(EHartree,root=0)
    EFock=comm.bcast(EFock,root=0)
    Eg0=comm.bcast(Eg0,root=0)
    Etot=comm.bcast(Etot,root=0)
    Ek=comm.bcast(Ek,root=0)
    ###########################################################

    return  Etot,Eg0,EHartree,EFock,paramH,paramF_up,rho_up,Ek

@njit
def kill_sigma_z(mat):
    for ik in range(mat.shape[0]):
        for iG0 in range(num_G):
            for iG1 in range(num_G):
                z00=mat[ik,iG0*nsub+0,iG1*nsub+0]
                z11=mat[ik,iG0*nsub+1,iG1*nsub+1]
                # z0011=.0
                z0011=(z00+z11)/2.
                mat[ik,iG0*nsub+0,iG1*nsub+0]=z0011
                mat[ik,iG0*nsub+1,iG1*nsub+1]=z0011
    return mat

def iteration_mpi(Nmax,convthr,mixing,paramH0,paramF0_up,n=order_kin,init_type=1):
    subset_size,num_more=divmod(kvec.shape[0],size)
    k_subsets=[kvec[i*(subset_size+1):(i+1)*(subset_size+1)] if i < num_more else kvec[i*subset_size+num_more:(i+1)*subset_size+num_more] for i in range(size)] # divide kvec into size subsets
    k_subset=comm.scatter(k_subsets,root=0)
    VQ_H=zeros((num_G),'complex')
    VQ_F={}
    for iQ in range(num_G):
        VQ_H[iQ]=VQ(Gvec[iQ])
        VQ_F[f'{iQ}']=VQ_mat(kvec,k_subset,iQ)
    Eg=zeros(Nmax,'float')
    EHartree=zeros(Nmax,'float')
    EFock=zeros(Nmax,'float')
    for it in range(Nmax):
        if it==0:
            paramHi=paramH0
            paramFi_up=paramF0_up
        Etot,Eg0t,EHartreet,EFockt,paramH,paramF_up,rhof_up,Ek=mean_field_mpi(kvec,Gvec,paramHi,paramFi_up,VQ_H,VQ_F,n)        
        if it==0 and rank==0:
            print('Start iteration')
            print('+----------------------------------------------------------------------------------+')
            print('Niter\tEtot\t\tdE\t\tEHartree\tEFock\t\tEkin')         
        Eg[it]=real(Etot)
        EHartree[it]=real(EHartreet)
        EFock[it]=real(EFockt)
        if it==0:
            rhoi_up=rhof_up
        if it>0 and rank==0:
            print(f'{it:3d}\t{Eg[it]:13.10f}\t{Eg[it]-Eg[it-1]:13.10f}\t{EHartree[it]:13.10f}\t{EFock[it]:13.10f}\t{Eg[it]-EHartree[it]-EFock[it]:13.10f}')
        if (it>1 and it<Nmax): 
            if (abs(Eg[it]-Eg[it-1])<convthr):
                if rank==0:
                    print('Niter\tEtot\t\tdE\t\tEHartree\tEFock\t\tEkin')         
                    print('+----------------------------------------------------------------------------------+')
                    print('Convergence is reached!')
                    print('+----------------------------------------------------------------------------------+')
                paramHsave=paramH
                paramFsave_up=paramF_up
                break
            if (Eg[it]-Eg[it-1])>0 and it>3 and init_type==3:
                if rank==0:
                    print('Niter\tEtot\t\tdE\t\tEHartree\tEFock\t\tEkin')         
                    print('+----------------------------------------------------------------------------------+')
                    print('!!!!!!!!!!!!!!!!!!!!!!!')
                    print('Ground state is missed!')
                    print('!!!!!!!!!!!!!!!!!!!!!!!')
                    print('+----------------------------------------------------------------------------------+')
                paramHsave=paramHi
                paramFsave_up=paramFi_up
                Etot=Eg[it-1]
                EHartreet=EHartree[it-1]
                EFockt=EFock[it-1]
                Ek=Eki
                break
            if abs(abs(Eg[it]-Eg[it-1])-abs(Eg[it-1]-Eg[it-2]))<1e-12:
                if rank==0:
                    print('Niter\tEtot\t\tdE\t\tEHartree\tEFock\t\tEkin')         
                    print('+----------------------------------------------------------------------------------+')
                    print('!!!!!!!!!!!!!')
                    print('Endless loop!')
                    print('!!!!!!!!!!!!!')
                    print('+----------------------------------------------------------------------------------+')
                paramHsave=paramH
                paramFsave_up=paramF_up
                break
        if (it==Nmax-1 and abs(Eg[it]-Eg[it-1])>convthr):
            if rank==0:
                print('Niter\tEtot\t\tdE\t\tEHartree\tEFock\t\tEkin')         
                print('+----------------------------------------------------------------------------------+')
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('Convergence is NOT reached!')
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('delta E is ', Eg[it]-Eg[it-1])
                print('+----------------------------------------------------------------------------------+')
            paramHsave=paramH
            paramFsave_up=paramF_up
        if it>0:    
            paramHi=paramH*mixing+(1-mixing)*paramHi
            paramFi_up=paramF_up*mixing+(1-mixing)*paramFi_up
            rhoi_up=rhof_up
            Eki=Ek.copy()
        if Nmax==1 or Nmax==2:
            paramHsave=paramH            
            paramFsave_up=paramF_up
    return paramHsave,paramFsave_up,real(Etot),real(EHartreet),real(EFockt),Ek

@njit
def init_cond(init_type,gap=-.001):    
    paramH0=zeros((num_kpt,num_G*nsub,num_G*nsub),'complex')
    paramF0_up=zeros((num_kpt,num_G*nsub,num_G*nsub),'complex')
    if init_type==0:
        for iQ in range(num_G):
            for iG in range(num_G):
                if iGpQ[iG,iQ]<500:
                    for isub0 in range(nsub):
                        pass
    elif init_type==1:
        for iQ in range(num_G):
            # if 1:
            if abs(norm(Gvec[iQ])-norm(bvec1))<1e-6:
                for iG in range(num_G):
                    if iGpQ[iG,iQ]<500:
                        for isub0 in range(nsub):
                            # paramH0[:,iGpQ[iG,iQ]*nsub+isub0,iG*nsub+isub0]=.0005*(-1)**isub0
                            paramF0_up[:,iGpQ[iG,iQ]*nsub+(1-isub0),iG*nsub+isub0]=gap*(-1)**isub0
                            # paramF0_up[:,iGpQ[iG,iQ]*nsub+isub0,iG*nsub+isub0]=-.001*(-1)**isub0

    elif init_type==2:
        for iQ in range(num_G):
            # if 1:
            if abs(norm(Gvec[iQ])-norm(bvec1))<1e-6:
                for iG in range(num_G):
                    if iGpQ[iG,iQ]<500:
                        for isub0 in range(nsub):
                            # paramH0[:,iGpQ[iG,iQ]*nsub+isub0,iG*nsub+isub0]=.0005*(-1)**isub0
                            paramF0_up[:,iGpQ[iG,iQ]*nsub+(1-isub0),iG*nsub+isub0]=gap*(-1)**isub0*1.j
                            # paramF0_up[:,iGpQ[iG,iQ]*nsub+isub0,iG*nsub+isub0]=-.001*(-1)**isub0

    elif init_type==3:
        for iQ in range(num_G):
            # if 1:
            if abs(norm(Gvec[iQ])-norm(bvec1))<1e-6:
                for iG in range(num_G):
                    if iGpQ[iG,iQ]<500:
                        for isub0 in range(nsub):
                            # paramH0[:,iGpQ[iG,iQ]*nsub+isub0,iG*nsub+isub0]=.0005*(-1)**isub0
                            paramF0_up[:,iGpQ[iG,iQ]*nsub+isub0,iG*nsub+isub0]=gap*(-1)**isub0
                            # paramF0_up[:,iGpQ[iG,iQ]*nsub+isub0,iG*nsub+isub0]=-.001*(-1)**isub0

    elif init_type==4:
        for iQ in range(num_G):
            # if 1:
            if abs(norm(Gvec[iQ])-norm(bvec1))<1e-6:
                for iG in range(num_G):
                    if iGpQ[iG,iQ]<500:
                        for isub0 in range(nsub):
                            # paramH0[:,iGpQ[iG,iQ]*nsub+isub0,iG*nsub+isub0]=.0005*(-1)**isub0
                            paramF0_up[:,iGpQ[iG,iQ]*nsub+isub0,iG*nsub+isub0]=gap
                            # paramF0_up[:,iGpQ[iG,iQ]*nsub+isub0,iG*nsub+isub0]=-.001*(-1)**isub0

    return paramH0,paramF0_up

# def bandstructure(kvec,Gvec,Ek=zeros([0,0]),init_cond=0,n=1,Etot='',Kp=False):
#     Ek0=eigvalsh(make_hamk0(kvec,Gvec,Uonsite,n))
#     plt.rcParams['font.size'] = 16
#     fig,(ax,ax_p)=subplots(1,2,figsize=(12,6))
#     for nband in range(Ek0.shape[1]):
#         if nband==0:
#             Ef0=chemical_potential(Ek0,filling/2)
#             Ek0_plot=Ek0[band_index]
#             ax.plot(kline,Ek0_plot[:,nband]-Ef0,'r--',label='Non-interacting')
#         else:
#             ax.plot(kline,Ek0_plot[:,nband]-Ef0,'r--')
#     for nband in range(Ek.shape[1]):
#         if nband==0:
#             Ef=chemical_potential(Ek,filling)
#             Ek_plot=Ek[band_index]
#             ax.plot(kline,Ek_plot[:,nband]-Ef,'b-',label='Hartree-Fock')
#         else:
#             ax.plot(kline,Ek_plot[:,nband]-Ef,'b-')
#     ax.legend(loc='upper right')
#     ax.set_title(f'Ls={Ls:.1f} E_tot={Etot:.6f}eV',pad=10)
#     ax.set_xticks(xticks)
#     ax.set_xticklabels(xlabels)
#     ax.axhline(y=0.,ls='--',lw=1.,c='k',zorder=-5)
#     ax.set_xlim((kline[0],kline[-1]))
#     # ax.set_ylim((Ek0.min(),Ek0.max()))
#     ax.set_ylabel('Energy (eV)')

#     for nband in range(Ek0.shape[1]):
#         if nband==0:
#             Ef0=chemical_potential(Ek0,filling/2)
#             Ek0_plot=Ek0[band_index]
#             ax_p.plot(kline,Ek0_plot[:,nband]-Ef0,'r--',label='Non-interacting')
#         else:
#             ax_p.plot(kline,Ek0_plot[:,nband]-Ef0,'r--')
#     for nband in range(Ek.shape[1]):
#         if nband==0:
#             Ef=chemical_potential(Ek,filling)
#             Ek_plot=Ek[band_index]
#             ax_p.plot(kline,Ek_plot[:,nband]-Ef,'b-',label='Hartree-Fock')
#         else:
#             ax_p.plot(kline,Ek_plot[:,nband]-Ef,'b-')
#     ax_p.legend(loc='upper right')
#     ax_p.set_title(f'Ls={Ls:.1f} E_tot={Etot:.6f}eV',pad=10)
#     ax_p.set_xticks(xticks)
#     ax_p.set_xticklabels(xlabels)
#     ax_p.axhline(y=0.,ls='--',lw=1.,c='k',zorder=-5)
#     ax_p.set_xlim((kline[0],kline[-1]))
#     ax_p.set_ylim((-.5,.5))
#     ax_p.set_ylabel('Energy (eV)')

#     fig.tight_layout()
#     fig.savefig(f"{path}/lat{lattice}_Ls{Ls:.1f}_nk{num_k1}_nD{nD}_D{Dfield:.3f}_eps{epsilon_d:.2f}_order{n:.0f}_gap{init_type}_filling{filling_e:.3f}.pdf")
#     fig.savefig(f"order{n:.0f}_fl{filling_e:.3f}_nD{nD}_D{Dfield:.3f}_eps{epsilon_d:.2f}_init{init_type}.pdf")

def bandstructure(kvec,Gvec,Ek=zeros([0,0]),init_cond=0,n=1,Etot='',gap_wigner=0,Kp=False):
    Ek0=eigvalsh(make_hamk0(kvec,Gvec,Uonsite,n))
    plt.rcParams['font.size'] = 16
    if Kp==False:
        fig,ax=subplots(1,1,figsize=(8,6))
    else:
        fig,(ax,ax_p)=subplots(1,2,figsize=(12,6))
    for nband in range(Ek0.shape[1]):
        if nband==0:
            Ef0=chemical_potential(Ek0,filling)
            Ek0_plot=Ek0[band_index]
            ax.plot(kline,Ek0_plot[:,nband]-Ef0,'r--',label='Non-interacting')
        else:
            ax.plot(kline,Ek0_plot[:,nband]-Ef0,'r--')
    for nband in range(Ek.shape[1]):
        if nband==0:
            Ef=chemical_potential(Ek,filling)
            Ek_plot=Ek[band_index]
            ax.plot(kline,Ek_plot[:,nband]-Ef,'b-',label='Hartree-Fock')
        else:
            ax.plot(kline,Ek_plot[:,nband]-Ef,'b-')
    ax.legend(loc='upper right')
    # ax.set_title(f'Ls={Ls0:.1f}$\AA$ E_tot={Etot:.5f}eV gap={gap_wigner*1e3:.2f}meV',pad=8,fontsize=12)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.axhline(y=0.,ls='--',lw=1.,c='k',zorder=-5)
    ax.set_xlim((kline[0],kline[-1]))
    ax.set_ylim((-.5,.5))
    ax.set_ylabel('Energy (eV)')

    if Kp==True:
        for nband in range(Ek0.shape[1]):
            if nband==0:
                Ef0=chemical_potential(Ek0,filling)
                Ek0_plot=Ek0[band_index_p]
                ax_p.plot(kline,Ek0_plot[:,nband]-Ef0,'r--',label='Non-interacting')
            else:
                ax_p.plot(kline,Ek0_plot[:,nband]-Ef0,'r--')
        for nband in range(Ek.shape[1]):
            if nband==0:
                Ef=chemical_potential(Ek,filling)
                Ek_plot=Ek[band_index_p]
                ax_p.plot(kline,Ek_plot[:,nband]-Ef,'b-',label='Hartree-Fock')
            else:
                ax_p.plot(kline,Ek_plot[:,nband]-Ef,'b-')
        ax_p.legend(loc='upper right')
        ax_p.set_title(f'Ls={Ls0:.1f}$\AA$ E_tot={Etot:.5f}eV gap={gap_wigner*1e3:.2f}meV',pad=8,fontsize=12)
        ax_p.set_xticks(xticks)
        ax_p.set_xticklabels(xlabels_p)
        ax_p.axhline(y=0.,ls='--',lw=1.,c='k',zorder=-5)
        ax_p.set_xlim((kline[0],kline[-1]))
        ax_p.set_ylim((-.3,.3))
        ax_p.set_ylabel('Energy (eV)')

    fig.tight_layout()
    fig.savefig(f"{path}/lat{lattice}_Ls{Ls0:.1f}_nk{num_k1}_nD{nD}_D{Dfield:.3f}_eps{epsilon_d:.2f}_order{n:.0f}_gap{init_type}_filling{filling_e:.3f}.pdf")
    fig.savefig(f"N{n:.0f}_Ls{Ls0:.1f}_nk{num_k1}_nD{nD}_D{Dfield:.3f}_eps{epsilon_d:.2f}_init{init_type}.pdf")


if __name__=='__main__':
    if size>num_kpt**2:
        if rank==0:
            print('')
            print('Too many cores!!!!!')
            print('')
            print(f'According to the system size, maximum number of cores is {num_kpt**2}')
            print('')
        sys.exit(0)

    # Nmax=1
    # Nmax=2
    # Nmax=3
    if init_type==0 or init_type==4:
        Nmax=300
        convthr=1e-5
    else:
        Nmax=5
        convthr=1e-7

    # convthr=1e-7
    # mixing=1
    mixing=.7
    paramH0,paramF0_up=init_cond(init_type)

    time1=time.time()
    if rank==0:
        print('+==================================================================================+')
        print(f'+                        Running in parallel on {size:4d} CPUs                          +')
        print('+                              Calculating HF parameters                           +')
        print(f'+                               Ls={Ls0:5.1f} nk={num_k1:2d} nD={nD:2d}                               +')
        print('+==================================================================================+')
    time1=time.time()
    paramH0,paramF0_up,Etot,EHartree,EFock,Ek=iteration_mpi(Nmax,convthr,mixing,paramH0,paramF0_up,n=order_kin,init_type=init_type)
    if rank==0:
        # gap=sort(Ek,-1)[0,Ek.shape[1]//2]-sort(Ek,-1)[0,Ek.shape[1]//2-1]
        gap_wigner=0
        # gap_wigner=min(sort(Ek,-1)[:,(num_G+1)])-max(sort(Ek,-1)[:,(num_G)])
        # print(f'gap_Gamma={min(sort(Ek,-1)[:,(num_G)])-max(sort(Ek,-1)[:,(num_G-1)])}')
        # print(f'argmin={np.argmin(sort(Ek,-1)[:,(num_G)])} argmax={np.argmax(sort(Ek,-1)[:,(num_G-1)])}')
        if lattice==1:
            savez(f'{path}/paramHF_rect_Ls{Ls0:.1f}_nk{num_k1}_nD{nD}_D{Dfield:.3f}_eps{epsilon_d:.2f}_gap{init_type}_fl{filling:.3f}_meff{meff}_order{order_kin:.1f}.npz',paramH=paramH0,paramF_up=paramF0_up,paramF_dn=0,Etot=Etot,EHartree=EHartree,EFock=EFock,gap=gap_wigner)
            # savez(f'./EHF_rect_Ls{Ls0:.1f}_nk{num_k1}_nD{nD}_D{Dfield:.3f}_eps{epsilon_d:.2f}_fl{filling:.3f}_meff{meff}_order{order_kin:.1f}.npz',Etot=Etot,EHartree=EHartree,EFock=EFock,Ekin=Etot-EHartree-EFock)
        elif lattice==2:
            savez(f'{path}/paramHF_tri_Ls{Ls0:.1f}_nk{num_k1}_nD{nD}_D{Dfield:.3f}_eps{epsilon_d:.2f}_gap{init_type}_fl{filling:.3f}_meff{meff}_order{order_kin:.1f}.npz',paramH=paramH0,paramF_up=paramF0_up,paramF_dn=0,Etot=Etot,EHartree=EHartree,EFock=EFock,gap=gap_wigner)
            # savez(f'./EHF_tri_Ls{Ls0:.1f}_nk{num_k1}_nD{nD}_D{Dfield:.3f}_eps{epsilon_d:.2f}_fl{filling:.3f}_meff{meff}_order{order_kin:.1f}.npz',Etot=Etot,EHartree=EHartree,EFock=EFock,Ekin=Etot-EHartree-EFock)
        savez(f'E_Ls{Ls0:.1f}_nk{num_k1}_nD{nD}_D{Dfield:.3f}_eps{epsilon_d:.2f}_gap{init_type}_fl{filling:.3f}_order{order_kin:.1f}.npz',Etot=Etot,EHartree=EHartree,EFock=EFock,gap=gap_wigner,Ls=Ls0)
        Ef=chemical_potential(Ek,filling,output=False)
        # bandstructure(kvec,Gvec,Ek,init_cond=init_type,n=order_kin,Etot=Etot,gap_wigner=gap_wigner)
        bandstructure(kvec,Gvec,Ek,init_cond=init_type,n=order_kin,Etot=Etot,gap_wigner=gap_wigner,Kp=False)
        time2=time.time()
        print('')
        print(f'HF calculation time elapse is {time2-time1:13.6f}s')
        print('')
        # print(f'The global band gap (upper k-index {sort(Ek,-1)[:,int(filling)].argmin()}) (lower k-index {sort(Ek,-1)[:,int(filling)-1].argmax()}) is (eV): {sort(Ek,-1)[:,int(filling)].min()-sort(Ek,-1)[:,int(filling)-1].max()}')
        # print(f'The band gap at Gamma point (k-index {0}) is (eV): {gap:.8f}')
        # print(f'The band gap  is (meV): {gap_wigner*1e3:.8f}')
        print(f'The Fermi energy is (eV): {Ef:.8f}')
        print(f'The converged ground state energy is (eV): {Etot:.8f}')
        print('+----------------------------------------------------------------------------------+')
        # print(band_index)
        # fig,ax=subplots(figsize=(12,12))
        # Hk0=make_hamk0(kvec,Gvec)
        # Hk0_up=Hk0+paramH0+paramF0_up
        # ax.matshow(abs(Hk0_up[0,:,:]))
        # print(abs(Hk0_up[0,:,:]))
        # fig.savefig(f'./Hk0_up.pdf')
        # fig,ax=subplots(figsize=(12,12))
        # ax.matshow(abs(Hk0[0,:,:]))
        # # print(abs(Hk0[0,:,:]))
        # fig.savefig(f'./Hk0.pdf')
        # fig,ax=subplots(figsize=(12,12))
        # ax.matshow(abs(paramH0[0,:,:]))
        # # print(abs(paramH0[0,:,:]))
        # fig.savefig(f'./paramH0.pdf')
        # fig,ax=subplots(figsize=(12,12))
        # ax.matshow(abs(paramF0_up[0,:,:]))
        # # print(abs(paramF0_up[0,:,:]))
        # fig.savefig(f'./paramF0_up.pdf')
        # # for ik in range(num_kpt):
        # #     print(ik,norm(kvec[ik]))
        # #     print(Hk0_up[ik])
        # Hk0=make_hamk0(kvec,Gvec,n=1)
        # Ek00up=eigvalsh(Hk0)
        # Ek_up=Ek[...,:Ek.shape[-1]//2]-Ef
        # color_pm=(Ek_up-Ek00up[...,:Ek.shape[-1]//2])[...,Ek_up.shape[-1]//2]
        # color_p=where(color_pm>=0)[0]
        # color_m=where(color_pm<0)[0]
        # # Ek_dn=Ek[...,Ek.shape[-1]//2:]-Ef
        # Ek_plot=Ek_up[...,Ek_up.shape[-1]//2]
        # Ek_plot-=Ek_plot.min()
        # kx=kvec[:,0].reshape(num_k1,num_k1)
        # kx=np.roll(kx,num_k1//2,axis=0)
        # kx=np.roll(kx,num_k1//2,axis=1)
        # ky=kvec[:,1].reshape(num_k1,num_k1)
        # ky=np.roll(ky,num_k1//2,axis=0)
        # ky=np.roll(ky,num_k1//2,axis=1)
        # Ek_plot=Ek_plot.reshape(num_k1,num_k1)
        # Ek_plot=np.roll(Ek_plot,num_k1//2,axis=0)
        # Ek_plot=np.roll(Ek_plot,num_k1//2,axis=1)
        # fig,ax=subplots(figsize=(12,12))
        # ax.scatter(kvec[color_p,0],kvec[color_p,1],Ek_plot[color_p]*1000,'r',zorder=5)
        # ax.scatter(kvec[color_m,0],kvec[color_m,1],Ek_plot[color_m]*1000,'b')
        # # contour=ax.contourf(kx,ky,Ek_plot,levels=50,cmap='inferno')
        # # fig.colorbar(contour, ax=ax)
        # ax.set_aspect(1)
        # # print(Ek_up[:,nband//2].reshape(num_k1,num_k1))
        # fig.savefig(f'./Ek.pdf')