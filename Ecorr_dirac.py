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

lattice=2 # 1=rectangular 2=triangular
Ls0=float(sys.argv[1])
num_k1 = int(sys.argv[2]) # 
nD = int(sys.argv[3]) # 
Dfield = float(sys.argv[4])
epsilon_d = float(sys.argv[5])
N1=int(sys.argv[6])
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
meff=1.3 ## this is the effective mass at the band edge of the insulating substrate, in untis of electron mass.

Ecorr_cut=int(sys.argv[8]) # QuasiParticle band number cutoff
# npole=int(sys.argv[9]) # the number of poles in MPA model
# withZ=1 # whether to linearize self-energy
# w_max=float(sys.argv[10]) # max real part of sampling frequency in MPA model
# wp=float(sys.argv[11]) # the imaginary part of sampling frequency in MPA model
# wp=w_max/10. # the imaginary part of sampling frequency in MPA model
Ek_HF_shift=0. # the number of poles in MPA model
delta=1e-3 # infinitesimal in Green's function
delta_self=1e-3
a0=2.46 # A: graphene C-C bond length
d0=7.0 # A
hbar=6.582119569
c0=2.99792458
me=0.51099895
hbar2me= (hbar**2*c0**2/me)*1e-2  # this is \hbar^2/me, in units of eV*ang^2, me is the electron mass


if lattice==1:
### Rectangular lattice ###
    # r_xy=1.
    r_xy=3.89/3.20
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
    if not os.path.exists(f'{path}/epsilon'):
        os.mkdir(f'{path}/epsilon')

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
            if output:
                print('The filling factor is ', occ)
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

def bandstructure(kvec,Gvec,Ek=zeros([0,0]),Ek_QP=zeros([0,0]),iQP=Ecorr_cut,n=1):
    Ek0=eigvalsh(make_hamk0(kvec,Gvec,Uonsite,n))
    plt.rcParams['font.size'] = 12
    fig,ax=subplots(figsize=(6,6))
    for nband in range(Ek0.shape[1]):
        if nband==0:
            # Ef0=chemical_potential(Ek0,filling)
            Ek0=Ek0[band_index]
            Ef0=max(Ek0[:,Ek0.shape[1]//2])
            ax.plot(kline,Ek0[:,nband]-Ef0,'r--',label='Non-interacting')
        else:
            ax.plot(kline,Ek0[:,nband]-Ef0,'r--')
    for nband in range(Ek.shape[1]):
        if nband==0:
            # Ef=chemical_potential(Ek,filling)
            Ek=Ek[band_index]
            Ef=max(Ek[:,Ek.shape[1]//2])
            ax.plot(kline,Ek[:,nband]-Ef,'b-',label='Hartree-Fock')
        else:
            ax.plot(kline,Ek[:,nband]-Ef,'b-')
    for nband in range(Ek_QP.shape[1]):
        if nband==0:
            # Ef_QP=chemical_potential(Ek_QP,filling)
            Ek_QP=Ek_QP[band_index]
            Ef_QP=max(Ek_QP[:,Ek_QP.shape[1]//2])
            ax.plot(kline,Ek_QP[:,nband]-Ef_QP,'m-',label='$GW$')
        else:
            ax.plot(kline,Ek_QP[:,nband]-Ef_QP,'m-')

    ax.legend(loc='upper right')
    # ax.set_title(f'Ls={Ls0:.1f}$\AA$ E_tot={Etot:.5f}eV gap={gap_wigner*1e3:.2f}meV',pad=8,fontsize=12)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.axhline(y=0.,ls='--',lw=1.,c='k',zorder=-5)
    ax.set_xlim((kline[0],kline[-1]))
    ax.set_ylim((-.5,.5))
    # ax.set_ylim((-1,1))
    ax.set_ylabel('Energy (eV)')
    fig.tight_layout()
    fig.savefig(f"{path}/lat{lattice}_Ls{Ls:.1f}_nk{num_k1}_nD{nD}_D{Dfield:.3f}_iQP{iQP+1}_eps{epsilon_d:.1f}_order{n:.0f}_gap{init_type}_filling{filling_e:.3f}.pdf")
    # if FLorWC==1:
    #     fig.savefig(f"{path}/lat{lattice}_Ls{Ls:.1f}_nk{num_k1}_nD{nD}_D{Dfield:.3f}_iQP{iQP+1}_FL.pdf")
    # elif FLorWC==2:
    #     fig.savefig(f"{path}/lat{lattice}_Ls{Ls:.1f}_nk{num_k1}_nD{nD}_D{Dfield:.3f}_iQP{iQP+1}_WC.pdf") 
    # elif FLorWC==3:
    #     fig.savefig(f"{path}/lat{lattice}_Ls{Ls:.1f}_nk{num_k1}_nD{nD}_D{Dfield:.3f}_iQP{iQP+1}_WC_fm.pdf") 
    # else:
    #     fig.savefig(f"{path}/lat{lattice}_Ls{Ls:.1f}_nk{num_k1}_nD{nD}_D{Dfield:.3f}_order{n:.0f}_iQP{iQP+1}.pdf") 

################################### GW QP energy ###################################

@njit
def kqmap(ik_subset):
    ikq=zeros((num_kpt,num_kpt),'int')
    G0shift=zeros((num_kpt,num_kpt),'int')
    G1shift=zeros((num_kpt,num_kpt),'int')
    for ik in ik_subset:
        for iq in range(num_kpt):
            for iikq in range(num_kpt):
                for iG0 in [-1,0,1]:
                    for iG1 in [-1,0,1]:
                        dk=kvec[ik]+kvec[iq]-(kvec[iikq]+iG0*bvec0+iG1*bvec1)
                        if norm(dk)<1e-9:
                            ikq[ik,iq]=iikq
                            G0shift[ik,iq]=iG0
                            G1shift[ik,iq]=iG1
    return ikq,G0shift,G1shift

subset_size,num_more=divmod(num_kpt,size)
ik_subsets=[range(num_kpt)[i*(subset_size+1):(i+1)*(subset_size+1)] if i < num_more else range(num_kpt)[i*subset_size+num_more:(i+1)*subset_size+num_more] for i in range(size)] # divide kvec into size subsets
ik_subset=comm.scatter(ik_subsets,root=0)
ik_subset=array(ik_subset)
ikq,G0shift,G1shift=kqmap(ik_subset)
ikq_gather=comm.gather(ikq,root=0)
G0shift_gather=comm.gather(G0shift,root=0)
G1shift_gather=comm.gather(G1shift,root=0)
if rank==0:
    ikq=concatenate(ikq_gather).reshape((-1,num_kpt,num_kpt))
    G0shift=concatenate(G0shift_gather).reshape((-1,num_kpt,num_kpt))
    G1shift=concatenate(G1shift_gather).reshape((-1,num_kpt,num_kpt))
    ikq=np.sum(ikq,axis=0)
    G0shift=np.sum(G0shift,axis=0)
    G1shift=np.sum(G1shift,axis=0)
else:
    ikq=None
    G0shift=None
    G1shift=None
ikq=comm.bcast(ikq,root=0)
G0shift=comm.bcast(G0shift,root=0)
G1shift=comm.bcast(G1shift,root=0)

@njit
def k2kq(iq,Ek,Vk):
    '''
    The index order of Enk is (kpt,band) and  the index order of Enk is (kpt,coeff,band)
    '''
    num_G=nD**2
    Vkq_cut=zeros((num_kpt,nD+1,nD+1,nsub,num_G,nsub),'complex')
    Vkq_cut[:,:-1,:-1:,:,:,:]=Vk.reshape((num_kpt,nD,nD,nsub,num_G,nsub))
    Vkq=zeros(Vk.shape,'complex')
    Ekq=Ek[ikq[:,iq]]
    for ik in range(num_kpt):
        iG0=arange(nD)+G0shift[ik,iq]
        iG1=arange(nD)+G1shift[ik,iq]
        Vkq[ik]=Vkq_cut[ikq[ik,iq]][iG0,:,:,:,:][:,iG1,:,:,:].reshape((num_G*nsub,num_G*nsub))
    vqQ=zeros((num_G))
    for iQ0 in range(num_G):
        norm_qQ=norm(kvec[iq]+Gvec[iQ0])
        if norm_qQ>1e-12:
            vqQ[iQ0]=np.tanh(norm_qQ*ds)/norm_qQ
        else:
            vqQ[iQ0]=.0
    return Ekq,Vkq,vqQ*Uvalue/epsilon_d

@njit
def epsilon(nu,Ek,Vk,Ekq,Vkq,vqQ,Ef):
    '''
    Calculate dielectric matrix from directly numerical integration.
    '''
    num_G=nD**2
    ChiQQ=zeros((num_G,num_G),'complex')
    nu1=nu+1.j*delta
    nu2=nu-1.j*delta
    for ik in range(num_kpt):
        count_fermi_m=0
        for Eki in Ek[ik]:
            if Ef-Eki>0:
                count_fermi_m+=1
        count_fermi_n=0
        for Eki in Ekq[ik]:
            if Ef-Eki>0:
                count_fermi_n+=1
        for m in range(count_fermi_m):
            for n in range(count_fermi_n,nband):
                denom=nu1+Ek[ik,m]-Ekq[ik,n]
                lambQ=zeros((num_G),'complex')
                for iQ in range(num_G):
                    for iG in range(num_G):
                        if iGpQ[iG,iQ]<500:
                            for isub in range(nsub):
                                lambQ[iQ]+=conj(Vkq[ik,iGpQ[iG,iQ]*nsub+isub,n])*Vk[ik,iG*nsub+isub,m]
                for iQ0 in range(num_G):
                    for iQ1 in range(num_G):
                        ChiQQ[iQ0,iQ1]+=conj(lambQ[iQ0])*lambQ[iQ1]/denom
        for m in range(count_fermi_m,nband):
            for n in range(count_fermi_n):
                denom=nu2+Ek[ik,m]-Ekq[ik,n]
                lambQ=zeros((num_G),'complex')
                for iQ in range(num_G):
                    for iG in range(num_G):
                        if iGpQ[iG,iQ]<500:
                            for isub in range(nsub):
                                lambQ[iQ]+=Vk[ik,iG*nsub+isub,m]*conj(Vkq[ik,iGpQ[iG,iQ]*nsub+isub,n])
                for iQ0 in range(num_G):
                    for iQ1 in range(num_G):
                        ChiQQ[iQ0,iQ1]-=conj(lambQ[iQ0])*lambQ[iQ1]/denom
    ChiQQ*=1./num_kpt
    v_Chi=zeros((num_G,num_G),'complex')
    for iQ0 in range(num_G):
        for iQ1 in range(num_G):
            v_Chi[iQ0,iQ1]+=vqQ[iQ0]*ChiQQ[iQ0,iQ1]
    return eye(num_G)-v_Chi,v_Chi,ChiQQ

def eps_intgrand(nu,Ekup,Vkup,Ekqup,Vkqup,vqQ,Ef):
    Chi_V=epsilon(1.j*nu,Ekup,Vkup,Ekqup,Vkqup,vqQ,Ef)[1]
    return trace(sp.linalg.logm(eye(num_G)-Chi_V)+Chi_V).real
    # return (log(det(eye(num_G)-Chi_V))+trace(Chi_V)).real

def Ecorr_quad_iq(Ekup,Vkup,Ekqup,Vkqup,vqQ,Ef):
    # return sp.integrate.quad(eps_intgrand,-5,5,args=(iq,Ek,Vk,Ef),points=[.0])[0]
    return sp.integrate.quad(eps_intgrand,-np.inf,np.inf,args=(Ekup,Vkup,Ekqup,Vkqup,vqQ,Ef))[0]

def Ecorr_mpi(Ekup,Vkup,Ef):
    '''
    Gather the correlation energy at all cores to MPI_RANK_0. (Calculate Sigma_kn before MPI_GATHER)
    '''
    ########################### MPI ###########################
    # If the size of array to be parallelized *can not be divided* by the number of cores,
    # the array will be diveded into subsets with 2 types of size:
    # {num_more} subsets have {subset_size+1} elements, lefted are the subsets with {subset_size} elements
    iq_zip=[iq for iq in range(num_kpt)]
    subset_size,num_more=divmod(len(iq_zip),size)
    iq_subsets=[iq_zip[i*(subset_size+1):(i+1)*(subset_size+1)] if i < num_more else iq_zip[i*subset_size+num_more:(i+1)*subset_size+num_more] for i in range(size)] # divide kvec into size subsets
    iq_subset=comm.scatter(iq_subsets,root=0)
    ###########################################################
    Ecorr_iq=.0
    for count,iq in enumerate(iq_subset):
        if rank==0:
            print(f'{count+1}/{len(iq_subset)}',end='\t')
            # print(f'Calculating correlation energy: ({count+1}/{len(iq_subset)}) on MPI_RANK_0')
            # print('')
        time5=time.time()
        Ekqup,Vkqup,vqQ=k2kq(iq,Ekup,Vkup)
        Ecorr_iq_tmp=Ecorr_quad_iq(Ekup,Vkup,Ekqup,Vkqup,vqQ,Ef)
        if rank==0:
            # print(f'Correlation energy of iq={iq} is {Ecorr_iq_tmp/4./pi}eV')
            print(f'{Ecorr_iq_tmp/4./pi:11.8f}',end='\t')
        Ecorr_iq+=Ecorr_iq_tmp
        if rank==0:
            time6=time.time()
            print(f'{time6-time5:13.6f}s')
            # print('')
            # print(f'Correlation energy time elapse is {time6-time5:13.6f}s')
            # print('+----------------------------------------------------------------------------------+')
    ########################### MPI ###########################
    Ecorr_iq_gather=comm.gather(Ecorr_iq,root=0)
    comm.barrier()
    if rank==0:
        Ecorr_iq=array(Ecorr_iq_gather)
        Ecorr=sum(Ecorr_iq,0)
        Ecorr*=1./num_kpt/4./pi
    else:
        Ecorr=None
    Ecorr=comm.bcast(Ecorr,root=0)
    ###########################################################
    
    return Ecorr

if size>num_kpt**2:
    if rank==0:
        print('')
        print('Too many cores!!!!!')
        print('')
        print(f'According to the system size, maximum number of cores is {num_kpt**2}')
        print('')
    sys.exit(0)


if lattice==1:
    data=load(f'{path}/paramHF_rect_Ls{Ls:.1f}_nk{num_k1}_nD{nD}_D{Dfield:.3f}_eps{epsilon_d:.2f}_gap{init_type}_fl{filling:.3f}_meff{meff}_order{order_kin:.1f}.npz')
elif lattice==2:
    data=load(f'{path}/paramHF_tri_Ls{Ls:.1f}_nk{num_k1}_nD{nD}_D{Dfield:.3f}_eps{epsilon_d:.2f}_gap{init_type}_fl{filling:.3f}_meff{meff}_order{order_kin:.1f}.npz')
if data['paramH'].shape[0]==1:
    paramH0=data['paramH'].repeat(num_kpt,axis=0)
else:
    paramH0=data['paramH']
paramF0_up=data['paramF_up']
paramF0_dn=data['paramF_dn']
if rank==0:
    print('+==================================================================================+')
    print(f'+                        Running in parallel on {size:4d} CPUs                          +')
    print('+                              HF parameters loaded                                +')
    print(f'+                               Ls={Ls:5.1f} nk={num_k1:2d} nD={nD:2d}                               +')
    print('+==================================================================================+')

    
Hk0=make_hamk0(kvec,Gvec,Uonsite,N1)
Hk0_up=Hk0+paramH0+paramF0_up
# Hk0_dn=Hk0+paramH0+paramF0_dn
Ek00up=eigvalsh(Hk0)
# Ek00dn=eigvalsh(Hk0)
Eup,Vup=eigh(Hk0_up)
# Edn,Vdn=eigh(Hk0_dn)
Ek00=Ek00up
Ek=Eup
Ef0=chemical_potential(Ek00,filling)
Ef=chemical_potential(Ek,filling)
if rank==0:
    print(f'Ef0 = {Ef0}')
    print(f'Ef = {Ef}')
Ek=Eup
Vk=Vup
f'{path}/Ek_QP_tri_Ls{Ls:.1f}_nk{num_k1}_nD{nD}_D{Dfield:.3f}_eps{epsilon_d:.2f}_fl{filling:.3f}_meff{meff}_order{order_kin:.1f}.npz'
Ek_corr=Ek.copy()
if Ecorr_cut!=0:
    if lattice==1:
        data_GW=load(f'{path}/Ek_QP_rect_Ls{Ls:.1f}_nk{num_k1}_nD{nD}_D{Dfield:.3f}_eps{epsilon_d:.2f}_gap{init_type}_fl{filling:.3f}_meff{meff}_order{order_kin:.1f}.npz')
    elif lattice==2:
        data_GW=load(f'{path}/Ek_QP_tri_Ls{Ls:.1f}_nk{num_k1}_nD{nD}_D{Dfield:.3f}_eps{epsilon_d:.2f}_gap{init_type}_fl{filling:.3f}_meff{meff}_order{order_kin:.1f}.npz')
    Ek_QP=data_GW['Ek_QP']
    Ek_corr[:,nband//2-Ecorr_cut:nband//2+Ecorr_cut]=Ek_QP[:,nband//2-Ecorr_cut:nband//2+Ecorr_cut]
    if rank==0:
        print('+==================================================================================+')
        print(f'+                        Running in parallel on {size:4d} CPUs                          +')
        print('+                             GW corrections loaded                                +')
        print(f'+                               Ls={Ls:5.1f} nk={num_k1:2d} nD={nD:2d}                               +')
        print('+==================================================================================+')



Ef_corr=chemical_potential(Ek_corr,filling,output=False)
if rank==0:
    print(f'Ef_corr = {Ef_corr}')

# if rank==0:
#     bandstructure(kvec,Gvec,Ek,2)


if rank==0:
    print('+==================================================================================+')
    print('Caluculating correlation energy')
    print('iq\tEcorr')         
    # print('')
time1=time.time()
Ecorr=Ecorr_mpi(Ek_corr,Vk,Ef_corr)
comm.barrier()
time2=time.time()
# Ecorr/=2.
# Etot_WC=Ecorr+EHF_WC
Etot=data['Etot']
EH=data['EHartree']
EF=data['EFock']
if rank==0:
    # if lattice==1:
    #     savez(f'./Ecorr_rect_Ls{Ls:.1f}_nk{num_k1}_nD{nD}_D{Dfield:.3f}_eps{epsilon_d:.2f}_fl{filling:.3f}_meff{meff}.npz',Ecorr=Ecorr)
    # elif lattice==2:
    #     savez(f'./Ecorr_tri_Ls{Ls:.1f}_nk{num_k1}_nD{nD}_D{Dfield:.3f}_eps{epsilon_d:.2f}_fl{filling:.3f}_meff{meff}.npz',Ecorr=Ecorr)
    print('iq\tEcorr')         
    print('+==================================================================================+')
    print(f'Correlation energy calculation time elapse is {time2-time1:13.6f}s')

    print(f'E_kinetic = {Etot-EH-EF:11.8f} eV')
    print(f'E_Hartree = {EH:11.8f} eV')
    print(f'E_Fock    = {EF:11.8f} eV')
    print(f'E_tot_HF  = {Etot:11.8f} eV')
    print(f'E_corr    = {Ecorr:11.8f} eV')
    print(f'E_tot     = {Etot+Ecorr:11.8f} eV')
    savez(f'./Ecorr_Ls{Ls:.1f}_nk{num_k1}_nD{nD}_D{Dfield:.3f}_eps{epsilon_d:.2f}_gap{init_type}_fl{filling_e:.3f}_Ecut{Ecorr_cut}.npz',Etot=Etot+Ecorr,Ecorr=Ecorr,Etot_HF=Etot,EH=EH,EF=EF,Ekin=Etot-EH-EF)
