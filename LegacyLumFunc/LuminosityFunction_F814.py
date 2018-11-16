# -*- coding: utf-8 -*-
import sys
import numpy as np
import astropy as ap
import matplotlib.pyplot as plt
import math as m
import argparse
from astropy.cosmology import FlatLambdaCDM
from matplotlib.backends.backend_pdf import PdfPages
import kcorrect
import kcorrect.utils as ut

parser = argparse.ArgumentParser()
parser.add_argument("filein",help="File containing magnitudes and redshifts")
parser.add_argument("-m","--mbin",type=int,help="Number of Absolute Magnitude bins. Default=19",default=18)
parser.add_argument("-p","--zmin",type=float,help="Minimum redshift to consider in luminosity function, default=0.5", default=0.35)
parser.add_argument("-q","--zmax",type=float,help="Maximum redshift to consider in luminosity function, default=1.3", default=0.55)
parser.add_argument("-r","--Mmin",type=float,help="Minimum redshift to consider in luminosity function, default=--26.7", default=-24)
parser.add_argument("-s","--Mmax",type=float,help="Minimum redshift to consider in luminosity function, default=-18.5", default=-15)
parser.add_argument("-ama","--appmax",type=float,help='Maximum apparent magnitude to consider part of the survey COSMOS i<22.5',default=22.5)
parser.add_argument("-ami","--appmin",type=float,help='Minimum apparent magnitude to consider part of the survey COSMOS i>15',default=15)
parser.add_argument("-om","--OmegaMatter",type=float,help="Omega Matter, if you want to define your own cosmology", default=0.286)
parser.add_argument("-ho","--HubbleConstant",type=float,help="Hubble Constant if you want to define your own cosmology",default=69.3)
parser.add_argument("-ps","--phistar",type=float,help="Phi* value for input Schechter Function",default=0.00715)
parser.add_argument("-ms","--mstar",type=float,help="M* value for input Schechter Fucntion",default=-21.17)
parser.add_argument("-al","--schalpha",type=float,help="alpha value for input Schechter Function",default=-1.03)
parser.add_argument("-fo","--fileout",help="Filename of PDF you want to generate",default='_LF_OUTPUT_VAL.txt')

args=parser.parse_args()

cosmo=FlatLambdaCDM(H0=args.HubbleConstant,Om0=args.OmegaMatter)
print('Reading File')
ID,RA_06,DEC_06,bmagtot,berrtot,vmagtot,verrtot,rmagtot,rerrtot,imagtot,ierrtot,zmagtot,zerrtot,zbesttot,RHtot,zusetot=np.loadtxt(args.filein,skiprows=1,unpack=True)

alpha=1.596*m.pi/180				#angular size of COSMOS pointing (Degrees) 
Mdif=float(args.Mmax-args.Mmin)
Mbinsize=Mdif/args.mbin				#Creates bin size based on input for number of bins
i=0						#Variable used in for loop

TSR=float(len(zusetot))/float(len(np.where(zusetot<4)[0]))
print('Only selecting source in redshift range')
bmags=bmagtot[np.where(np.logical_and(zbesttot>args.zmin,zbesttot<args.zmax))[0]]
berrs=berrtot[np.where(np.logical_and(zbesttot>args.zmin,zbesttot<args.zmax))[0]]
vmags=vmagtot[np.where(np.logical_and(zbesttot>args.zmin,zbesttot<args.zmax))[0]]
verrs=verrtot[np.where(np.logical_and(zbesttot>args.zmin,zbesttot<args.zmax))[0]]
rmags=rmagtot[np.where(np.logical_and(zbesttot>args.zmin,zbesttot<args.zmax))[0]]
rerrs=rerrtot[np.where(np.logical_and(zbesttot>args.zmin,zbesttot<args.zmax))[0]]
imags=imagtot[np.where(np.logical_and(zbesttot>args.zmin,zbesttot<args.zmax))[0]]
ierrs=ierrtot[np.where(np.logical_and(zbesttot>args.zmin,zbesttot<args.zmax))[0]]
zmags=zmagtot[np.where(np.logical_and(zbesttot>args.zmin,zbesttot<args.zmax))[0]]
zerrs=zmagtot[np.where(np.logical_and(zbesttot>args.zmin,zbesttot<args.zmax))[0]]
zbest=zbesttot[np.where(np.logical_and(zbesttot>args.zmin,zbesttot<args.zmax))[0]]
zuse=zusetot[np.where(np.logical_and(zbesttot>args.zmin,zbesttot<args.zmax))[0]]
RH=RHtot[np.where(np.logical_and(zbesttot>args.zmin,zbesttot<args.zmax))[0]]

bmags=bmags+0.189
vmags=vmags+0.04
rmags=rmags-0.04
zmags=zmags+0.054
print('Converting to maggies')
bmaggies=ut.mag2maggies(bmags)
vmaggies=ut.mag2maggies(rmags)
rmaggies=ut.mag2maggies(vmags)
imaggies=ut.mag2maggies(imags)
zmaggies=ut.mag2maggies(zmags)

binvervar=ut.invariance(bmaggies,berrs)
vinvervar=ut.invariance(vmaggies,verrs)
rinvervar=ut.invariance(rmaggies,rerrs)
iinvervar=ut.invariance(imaggies,ierrs)
zinvervar=ut.invariance(zmaggies,zerrs)

print('Dividing up sources based on i-band filter')
aarr=np.stack((zbest,bmaggies,vmaggies,rmaggies,imaggies,zmaggies,binvervar,vinvervar,rinvervar,iinvervar,zinvervar),axis=-1)
carr=np.ndarray((len(bmaggies),6))
rmarr=np.ndarray((len(bmaggies),6))
rmarr0=np.ndarray((len(bmaggies),6))

print('Computing k-corrections')
kcorrect.load_templates()
kcorrect.load_filters('/home/lhunt/programs/kcorrect/data/templates/Lum_Func_Filters.dat')


for i in range(0,len(carr)):
	c=kcorrect.fit_coeffs(aarr[i])
	for j in range(0,len(carr[i])):
		carr[i][j]=c[j]

for i in range(0,len(carr)):
	rm=kcorrect.reconstruct_maggies(carr[i][1:],redshift=carr[i][0])
	for j in range(0,len(rmarr[i])):
		rmarr[i][j]=rm[j]

kcorrect.load_templates()
kcorrect.load_filters('/home/lhunt/programs/kcorrect/data/templates/SUPRIME_B_ONLY.dat')

for i in range(0,len(rmarr)):
	rm0=kcorrect.reconstruct_maggies(carr[i][1:],redshift=0)
	for j in range(0,len(rmarr[i])):
		rmarr0[i][j]=rm0[j]

kcorr=-2.5*np.log10(rmarr/rmarr0)

M=np.zeros_like(zbest)
print('Calculating Absolute Magnitudes')
MinimumM=0
for i in range(0,len(zbest)):
	if zbest[i]>0:
		if zbest[i]<=0.1:
			M[i]=bmags[i]-cosmo.distmod(zbest[i]).value-kcorr[i][1]
		if zbest[i]<=0.35 and zbest[i]>0.1:
			M[i]=vmags[i]-cosmo.distmod(zbest[i]).value-kcorr[i][2]
		if zbest[i]<=0.55 and zbest[i]>0.35:
			M[i]=rmags[i]-cosmo.distmod(zbest[i]).value-kcorr[i][3]
		if zbest[i]<=0.75 and zbest[i]>0.55:
			M[i]=imags[i]-cosmo.distmod(zbest[i]).value-kcorr[i][4]
		if zbest[i]>0.75:
			M[i]=zmags[i]-cosmo.distmod(zbest[i]).value-kcorr[i][5]
		if M[i]<MinimumM and M[i]>-40:
			MinimumM=M[i]
			print(MinimumM)

M_Cutoff=22.5-cosmo.distmod(args.zmax).value-kcorr[len(kcorr)-1][4]
print(M_Cutoff)

print('Calculatiing zmin and zmax')

zupper=np.zeros_like(zbest)
zlower=np.zeros_like(zbest)
kcorrect.load_templates()
kcorrect.load_filters('/home/lhunt/programs/kcorrect/data/templates/Lum_Func_Filters.dat')

for i in range(0,len(imags)):
	tempz=zbest[i]
	upperlim=args.appmax-imags[i]+cosmo.distmod(tempz).value+kcorr[i][4]
	matchval=0
	while matchval<upperlim:
		matchval=cosmo.distmod(tempz).value-2.5*np.log10((kcorrect.reconstruct_maggies(carr[i][1:],redshift=tempz)[4])/rmarr0[i][4])
		tempz=tempz+0.1
	tempz=tempz-0.1
	while matchval<upperlim:
		matchval=cosmo.distmod(tempz).value-2.5*np.log10((kcorrect.reconstruct_maggies(carr[i][1:],redshift=tempz)[4])/rmarr0[i][4])
		tempz=tempz+0.01
	tempz=tempz-0.01
	while matchval<upperlim:
		matchval=cosmo.distmod(tempz).value-2.5*np.log10((kcorrect.reconstruct_maggies(carr[i][1:],redshift=tempz)[4])/rmarr0[i][4])
		tempz=tempz+0.001
	tempz=tempz-0.001
	while matchval<upperlim:
		matchval=cosmo.distmod(tempz).value-2.5*np.log10((kcorrect.reconstruct_maggies(carr[i][1:],redshift=tempz)[4])/rmarr0[i][4])
		tempz=tempz+0.0001
	tempz=tempz-0.0001
	zupper[i]=tempz
	lowerlim=args.appmin-imags[i]+cosmo.distmod(tempz).value+kcorr[i][4]
	matchval=100
	tempz=zbest[i]
	while matchval>lowerlim:
		matchval=cosmo.distmod(tempz).value-2.5*np.log10((kcorrect.reconstruct_maggies(carr[i][1:],redshift=tempz)[4])/rmarr0[i][4])
		tempz=tempz-0.1
	if tempz<0:
		tempz=0
		matchval=0	
	tempz=tempz+0.1
	while matchval>lowerlim:
		matchval=cosmo.distmod(tempz).value-2.5*np.log10((kcorrect.reconstruct_maggies(carr[i][1:],redshift=tempz)[4])/rmarr0[i][4])
		tempz=tempz-0.01
	tempz=tempz+0.01
	while matchval>lowerlim:
		matchval=cosmo.distmod(tempz).value-2.5*np.log10((kcorrect.reconstruct_maggies(carr[i][1:],redshift=tempz)[4])/rmarr0[i][4])
		tempz=tempz-0.001
	tempz=tempz+0.001
	while matchval>lowerlim:
		matchval=cosmo.distmod(tempz).value-2.5*np.log10((kcorrect.reconstruct_maggies(carr[i][1:],redshift=tempz)[4])/rmarr0[i][4])
		tempz=tempz-0.0001
	tempz=tempz+0.0001
	if tempz==0.1111:
		tempz=0
	zlower[i]=tempz

zminprobs=[]
print('Calculating Comoving Volume')
CMV=np.zeros_like(zupper)
for i in range(0,len(CMV)):
	if zupper[i]<args.zmax:
		if zlower[i]>args.zmin:
			CMV[i]=cosmo.comoving_volume(zupper[i]).value/(4*np.pi/0.000312)-cosmo.comoving_volume(zlower[i]).value/(4*np.pi/(0.000312))
		else:
			CMV[i]=cosmo.comoving_volume(zupper[i]).value/(4*np.pi/0.000312)-cosmo.comoving_volume(args.zmin).value/(4*np.pi/(0.000312))
	else:
		if zlower[i]>args.zmin:
			CMV[i]=cosmo.comoving_volume(args.zmax).value/(4*np.pi/0.000312)-cosmo.comoving_volume(zlower[i]).value/(4*np.pi/(0.000312))
			zminprobs.append(np.array([zlower[i],zbest[i],zuse[i],imags[i],M[i],kcorr[i][4]]))
		else:
			CMV[i]=cosmo.comoving_volume(args.zmax).value/(4*np.pi/0.000312)-cosmo.comoving_volume(args.zmin).value/(4*np.pi/(0.000312))

testing=np.asarray(zminprobs)

print('Binning Absolute Magnitudes')
MBinLow=np.arange(args.mbin)*1.
MBinUp=np.arange(args.mbin)*1.
WhereMbin=[]
MVariable=args.Mmin
for i in range(0,args.mbin):		
	WhereMbin.append(np.where((M<=MVariable+Mbinsize) & (M>MVariable))[0])
	MBinLow[i]=MVariable
	MVariable=MVariable+Mbinsize
	MBinUp[i]=MVariable

Wherembin=[]
mvariable=args.appmin
mdif=args.appmax-args.appmin
mbinsize=0.5
magbins=int(mdif/mbinsize)
for i in range(0,magbins):
	Wherembin.append(np.where((imags<=mvariable+mbinsize) & (imags>mvariable))[0])
	mvariable=mvariable+mbinsize

WeightArray=np.zeros_like(bmags)
print('Calculating spectroscopic weights')
for i in range(0,magbins):
	goodspec=float(len(np.where(zuse[Wherembin[i]]<3)[0]))
	badspec=float(len(np.where(zuse[Wherembin[i]]==3)[0]))
	if goodspec!=0:
		specweight=(goodspec+badspec)/goodspec
	else:
		specweight=0
	WeightArray[Wherembin[i]]=specweight


LumFunc=np.array([])
LumFuncErr=np.array([])
NGal=np.array([])
AveCMV=np.array([])
MBinMid=(MBinUp+MBinLow)/2.
print('Calculating values for Luminosity Function')
for i in range(0,len(WhereMbin)):
	val=0.0
	err=0.0
	NGalparam=0
	TOTVOL=0.0
	for j in range(0,len(WhereMbin[i])):
		if zuse[WhereMbin[i][j]]==1 or zuse[WhereMbin[i][j]]==2:
			if CMV[WhereMbin[i][j]]>0:
				val=val+(TSR*WeightArray[WhereMbin[i][j]])/CMV[WhereMbin[i][j]]
				err=err+np.power(WeightArray[WhereMbin[i][j]],2)/(np.power((CMV[WhereMbin[i][j]])*Mbinsize,2))
				NGalparam=NGalparam+1
				TOTVOL=TOTVOL+CMV[WhereMbin[i][j]]
	LumFunc=np.append(LumFunc,val/Mbinsize)
	NGal=np.append(NGal,NGalparam)
	LumFuncErr=np.append(LumFuncErr,np.sqrt(err))
	if NGalparam!=0:
		AveCMV=np.append(AveCMV,TOTVOL/NGalparam)
	else:
		AveCMV=np.append(AveCMV,0)
LogErr=LumFuncErr/(LumFunc*np.log(10))
outarray=np.stack((MBinMid,LumFunc,LumFuncErr,LogErr,NGal,AveCMV),axis=-1)
np.savetxt(args.fileout,outarray,header='%.3f %.3f %d %.3f' % (args.zmax,args.zmin,args.mbin,M_Cutoff))
np.savetxt('zmincandidates.txt',testing)
print(LumFunc)
		






