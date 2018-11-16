# -*- coding: utf-8 -*-
#Added an argument to allow switching between Vega Magnitudes. Also fixed not converting values from kcorrect code
#Changed absolute magnitude calculation to match Ilbert et al. including zero point offset. LH 4/21/17
#Changed the LCBG loadtxt statement to match the new LCBG selection format. Important LCBG
#selection criteria changed to reflect photometry that is important to Luminosity Function paper
#Commented out for loop to calculate absolute magnitudes. Just use B magnitudes from kcorrect code
#(corrB).
#Remove K-band data, throws off fit to SED 
#
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
import os

parser = argparse.ArgumentParser()
parser.add_argument("filein",help="File containing magnitudes and redshifts")
parser.add_argument("-m","--mbin",type=int,help="Number of Absolute Magnitude bins. Default=19",default=18)
parser.add_argument("-p","--zmin",type=float,help="Minimum redshift to consider in luminosity function, default=0.5", default=0.35)
parser.add_argument("-q","--zmax",type=float,help="Maximum redshift to consider in luminosity function, default=1.3", default=0.55)
parser.add_argument("-r","--Mmin",type=float,help="Minimum redshift to consider in luminosity function, default=--26.7", default=-24)
parser.add_argument("-s","--Mmax",type=float,help="Minimum redshift to consider in luminosity function, default=-18.5", default=-15)
parser.add_argument("-ama","--appmax",type=float,help='Maximum apparent magnitude to consider part of the survey COSMOS i<22.5',default=22.5)
parser.add_argument("-ami","--appmin",type=float,help='Minimum apparent magnitude to consider part of the survey COSMOS i>15',default=15)
parser.add_argument("-om","--OmegaMatter",type=float,help="Omega Matter, if you want to define your own cosmology", default=0.3)
parser.add_argument("-ho","--HubbleConstant",type=float,help="Hubble Constant if you want to define your own cosmology",default=70)
parser.add_argument("-fo","--fileout",help="Filename of PDF you want to generate",default='_LF_OUTPUT_VAL.txt')
parser.add_argument("-fl","--filts",help="Filename of all filter lists",default='FILTERLIST.txt')
parser.add_argument("-LCBG","--LCBGLIST",action="store_true",help="Make Luminosity Function with LCBGs only?")
parser.add_argument("-nv","--novega",action="store_true",help="Do not apply correction to switch from AB to Vega magnitudes")
args=parser.parse_args()

cosmo=FlatLambdaCDM(H0=args.HubbleConstant,Om0=args.OmegaMatter)
print('Reading File')

ID,RA,DECL,u,uerr,b,berr,v,verr,r,rerr,im,ierr,z,zerr,kUV,kUVerr,NUV,NUVerr,rh,zbests,zuses,zphots,SGCLASS=np.loadtxt(args.filein,unpack=True)

#if args.LCBGLIST:
#	LCBGFILE='/home/lrhunt/CATALOGS/LCBG_LISTS/COSMOS_LCBGS_VEGA.txt'
#	LCBGID,LCBGRA,LCBGDEC,LCBGZ,LCBGFLAG,LCBGb,LCBGv,LCBGRAD,LCBGCORRU,LCBGCORRB,LCBGCORRV=np.loadtxt(LCBGFILE,skiprows=1,unpack=True)


print(args.appmax,args.appmin)
Mdif=float(args.Mmax-args.Mmin)
Mbinsize=Mdif/args.mbin				#Creates bin size based on input for number of bins
i=0						#Variable used in for loop
magrange=np.linspace(args.appmin,args.appmax,16)
zrange=np.linspace(0,1.2,7)
tf=open('weights.txt','w')
Weighttot=np.zeros_like(b)
print('Calculating spectroscopic weights')
for k in range(0,len(magrange)-1):
	x=len(np.where((im>=magrange[k]) & (im<magrange[k+1]) & ((zuses==1)|(zuses==2)) & (SGCLASS==0))[0])
	y=len(np.where((im>=magrange[k]) & (im<magrange[k+1]) & ((zuses==3)|(zuses==4)) & (SGCLASS==0))[0])
        if (x>0):
            Weighttot[np.where((im>=magrange[k]) & (im<magrange[k+1]) & ((zuses==1)|(zuses==2)) & (SGCLASS==0))[0]]=float(x+y)/float(x)   
            print('Good Spec={}, Bad Spec={}, Weight={}, magrange={}-{}'.format(x,y,float(x+y)/float(x),magrange[k],magrange[k+1]))
            tf.write('Good Spec={}, Bad Spec={}, Weight={}, magrange={}-{}\n'.format(x,y,float(x+y)/float(x),magrange[k],magrange[k+1]))

tf.close()


print('Only selecting source in redshift range')
ID=ID[np.where((zuses<3)&(im<args.appmax)&(im>args.appmin)&(SGCLASS==0)&(zbests>=args.zmin)&(zbests<=args.zmax))[0]]
RA=RA[np.where((zuses<3)&(im<args.appmax)&(im>args.appmin)&(SGCLASS==0)&(zbests>=args.zmin)&(zbests<=args.zmax))[0]]
DECL=DECL[np.where((zuses<3)&(im<args.appmax)&(im>args.appmin)&(SGCLASS==0)&(zbests>=args.zmin)&(zbests<=args.zmax))[0]]
umags=u[np.where((zuses<3)&(im<args.appmax)&(im>args.appmin)&(SGCLASS==0)&(zbests>=args.zmin)&(zbests<=args.zmax))[0]]
uerrs=uerr[np.where((zuses<3)&(im<args.appmax)&(im>args.appmin)&(SGCLASS==0)&(zbests>=args.zmin)&(zbests<=args.zmax))[0]]
bmags=b[np.where((zuses<3)&(im<args.appmax)&(im>args.appmin)&(SGCLASS==0)&(zbests>=args.zmin)&(zbests<=args.zmax))[0]]
berrs=berr[np.where((zuses<3)&(im<args.appmax)&(im>args.appmin)&(SGCLASS==0)&(zbests>=args.zmin)&(zbests<=args.zmax))[0]]
vmags=v[np.where((zuses<3)&(im<args.appmax)&(im>args.appmin)&(SGCLASS==0)&(zbests>=args.zmin)&(zbests<=args.zmax))[0]]
verrs=verr[np.where((zuses<3)&(im<args.appmax)&(im>args.appmin)&(SGCLASS==0)&(zbests>=args.zmin)&(zbests<=args.zmax))[0]]
rmags=r[np.where((zuses<3)&(im<args.appmax)&(im>args.appmin)&(SGCLASS==0)&(zbests>=args.zmin)&(zbests<=args.zmax))[0]]
rerrs=rerr[np.where((zuses<3)&(im<args.appmax)&(im>args.appmin)&(SGCLASS==0)&(zbests>=args.zmin)&(zbests<=args.zmax))[0]]
imags=im[np.where((zuses<3)&(im<args.appmax)&(im>args.appmin)&(SGCLASS==0)&(zbests>=args.zmin)&(zbests<=args.zmax))[0]]
ierrs=ierr[np.where((zuses<3)&(im<args.appmax)&(im>args.appmin)&(SGCLASS==0)&(zbests>=args.zmin)&(zbests<=args.zmax))[0]]
zmags=z[np.where((zuses<3)&(im<args.appmax)&(im>args.appmin)&(SGCLASS==0)&(zbests>=args.zmin)&(zbests<=args.zmax))[0]]
zerrs=zerr[np.where((zuses<3)&(im<args.appmax)&(im>args.appmin)&(SGCLASS==0)&(zbests>=args.zmin)&(zbests<=args.zmax))[0]]
kmags=kUV[np.where((zuses<3)&(im<args.appmax)&(im>args.appmin)&(SGCLASS==0)&(zbests>=args.zmin)&(zbests<=args.zmax))[0]]
kerrs=kUVerr[np.where((zuses<3)&(im<args.appmax)&(im>args.appmin)&(SGCLASS==0)&(zbests>=args.zmin)&(zbests<=args.zmax))[0]]
zbest=zbests[np.where((zuses<3)&(im<args.appmax)&(im>args.appmin)&(SGCLASS==0)&(zbests>=args.zmin)&(zbests<=args.zmax))[0]]
zphot=zphots[np.where((zuses<3)&(im<args.appmax)&(im>args.appmin)&(SGCLASS==0)&(zbests>=args.zmin)&(zbests<=args.zmax))[0]]
zuse=zuses[np.where((zuses<3)&(im<args.appmax)&(im>args.appmin)&(SGCLASS==0)&(zbests>=args.zmin)&(zbests<=args.zmax))[0]]
WeightArray=Weighttot[np.where((zuses<3)&(im<args.appmax)&(im>args.appmin)&(SGCLASS==0)&(zbests>=args.zmin)&(zbests<=args.zmax))[0]]
rh=rh[np.where((zuses<3)&(im<args.appmax)&(im>args.appmin)&(SGCLASS==0)&(zbests>=args.zmin)&(zbests<=args.zmax))[0]]


print("number of sources = {}".format(len(bmags)))

print('Converting to maggies')
umaggies=ut.mag2maggies(umags)
bmaggies=ut.mag2maggies(bmags)
vmaggies=ut.mag2maggies(vmags)
rmaggies=ut.mag2maggies(rmags)
imaggies=ut.mag2maggies(imags)
zmaggies=ut.mag2maggies(zmags)

uinvervar=ut.invariance(umaggies,uerrs)
binvervar=ut.invariance(bmaggies,berrs)
vinvervar=ut.invariance(vmaggies,verrs)
rinvervar=ut.invariance(rmaggies,rerrs)
iinvervar=ut.invariance(imaggies,ierrs)
zinvervar=ut.invariance(zmaggies,zerrs)

allmaggies=np.stack((umaggies,bmaggies,vmaggies,rmaggies,imaggies,zmaggies),axis=-1)
allinvervar=np.stack((uinvervar,binvervar,vinvervar,rinvervar,iinvervar,zinvervar),axis=-1)

carr=np.ndarray((len(bmaggies),6))
rmarr=np.ndarray((len(bmaggies),7))
rmarr0=np.ndarray((len(bmaggies),7))
rmarr0B=np.ndarray((len(bmaggies),7))
rmarr0V=np.ndarray((len(bmaggies),7))
rmarr0U=np.ndarray((len(bmaggies),7))

print('Computing k-corrections')
kcorrect.load_templates()
kcorrect.load_filters('/home/lrhunt/programs/kcorrect/data/templates/Lum_Func_Filters_US.dat')


for i in range(0,len(carr)):
	carr[i]=kcorrect.fit_nonneg(zbest[i],allmaggies[i],allinvervar[i])
for i in range(0,len(carr)):
	rmarr[i]=kcorrect.reconstruct_maggies(carr[i])
	rmarr0[i]=kcorrect.reconstruct_maggies(carr[i],redshift=0)

kcorrect.load_templates()
kcorrect.load_filters('/home/lrhunt/programs/kcorrect/data/templates/BESSEL_B2.dat')
for i in range(0,len(carr)):
	rmarr0B[i]=kcorrect.reconstruct_maggies(carr[i],redshift=0)

kcorrect.load_templates()
kcorrect.load_filters('/home/lrhunt/programs/kcorrect/data/templates/BESSEL_V2.dat')

for i in range(0,len(carr)):
	rmarr0V[i]=kcorrect.reconstruct_maggies(carr[i],redshift=0)

kcorrect.load_templates()
kcorrect.load_filters('/home/lrhunt/programs/kcorrect/data/templates/BESSEL_U2.dat')

for i in range(0,len(carr)):
	rmarr0U[i]=kcorrect.reconstruct_maggies(carr[i],redshift=0)


kcorrM=-2.5*np.log10(rmarr/rmarr0B)
kcorr=-2.5*np.log10(rmarr/rmarr0)
corrB=-2.5*np.log10(rmarr0B)+0.09
corrV=-2.5*np.log10(rmarr0V)-0.02
corrU=-2.5*np.log10(rmarr0U)-0.79

recmags=-2.5*np.log10(rmarr)
np.savetxt('kcorr.txt',kcorr)
#if args.LCBGLIST:
#	indices = np.where(np.in1d(ID,LCBGID))[0]
#	WeightArray=WeightArray[indices]
#for i in range(0,len(indices)):
#	print(indices[i])



M=np.zeros_like(zbest)
print('Calculating Absolute Magnitudes')
#M=corrB[:,2]-cosmo.distmod(zbest).value
MinimumM=0
MaximumM=-140
for i in range(0,len(zbest)):
	if zbest[i]<=0.1:
		M[i]=bmags[i]-0.05122-cosmo.distmod(zbest[i]).value-kcorrM[i][2]
	if zbest[i]<=0.35 and zbest[i]>0.1:
		M[i]=vmags[i]+0.069802-cosmo.distmod(zbest[i]).value-kcorrM[i][3]
	if zbest[i]<=0.55 and zbest[i]>0.35:
		M[i]=rmags[i]-0.01267-cosmo.distmod(zbest[i]).value-kcorrM[i][4]
	if zbest[i]<=0.75 and zbest[i]>0.55:
		M[i]=imags[i]-0.004512-cosmo.distmod(zbest[i]).value-kcorrM[i][5]
	if zbest[i]>0.75:
		M[i]=zmags[i]-0.00177-cosmo.distmod(zbest[i]).value-kcorrM[i][6]
	if M[i]<MinimumM and M[i]>-40:
		MinimumM=M[i]
		print('Minumum M={}'.format(MinimumM))
	if M[i]>MaximumM and M[i]<0:
		MaximumM=M[i]
		print('Maximum M={}'.format(MaximumM))

M=M+0.09

SBe=M+2.5*np.log10((2*np.pi*np.power(cosmo.angular_diameter_distance(zbest).value*np.tan(rh*0.03*4.84814e-6)*(814/(445*(1+zbest)))**0.108*1e3,2)))+2.5*np.log10((360*60*60/(2*np.pi*0.01))**2)

bv=corrB[:,3]-corrV[:,4]

LCBGS=np.where((M<=-18.5)&(SBe<=21)&(bv<0.6))[0]
LCBGS=np.ndarray.astype(LCBGS,dtype=int)

if args.novega:
	M=M-0.09

#if args.LCBGLIST:
#	M=M[indices]

print('Calculatiing zmin and zmax')

zupper=np.zeros_like(zbest)
zlower=np.zeros_like(zbest)
zlookup=np.linspace(0,1,1000)

kcorrect.load_templates()
kcorrect.load_filters('/home/lrhunt/programs/kcorrect/data/templates/Lum_Func_Filters_US.dat')

for i in range(0,len(carr)):
	upperlim=args.appmax-imags[i]+cosmo.distmod(zbest[i]).value-2.5*np.log10(rmarr[i][5])
	lowerlim=args.appmin-imags[i]+cosmo.distmod(zbest[i]).value-2.5*np.log10(rmarr[i][5])
	rmarrlookup=np.ndarray((1000,7))
	for j in range(0,1000):
		rmarrlookup[j]=kcorrect.reconstruct_maggies(carr[i],redshift=zlookup[j])
	ilookup=-2.5*np.log10(rmarrlookup[:,5])
	w=cosmo.distmod(zlookup).value+ilookup
	if not (np.isnan(upperlim) & np.isnan(lowerlim)):
		zupper[i]=round(zlookup[np.abs(w-upperlim).argmin()],4)
		zlower[i]=round(zlookup[np.abs(w-lowerlim).argmin()],4)
		


zminprobs=[]
print('Calculating Comoving Volume')
CMV=np.zeros_like(zupper)
for i in range(0,len(CMV)):
	if zupper[i]<args.zmax:
		if zlower[i]>args.zmin:
			CMV[i]=cosmo.comoving_volume(zupper[i]).value/(4*np.pi/0.0003116)-cosmo.comoving_volume(zlower[i]).value/(4*np.pi/(0.0003116))
		else:
			CMV[i]=cosmo.comoving_volume(zupper[i]).value/(4*np.pi/0.0003116)-cosmo.comoving_volume(args.zmin).value/(4*np.pi/(0.0003116))
	else:
		if zlower[i]>args.zmin:
			CMV[i]=cosmo.comoving_volume(args.zmax).value/(4*np.pi/0.0003116)-cosmo.comoving_volume(zlower[i]).value/(4*np.pi/(0.0003116))
		else:
			CMV[i]=cosmo.comoving_volume(args.zmax).value/(4*np.pi/0.0003116)-cosmo.comoving_volume(args.zmin).value/(4*np.pi/(0.0003116))


if args.LCBGLIST:
	M=M[LCBGS]
	WeightArray=WeightArray[LCBGS]
	CMV=CMV[LCBGS]


print('Binning Absolute Magnitudes')
MBinLow=np.arange(args.mbin)*1.
MBinUp=np.arange(args.mbin)*1.
WhereMbin=[]
MVariable=args.Mmin

for i in range(0,args.mbin):	
	WhereMbin.append(np.where((M<=(MVariable+Mbinsize)) & (M>MVariable))[0])
	MBinLow[i]=MVariable
	MVariable=MVariable+Mbinsize
	MBinUp[i]=MVariable

LumFunc=np.array([])
LumFuncErr=np.array([])
NGal=np.array([])
AveCMV=np.array([])
AveWeight=np.array([])
MBINAVE=np.array([])
MBinMid=(MBinUp+MBinLow)/2.


print('Calculating values for Luminosity Function')
for i in range(0,len(WhereMbin)):
	val=0.0
	err=0.0
	NGalparam=0
	TOTVOL=0.0
	TOTWEIGHT=0.0
	for j in range(0,len(WhereMbin[i])):
		if CMV[WhereMbin[i][j]]>0:
			val=val+(WeightArray[WhereMbin[i][j]])/CMV[WhereMbin[i][j]]
			err=err+np.power(WeightArray[WhereMbin[i][j]],2)/(np.power((CMV[WhereMbin[i][j]])*Mbinsize,2))
			NGalparam=NGalparam+1
			TOTVOL=TOTVOL+CMV[WhereMbin[i][j]]
			TOTWEIGHT=TOTWEIGHT+WeightArray[WhereMbin[i][j]]
	MBINAVE=np.append(MBINAVE,np.mean(M[WhereMbin[i]]))
	LumFunc=np.append(LumFunc,val/Mbinsize)
	NGal=np.append(NGal,NGalparam)
	LumFuncErr=np.append(LumFuncErr,np.sqrt(err))
	if NGalparam!=0:
		AveCMV=np.append(AveCMV,TOTVOL/NGalparam)
		AveWeight=np.append(AveWeight,TOTWEIGHT/NGalparam)
	else:
		AveCMV=np.append(AveCMV,0)
		AveWeight=np.append(AveWeight,0)


maxvol=cosmo.comoving_volume(args.zmax).value/(4*np.pi/0.0003116)-cosmo.comoving_volume(args.zmin).value/(4*np.pi/(0.0003116))
LogErr=LumFuncErr/(LumFunc*np.log(10))
print(MBINAVE)
outarray=np.stack((MBinMid,MBINAVE,LumFunc,LumFuncErr,LogErr,NGal,AveCMV,AveWeight),axis=-1)
if not args.LCBGLIST:
	zrangearr=np.stack((M,imags,zbest,zupper,zlower),axis=-1)
	np.savetxt('redshiftrange_lf.txt',zrangearr)
np.savetxt(args.fileout,outarray,header='{} {} {} {} {}'.format(args.zmax,args.zmin,args.mbin,Mbinsize,maxvol))
if args.LCBGLIST:
	LCBGCMV=np.stack((ID[LCBGS],zbest[LCBGS],M,zupper[LCBGS],zlower[LCBGS],WeightArray),axis=-1)
	np.savetxt('LCBG_CMV.txt',LCBGCMV,header='ID	zbest	M	zup	zlow')
	np.savetxt('LCBG_M_LIST.txt',M)
print(LumFunc)
