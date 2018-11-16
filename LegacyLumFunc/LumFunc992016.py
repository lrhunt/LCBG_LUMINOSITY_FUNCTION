# -*- coding: utf-8 -*-
import sys
import numpy as np
import astropy as ap
import matplotlib.pyplot as plt
import math as m
import argparse
import kcorrect
from astropy.cosmology import WMAP9 as cosmo
from matplotlib.backends.backend_pdf import PdfPages


			###############    SEGMENT 1    ###############


parser = argparse.ArgumentParser()
parser.add_argument("filein",help="File containing magnitudes and redshifts")
parser.add_argument("-d","--diagnostic",action="store_true",help="Include diagnostic plots. Vmax vs Magnitude, Number of Galaxies per M bin and Density (Sum of Luminosity Function) vs Redshift")
parser.add_argument("-z","--zbin",type=int,help="Number of redshift bins. Default=4",default=4)
parser.add_argument("-m","--mbin",type=int,help="Number of Absolute Magnitude bins. Default=17",default=17)
parser.add_argument("-p","--zmin",type=float,help="Minimum redshift to consider in luminosity function, default=0.5", default=0.5)
parser.add_argument("-q","--zmax",type=float,help="Maximum redshift to consider in luminosity function, default=1.3", default=1.3)
parser.add_argument("-r","--Mmin",type=float,help="Minimum redshift to consider in luminosity function, default=--26.7", default=-26.7)
parser.add_argument("-s","--Mmax",type=float,help="Minimum redshift to consider in luminosity function, default=-18.5", default=-18.5)
parser.add_argument("-op","--overplot",action="store_true",help="Overplot the Schechter Function from Zucca,2009 (Luminosity Function from zCosmos 10k)")
parser.add_argument("-mx","--appmax",type=float,help="Maximum apparent magnitude of survey, default=22.5",default=29.37)
parser.add_argument("-mn","--appmin",type=float,help="Maximum apparent magnitude of survey, default=22.5",default=17)
args=parser.parse_args()

b06,b06err,v06,v06err,r06,r06err,i06,i06err,z06,z06err,zarr,classif,zuse=np.loadtxt(args.filein,skiprows=1,unpack=True)	#creates apparent and absolute magnitude arrays and redshift array

			###############    SEGMENT 2    ###############
zuse1=len(np.where(zuse==1)[0])*1.0
zuse2=len(np.where(zuse==2)[0])*1.0
zuse3=len(np.where(zuse==3)[0])*1.0
zuse4=len(np.where(zuse==4)[0])*1.0
TargetedForSpec=(zuse1+zuse2+zuse3)/(zuse1+zuse2+zuse3+zuse4)
WeightForSpec=1/TargetedForSpec
alpha=1.596*np.pi/180				#angular size of COSMOS pointing (Degrees) 
zdif=args.zmax-args.zmin			#Calculates difference between max and min redshift
zbinsize=zdif/args.zbin				#Creates bin size based on input for number of bins
Mdif=args.Mmax-args.Mmin			#Calculates difference between max and min Magnitude
Mbinsize=Mdif/args.mbin				#Creates bin size based on input for number of bins
zVariable=args.zmin				#Creates a variable to use in loops through redshift
MPopsByz=[]					#List for galaxies binned by redshift
zBinWhere=[]					#Index. Allows us to index zarr and b06 later.
i=0						#Variable used in for loop
zbinup=np.arange(args.zbin)*1.			#Upper bound for zbin
zbinlow=np.arange(args.zbin)*1.			#Lower bound for zbin

#applying offsets from Capak, (2007)

b06=b06+0.189
v06=v06+0.04
r06=r06-0.04
z06=z06+0.054

#calculating maggies for use in Blanton kcorrect code

bmaggies=np.power(10,b06/(-2.5))
vmaggies=np.power(10,v06/(-2.5))
rmaggies=np.power(10,r06/(-2.5))
imaggies=np.power(10,i06/(-2.5))
zmaggies=np.power(10,z06/(-2.5))

binvervar=np.power(0.4*np.log(10)*bmaggies*b06err,-2)
vinvervar=np.power(0.4*np.log(10)*vmaggies*v06err,-2)
rinvervar=np.power(0.4*np.log(10)*rmaggies*r06err,-2)
iinvervar=np.power(0.4*np.log(10)*imaggies*i06err,-2)
zinvervar=np.power(0.4*np.log(10)*zmaggies*z06err,-2)

#calculating kcorrections for each source. z=0 maggies calculated independently and assumed to not change, therefore just read in from file

kcorrect.load_templates()
kcorrect.load_filters('/home/lhunt/programs/kcorrect/data/templates/Lum_Func_Filters.dat')

karray=np.stack((zarr,bmaggies,vmaggies,rmaggies,imaggies,zmaggies,binvervar,vinvervar,rinvervar,iinvervar,zinvervar),axis=-1)
zarr0,b06rm0,v06rm0,r06rm0,i06rm0,z06rm0=np.loadtxt('outmaggies_0.txt',unpack=True)

kb=np.zeros_like(b06,dtype=np.float)
kv=np.zeros_like(b06,dtype=np.float)
kr=np.zeros_like(b06,dtype=np.float)
ki=np.zeros_like(b06,dtype=np.float)
kz=np.zeros_like(b06,dtype=np.float)

for i in range(0,len(karray)):
	coeff=kcorrect.fit_nonneg(karray[i][0],karray[i][1:6],karray[i][6:12])
	m=kcorrect.reconstruct_maggies(coeff)
	kb[i]=-2.5*np.log10(m[1]/b06rm0[i])+5*np.log10(0.6932)
	kv[i]=-2.5*np.log10(m[2]/v06rm0[i])+5*np.log10(0.6932)
	kr[i]=-2.5*np.log10(m[3]/r06rm0[i])+5*np.log10(0.6932)
	ki[i]=-2.5*np.log10(m[4]/i06rm0[i])+5*np.log10(0.6932)
	kz[i]=-2.5*np.log10(m[5]/z06rm0[i])+5*np.log10(0.6932)
		
#Calculating absolute magnitudes with k-correction

M=np.empty_like(b06,dtype='float')
i=0

for i in range(0,len(zarr)):
	if zarr[i]>0:
		if zarr[i]<=0.16:
			M[i]=b06[i]-5*(np.log10(100000*cosmo.luminosity_distance(zarr[i]).value))-kb[i]
		if zarr[i]<=0.31 and zarr[i]>0.16:
			M[i]=v06[i]-5*(np.log10(100000*cosmo.luminosity_distance(zarr[i]).value))-kv[i]
		if zarr[i]<=0.55 and zarr[i]>0.31:
			M[i]=r06[i]-5*(np.log10(100000*cosmo.luminosity_distance(zarr[i]).value))-kr[i]
		if zarr[i]<=0.9 and zarr[i]>0.55:
			M[i]=i06[i]-5*(np.log10(100000*cosmo.luminosity_distance(zarr[i]).value))-ki[i]
		if zarr[i]>0.9:
			M[i]=z06[i]-5*(np.log10(100000*cosmo.luminosity_distance(zarr[i]).value))-kz[i]

#Calculating color, used for weighting of sources

bv=b06-v06
for i in range(0,len(bv)):
	if (bv[i]==0 and b06[i]==-999.98999):
		bv[i]=-999.98999

#Determining wheighting. Finding number of sources in each color/mag bin that have spectroscopic redshifts

sortcolorwhere=[]
sortcolori=[]
sortweighting=[]
colorvar=min(bv[np.where(bv>-3)[0]])
colorbinsize=(max(bv[np.where(bv<4)[0]])-colorvar)/args.mbin
ivariable=min(i06[np.where(i06>0)[0]])
ibinsize=(max(i06)-min(i06[np.where(i06>0)[0]]))/args.mbin
for i in range(0,args.mbin):
	sortcolorwhere.append(np.where((bv>=colorvar) & (bv<colorvar+colorbinsize))[0])
	sortcolori.append(i06[sortcolorwhere[i]])
	colorvar=colorvar+colorbinsize
	
for i in range(0,args.mbin):
	sortweighting.append([])
	j=0
	ivariable=min(i06[np.where(i06>0)[0]])
	for j in range(0,args.mbin):
		sortweighting[i].append(sortcolorwhere[i][np.where((sortcolori[i]>=ivariable) & (sortcolori[i]<ivariable+ibinsize))[0]])
		ivariable=ivariable+ibinsize

colorvar=min(bv[np.where(bv>-3)[0]])


	#This loop goes through all the sources and separates them based on redshift.
	#MPopsByz is filled with the Magnitude of the objects in a given redshift bin. zBinWhere gives the index for b06,M,zarr (b06[MPopsByz[0]],M[MPopsByz[0]],zarr[MPopsByz[0]] will give m,M,z for the same source)
for i in range(0,args.zbin):
	zBinWhere.append(np.where((zarr>=zVariable) & (zarr<zVariable+zbinsize))[0])
	MPopsByz.append(M[zBinWhere[i]])
	zbinlow[i]=zVariable				
	zVariable=zVariable+zbinsize
	zbinup[i]=zVariable


			###############    SEGMENT 3    ###############


i=0
zVariable=args.zmin		
LumFuncWhere=[]			#Index array. To point back to b06,zarr,M for respective values. 
MBinLow=np.arange(args.mbin)*1.	#Lower bound for Mbin. If we have 12 bins and the upper and lower bounds for Magnitude are -18.5 and -24.5 then each bin is 0.5 magnitude. MBinLow[0]=-18.5
MBinUp=np.arange(args.mbin)*1.	#Upper bound for Mbin. If we have 12 bins and the upper and lower bounds for Magnitude are -18.5 and -24.5 then each bin is 0.5 magnitude. MBinUp[0]=-19.0
	# Calculate index array. Will sort sources by redshift; then in each redshift bin, will sort by Magnitude. The numbers stored are indexes for b06,M,zarr Structure=LumFuncWhere[RedshiftBin][MagnitudeBin][Index pointing to b06,zarr,M] 
for i in range(0,args.zbin):
	LumFuncWhere.append([])
	j=0
	MVariable=args.Mmin
	for j in range(0,args.mbin):
		LumFuncWhere[i].append(zBinWhere[i][np.where((MPopsByz[i]<=MVariable+Mbinsize) & (MPopsByz[i]>MVariable))[0]])
		MBinUp[j]=MVariable
		MVariable=MVariable+Mbinsize
		MBinLow[j]=MVariable
	zVariable=zVariable+zbinsize

#Calculate upper and lower redshift bins for each source. This is done by comparing the distance modulus of the apparent magnitude of the source and the upper (lower) Magnitude limit of
#the bin that source is in. If that redshift is above (below) the maximum (minimum) redshift for the sources bin, zupper (zlower) is the maximum redshift for the sources bin. 

zupper=[]
zlower=[]

for i in range(0,len(LumFuncWhere)):
	zupper.append([])
	zlower.append([])
	for j in range(0,len(LumFuncWhere[i])):
		zupper[i].append(np.arange(len(LumFuncWhere[i][j])*1.))
		zlower[i].append(np.arange(len(LumFuncWhere[i][j])*1.))
		for k in range(0,len(LumFuncWhere[i][j])):
			if zarr[LumFuncWhere[i][j][k]]<=0.16:
				tempm=b06[LumFuncWhere[i][j][k]]
				tempz=zarr[LumFuncWhere[i][j][k]]
				while np.logical_and(tempm<args.appmax,tempz<=zbinup[i]):
					coeff=kcorrect.fit_nonneg(tempz,karray[LumFuncWhere[i][j][k]][1:6],karray[LumFuncWhere[i][j][k]][6:12])
					m=kcorrect.reconstruct_maggies(coeff)
					tempm=M[LumFuncWhere[i][j][k]]+5*(np.log10(100000*cosmo.luminosity_distance(tempz).value))+(-2.5*np.log10(m[1]/b06rm0[LumFuncWhere[i][j][k]])-5*np.log10(0.6932))
					tempz=tempz+0.01	
				tempz=tempz-0.01
				while np.logical_and(tempm<args.appmax,tempz<=zbinup[i]):
					coeff=kcorrect.fit_nonneg(tempz,karray[LumFuncWhere[i][j][k]][1:6],karray[LumFuncWhere[i][j][k]][6:12])
					m=kcorrect.reconstruct_maggies(coeff)
					tempm=M[LumFuncWhere[i][j][k]]+5*(np.log10(100000*cosmo.luminosity_distance(tempz).value))+(-2.5*np.log10(m[1]/b06rm0[LumFuncWhere[i][j][k]])-5*np.log10(0.6932))
					tempz=tempz+0.001
				tempz=tempz-0.001					
				while np.logical_and(tempm<args.appmax,tempz<=zbinup[i]):
					coeff=kcorrect.fit_nonneg(tempz,karray[LumFuncWhere[i][j][k]][1:6],karray[LumFuncWhere[i][j][k]][6:12])
					m=kcorrect.reconstruct_maggies(coeff)
					tempm=M[LumFuncWhere[i][j][k]]+5*(np.log10(100000*cosmo.luminosity_distance(tempz).value))+(-2.5*np.log10(m[1]/b06rm0[LumFuncWhere[i][j][k]])-5*np.log10(0.6932))
					tempz=tempz+0.0001
				if tempz>=zbinup[i]:
					tempz=zbinup[i]
				zupper[i][j][k]=tempz
				tempm=b06[LumFuncWhere[i][j][k]]
				tempz=zarr[LumFuncWhere[i][j][k]]
				while np.logical_and(tempm>args.appmin,tempz>=zbinlow[i]):
					coeff=kcorrect.fit_nonneg(tempz,karray[LumFuncWhere[i][j][k]][1:6],karray[LumFuncWhere[i][j][k]][6:12])
					m=kcorrect.reconstruct_maggies(coeff)
					tempm=M[LumFuncWhere[i][j][k]]+5*(np.log10(100000*cosmo.luminosity_distance(tempz).value))+(-2.5*np.log10(m[1]/b06rm0[LumFuncWhere[i][j][k]])-5*np.log10(0.6932))
					tempz=tempz-0.01
				tempz=tempz+0.01
				while np.logical_and(tempm>args.appmin,tempz>=zbinlow[i]):
					coeff=kcorrect.fit_nonneg(tempz,karray[LumFuncWhere[i][j][k]][1:6],karray[LumFuncWhere[i][j][k]][6:12])
					m=kcorrect.reconstruct_maggies(coeff)
					tempm=M[LumFuncWhere[i][j][k]]+5*(np.log10(100000*cosmo.luminosity_distance(tempz).value))+(-2.5*np.log10(m[1]/b06rm0[LumFuncWhere[i][j][k]])-5*np.log10(0.6932))
					tempz=tempz-0.001
				tempz=tempz+0.001					
				while np.logical_and(tempm>args.appmin,tempz>=zbinlow[i]):
					coeff=kcorrect.fit_nonneg(tempz,karray[LumFuncWhere[i][j][k]][1:6],karray[LumFuncWhere[i][j][k]][6:12])
					m=kcorrect.reconstruct_maggies(coeff)
					tempm=M[LumFuncWhere[i][j][k]]+5*(np.log10(100000*cosmo.luminosity_distance(tempz).value))+(-2.5*np.log10(m[1]/b06rm0[LumFuncWhere[i][j][k]])-5*np.log10(0.6932))
					tempz=tempz-0.0001
				if tempz>=zbinup[i]:
					tempz=zbinup[i]
				zlower[i][j][k]=tempz
			if zarr[LumFuncWhere[i][j][k]]<=0.31 and zarr[LumFuncWhere[i][j][k]]>0.16:
				tempm=v06[LumFuncWhere[i][j][k]]
				tempz=zarr[LumFuncWhere[i][j][k]]
				while np.logical_and(tempm<args.appmax,tempz<=zbinup[i]):
					coeff=kcorrect.fit_nonneg(tempz,karray[LumFuncWhere[i][j][k]][1:6],karray[LumFuncWhere[i][j][k]][6:12])
					m=kcorrect.reconstruct_maggies(coeff)
					tempm=v06[LumFuncWhere[i][j][k]]+5*(np.log10(100000*cosmo.luminosity_distance(tempz).value))+(-2.5*np.log10(m[2]/v06rm0[LumFuncWhere[i][j][k]])-5*np.log10(0.6932))
					tempz=tempz+0.01	
				tempz=tempz-0.01
				while np.logical_and(tempm<args.appmax,tempz<=zbinup[i]):
					coeff=kcorrect.fit_nonneg(tempz,karray[LumFuncWhere[i][j][k]][1:6],karray[LumFuncWhere[i][j][k]][6:12])
					m=kcorrect.reconstruct_maggies(coeff)
					tempm=v06[LumFuncWhere[i][j][k]]+5*(np.log10(100000*cosmo.luminosity_distance(tempz).value))+(-2.5*np.log10(m[2]/v06rm0[LumFuncWhere[i][j][k]])-5*np.log10(0.6932))
					tempz=tempz+0.001
				tempz=tempz-0.001					
				while np.logical_and(tempm<args.appmax,tempz<=zbinup[i]):
					coeff=kcorrect.fit_nonneg(tempz,karray[LumFuncWhere[i][j][k]][1:6],karray[LumFuncWhere[i][j][k]][6:12])
					m=kcorrect.reconstruct_maggies(coeff)
					tempm=v06[LumFuncWhere[i][j][k]]+5*(np.log10(100000*cosmo.luminosity_distance(tempz).value))+(-2.5*np.log10(m[2]/v06rm0[LumFuncWhere[i][j][k]])-5*np.log10(0.6932))
					tempz=tempz+0.0001
				if tempz>=zbinup[i]:
					tempz=zbinup[i]
				zupper[i][j][k]=tempz
				tempm=v06[LumFuncWhere[i][j][k]]
				tempz=zarr[LumFuncWhere[i][j][k]]
				while np.logical_and(tempm>args.appmin,tempz>=zbinlow[i]):
					coeff=kcorrect.fit_nonneg(tempz,karray[LumFuncWhere[i][j][k]][1:6],karray[LumFuncWhere[i][j][k]][6:12])
					m=kcorrect.reconstruct_maggies(coeff)
					tempm=v06[LumFuncWhere[i][j][k]]+5*(np.log10(100000*cosmo.luminosity_distance(tempz).value))+(-2.5*np.log10(m[2]/v06rm0[LumFuncWhere[i][j][k]])-5*np.log10(0.6932))
					tempz=tempz-0.01
				tempz=tempz+0.01
				while np.logical_and(tempm>args.appmin,tempz>=zbinlow[i]):
					coeff=kcorrect.fit_nonneg(tempz,karray[LumFuncWhere[i][j][k]][1:6],karray[LumFuncWhere[i][j][k]][6:12])
					m=kcorrect.reconstruct_maggies(coeff)
					tempm=v06[LumFuncWhere[i][j][k]]+5*(np.log10(100000*cosmo.luminosity_distance(tempz).value))+(-2.5*np.log10(m[2]/v06rm0[LumFuncWhere[i][j][k]])-5*np.log10(0.6932))
					tempz=tempz-0.001
				tempz=tempz+0.001					
				while np.logical_and(tempm>args.appmin,tempz>=zbinlow[i]):
					coeff=kcorrect.fit_nonneg(tempz,karray[LumFuncWhere[i][j][k]][1:6],karray[LumFuncWhere[i][j][k]][6:12])
					m=kcorrect.reconstruct_maggies(coeff)
					tempm=v06[LumFuncWhere[i][j][k]]+5*(np.log10(100000*cosmo.luminosity_distance(tempz).value))+(-2.5*np.log10(m[2]/v06rm0[LumFuncWhere[i][j][k]])-5*np.log10(0.6932))
					tempz=tempz-0.0001
				if tempz>=zbinup[i]:
					tempz=zbinup[i]
				zlower[i][j][k]=tempz
			if zarr[LumFuncWhere[i][j][k]]<=0.55 and zarr[LumFuncWhere[i][j][k]]>0.31:
				tempm=r06[LumFuncWhere[i][j][k]]
				tempz=zarr[LumFuncWhere[i][j][k]]
				while np.logical_and(tempm<args.appmax,tempz<=zbinup[i]):
					coeff=kcorrect.fit_nonneg(tempz,karray[LumFuncWhere[i][j][k]][1:6],karray[LumFuncWhere[i][j][k]][6:12])
					m=kcorrect.reconstruct_maggies(coeff)
					tempm=r06[LumFuncWhere[i][j][k]]+5*(np.log10(100000*cosmo.luminosity_distance(tempz).value))+(-2.5*np.log10(m[3]/r06rm0[LumFuncWhere[i][j][k]])-5*np.log10(0.6932))
					tempz=tempz+0.01	
				tempz=tempz-0.01
				while np.logical_and(tempm<args.appmax,tempz<=zbinup[i]):
					coeff=kcorrect.fit_nonneg(tempz,karray[LumFuncWhere[i][j][k]][1:6],karray[LumFuncWhere[i][j][k]][6:12])
					m=kcorrect.reconstruct_maggies(coeff)
					tempm=r06[LumFuncWhere[i][j][k]]+5*(np.log10(100000*cosmo.luminosity_distance(tempz).value))+(-2.5*np.log10(m[3]/r06rm0[LumFuncWhere[i][j][k]])-5*np.log10(0.6932))
					tempz=tempz+0.001
				tempz=tempz-0.001					
				while np.logical_and(tempm<args.appmax,tempz<=zbinup[i]):
					coeff=kcorrect.fit_nonneg(tempz,karray[LumFuncWhere[i][j][k]][1:6],karray[LumFuncWhere[i][j][k]][6:12])
					m=kcorrect.reconstruct_maggies(coeff)
					tempm=r06[LumFuncWhere[i][j][k]]+5*(np.log10(100000*cosmo.luminosity_distance(tempz).value))+(-2.5*np.log10(m[3]/r06rm0[LumFuncWhere[i][j][k]])-5*np.log10(0.6932))
					tempz=tempz+0.0001
				if tempz>=zbinup[i]:
					tempz=zbinup[i]
				zupper[i][j][k]=tempz
				tempm=r06[LumFuncWhere[i][j][k]]
				tempz=zarr[LumFuncWhere[i][j][k]]
				while np.logical_and(tempm>args.appmin,tempz>=zbinlow[i]):
					coeff=kcorrect.fit_nonneg(tempz,karray[LumFuncWhere[i][j][k]][1:6],karray[LumFuncWhere[i][j][k]][6:12])
					m=kcorrect.reconstruct_maggies(coeff)
					tempm=r06[LumFuncWhere[i][j][k]]+5*(np.log10(100000*cosmo.luminosity_distance(tempz).value))+(-2.5*np.log10(m[3]/r06rm0[LumFuncWhere[i][j][k]])-5*np.log10(0.6932))
					tempz=tempz-0.01
				tempz=tempz+0.01
				while np.logical_and(tempm>args.appmin,tempz>=zbinlow[i]):
					coeff=kcorrect.fit_nonneg(tempz,karray[LumFuncWhere[i][j][k]][1:6],karray[LumFuncWhere[i][j][k]][6:12])
					m=kcorrect.reconstruct_maggies(coeff)
					tempm=r06[LumFuncWhere[i][j][k]]+5*(np.log10(100000*cosmo.luminosity_distance(tempz).value))+(-2.5*np.log10(m[3]/r06rm0[LumFuncWhere[i][j][k]])-5*np.log10(0.6932))
					tempz=tempz-0.001
				tempz=tempz+0.001					
				while np.logical_and(tempm>args.appmin,tempz>=zbinlow[i]):
					coeff=kcorrect.fit_nonneg(tempz,karray[LumFuncWhere[i][j][k]][1:6],karray[LumFuncWhere[i][j][k]][6:12])
					m=kcorrect.reconstruct_maggies(coeff)
					tempm=r06[LumFuncWhere[i][j][k]]+5*(np.log10(100000*cosmo.luminosity_distance(tempz).value))+(-2.5*np.log10(m[3]/r06rm0[LumFuncWhere[i][j][k]])-5*np.log10(0.6932))
					tempz=tempz-0.0001
				if tempz>=zbinup[i]:
					tempz=zbinup[i]
				zlower[i][j][k]=tempz
			if zarr[LumFuncWhere[i][j][k]]<=0.9 and zarr[LumFuncWhere[i][j][k]]>0.55:
				tempm=i06[LumFuncWhere[i][j][k]]
				tempz=zarr[LumFuncWhere[i][j][k]]
				while np.logical_and(tempm<args.appmax,tempz<=zbinup[i]):
					coeff=kcorrect.fit_nonneg(tempz,karray[LumFuncWhere[i][j][k]][1:6],karray[LumFuncWhere[i][j][k]][6:12])
					m=kcorrect.reconstruct_maggies(coeff)
					tempm=i06[LumFuncWhere[i][j][k]]+5*(np.log10(100000*cosmo.luminosity_distance(tempz).value))+(-2.5*np.log10(m[4]/i06rm0[LumFuncWhere[i][j][k]])-5*np.log10(0.6932))
					tempz=tempz+0.01	
				tempz=tempz-0.01
				while np.logical_and(tempm<args.appmax,tempz<=zbinup[i]):
					coeff=kcorrect.fit_nonneg(tempz,karray[LumFuncWhere[i][j][k]][1:6],karray[LumFuncWhere[i][j][k]][6:12])
					m=kcorrect.reconstruct_maggies(coeff)
					tempm=i06[LumFuncWhere[i][j][k]]+5*(np.log10(100000*cosmo.luminosity_distance(tempz).value))+(-2.5*np.log10(m[4]/i06rm0[LumFuncWhere[i][j][k]])-5*np.log10(0.6932))
					tempz=tempz+0.001
				tempz=tempz-0.001					
				while np.logical_and(tempm<args.appmax,tempz<=zbinup[i]):
					coeff=kcorrect.fit_nonneg(tempz,karray[LumFuncWhere[i][j][k]][1:6],karray[LumFuncWhere[i][j][k]][6:12])
					m=kcorrect.reconstruct_maggies(coeff)
					tempm=i06[LumFuncWhere[i][j][k]]+5*(np.log10(100000*cosmo.luminosity_distance(tempz).value))+(-2.5*np.log10(m[4]/i06rm0[LumFuncWhere[i][j][k]])-5*np.log10(0.6932))
					tempz=tempz+0.0001
				if tempz>=zbinup[i]:
					tempz=zbinup[i]
				zupper[i][j][k]=tempz
				tempm=i06[LumFuncWhere[i][j][k]]
				tempz=zarr[LumFuncWhere[i][j][k]]
				while np.logical_and(tempm>args.appmin,tempz>=zbinlow[i]):
					coeff=kcorrect.fit_nonneg(tempz,karray[LumFuncWhere[i][j][k]][1:6],karray[LumFuncWhere[i][j][k]][6:12])
					m=kcorrect.reconstruct_maggies(coeff)
					tempm=i06[LumFuncWhere[i][j][k]]+5*(np.log10(100000*cosmo.luminosity_distance(tempz).value))+(-2.5*np.log10(m[4]/i06rm0[LumFuncWhere[i][j][k]])-5*np.log10(0.6932))
					tempz=tempz-0.01
				tempz=tempz+0.01
				while np.logical_and(tempm>args.appmin,tempz>=zbinlow[i]):
					coeff=kcorrect.fit_nonneg(tempz,karray[LumFuncWhere[i][j][k]][1:6],karray[LumFuncWhere[i][j][k]][6:12])
					m=kcorrect.reconstruct_maggies(coeff)
					tempm=i06[LumFuncWhere[i][j][k]]+5*(np.log10(100000*cosmo.luminosity_distance(tempz).value))+(-2.5*np.log10(m[4]/i06rm0[LumFuncWhere[i][j][k]])-5*np.log10(0.6932))
					tempz=tempz-0.001
				tempz=tempz+0.001					
				while np.logical_and(tempm>args.appmin,tempz>=zbinlow[i]):
					coeff=kcorrect.fit_nonneg(tempz,karray[LumFuncWhere[i][j][k]][1:6],karray[LumFuncWhere[i][j][k]][6:12])
					m=kcorrect.reconstruct_maggies(coeff)
					tempm=i06[LumFuncWhere[i][j][k]]+5*(np.log10(100000*cosmo.luminosity_distance(tempz).value))+(-2.5*np.log10(m[4]/i06rm0[LumFuncWhere[i][j][k]])-5*np.log10(0.6932))
					tempz=tempz-0.0001
				if tempz>=zbinup[i]:
					tempz=zbinup[i]
				zlower[i][j][k]=tempz
			if zarr[LumFuncWhere[i][j][k]]>0.9:
				tempm=z06[LumFuncWhere[i][j][k]]
				tempz=zarr[LumFuncWhere[i][j][k]]
				while np.logical_and(tempm<args.appmax,tempz<=zbinup[i]):
					coeff=kcorrect.fit_nonneg(tempz,karray[LumFuncWhere[i][j][k]][1:6],karray[LumFuncWhere[i][j][k]][6:12])
					m=kcorrect.reconstruct_maggies(coeff)
					tempm=z06[LumFuncWhere[i][j][k]]+5*(np.log10(100000*cosmo.luminosity_distance(tempz).value))+(-2.5*np.log10(m[5]/z06rm0[LumFuncWhere[i][j][k]])-5*np.log10(0.6932))
					tempz=tempz+0.01	
				tempz=tempz-0.01
				while np.logical_and(tempm<args.appmax,tempz<=zbinup[i]):
					coeff=kcorrect.fit_nonneg(tempz,karray[LumFuncWhere[i][j][k]][1:6],karray[LumFuncWhere[i][j][k]][6:12])
					m=kcorrect.reconstruct_maggies(coeff)
					tempm=z06[LumFuncWhere[i][j][k]]+5*(np.log10(100000*cosmo.luminosity_distance(tempz).value))+(-2.5*np.log10(m[5]/z06rm0[LumFuncWhere[i][j][k]])-5*np.log10(0.6932))
					tempz=tempz+0.001
				tempz=tempz-0.001					
				while np.logical_and(tempm<args.appmax,tempz<=zbinup[i]):
					coeff=kcorrect.fit_nonneg(tempz,karray[LumFuncWhere[i][j][k]][1:6],karray[LumFuncWhere[i][j][k]][6:12])
					m=kcorrect.reconstruct_maggies(coeff)
					tempm=z06[LumFuncWhere[i][j][k]]+5*(np.log10(100000*cosmo.luminosity_distance(tempz).value))+(-2.5*np.log10(m[5]/z06rm0[LumFuncWhere[i][j][k]])-5*np.log10(0.6932))
					tempz=tempz+0.0001
				if tempz>=zbinup[i]:
					tempz=zbinup[i]
				zupper[i][j][k]=tempz
				tempm=z06[LumFuncWhere[i][j][k]]
				tempz=zarr[LumFuncWhere[i][j][k]]
				while np.logical_and(tempm>args.appmin,tempz>=zbinlow[i]):
					coeff=kcorrect.fit_nonneg(tempz,karray[LumFuncWhere[i][j][k]][1:6],karray[LumFuncWhere[i][j][k]][6:12])
					m=kcorrect.reconstruct_maggies(coeff)
					tempm=z06[LumFuncWhere[i][j][k]]+5*(np.log10(100000*cosmo.luminosity_distance(tempz).value))+(-2.5*np.log10(m[5]/z06rm0[LumFuncWhere[i][j][k]])-5*np.log10(0.6932))
					tempz=tempz-0.01
				tempz=tempz+0.01
				while np.logical_and(tempm>args.appmin,tempz>=zbinlow[i]):
					coeff=kcorrect.fit_nonneg(tempz,karray[LumFuncWhere[i][j][k]][1:6],karray[LumFuncWhere[i][j][k]][6:12])
					m=kcorrect.reconstruct_maggies(coeff)
					tempm=z06[LumFuncWhere[i][j][k]]+5*(np.log10(100000*cosmo.luminosity_distance(tempz).value))+(-2.5*np.log10(m[5]/z06rm0[LumFuncWhere[i][j][k]])-5*np.log10(0.6932))
					tempz=tempz-0.001
				tempz=tempz+0.001					
				while np.logical_and(tempm>args.appmin,tempz>=zbinlow[i]):
					coeff=kcorrect.fit_nonneg(tempz,karray[LumFuncWhere[i][j][k]][1:6],karray[LumFuncWhere[i][j][k]][6:12])
					m=kcorrect.reconstruct_maggies(coeff)
					tempm=z06[LumFuncWhere[i][j][k]]+5*(np.log10(100000*cosmo.luminosity_distance(tempz).value))+(-2.5*np.log10(m[5]/z06rm0[LumFuncWhere[i][j][k]])-5*np.log10(0.6932))
					tempz=tempz-0.0001
				if tempz>=zbinup[i]:
					tempz=zbinup[i]
				zlower[i][j][k]=tempz


			###############    SEGMENT 7    ###############


CMV=[]			#Comoving Volume array. Find Comoving Volume for each source. Final Structure CMV[zbin][mbin][comovingvolume]
p=0
q=0
s=0
	#This loop calculates the Comoving Volume for each source (based on the redshift bins above). It uses the comoving_volume utility from astropy, which calculates the comoving volume out to 
	#the designated redshift. I calculate the comoving volume of each source by calculate the comoving volume at the upper redshift bound. I divide by the solid angle of the sphere and multiply 
	#by the solid angle of the COSMOS field. I do the same for the lower redshift bound, and then take the difference between the two values. That leaves the comoving volume for the source.
for p in range(0,len(LumFuncWhere)):
	CMV.append([])
	for q in range(0,len(LumFuncWhere[p])):
		CMV[p].append(np.arange(len(LumFuncWhere[p][q])*1.))
		if len(CMV[p][q])!=0:
			CMV[p][q]=cosmo.comoving_volume(zupper[p][q]).value/(4*np.pi/0.000312)-cosmo.comoving_volume(zlower[p][q]).value/(4*np.pi/(0.000312))		#Comoving volume from astropy calculates full comoving volume at z. Multiply by ratio of solid angle of full sky and solid angle of COSMOS field. Take difference of comoving volume at zmax and comoving volume at zmin for each source to find Max comoving volume the source could fall in and still be part of its bin. 


WeightArray=np.empty_like(b06)
for i in range(0,len(sortweighting)):
	for j in range(0,len(sortweighting[i])):
		zuse1w=len(np.where(zuse[sortweighting[i][j]]==1)[0])*1.0
		zuse2w=len(np.where(zuse[sortweighting[i][j]]==2)[0])*1.0
		zuse3w=len(np.where(zuse[sortweighting[i][j]]==3)[0])*1.0
		zuse4w=len(np.where(zuse[sortweighting[i][j]]==4)[0])*1.0
		if (zuse1w+zuse2w)!=0:
			SSR=(zuse1w+zuse2w+zuse3w)/(zuse1w+zuse2w)
		else:
			SSR=1
		if (zuse1w+zuse2w+zuse3w)!=0:
			TSR=(zuse1w+zuse2w+zuse3w+zuse4w)/(zuse1w+zuse2w+zuse3w)
		else:
			TSR=1
		WeightArray[sortweighting[i][j]]=SSR

np.savetxt('WeightArray.txt',WeightArray)
			###############    SEGMENT 8    ###############


LumFunc=[]		#Array for Luminosity Function Structure=Lumfunc[RedshiftBin][sum(1/CMV) for each MagBin]
LumFuncErr=[]		#Array for Error (Poission Error)
LogErr=[]		#Array for Log(Error)
NGal=[]
Density=np.arange(len(CMV)*1.)
	#This loop calculates the value for the luminosity function and the Poisson errors for each source. Follows paper from Willmer (2006). LumFunc=sum(1/Vmax(i)) over all sources in a given 
	#absolute magnitude bin. 
for i in range(0,len(CMV)):
	LumFunc.append(np.arange(len(CMV[i]))*1.)
	LumFuncErr.append(np.arange(len(CMV[i]))*1.)
	LogErr.append(np.arange(len(CMV[i]))*1.)
	NGal.append(np.arange(len(CMV[i]))*1.)
	for j in range(0,len(CMV[i])):
		val=0.0
		err=0.0
		for k in range(0,len(CMV[i][j])):
			if i06[LumFuncWhere[i][j][k]]>15 and i06[LumFuncWhere[i][j][k]]<22.5:
				val=val+(WeightForSpec*WeightArray[LumFuncWhere[i][j][k]])/(CMV[i][j][k])	#LumFunc=Sum(1/Vmaxi) from i=0 to N  
				if CMV[i][j][k]==0:
					err=err+0		
				else:
					err=err+1./((CMV[i][j][k]*zbinsize)**2)		#Poission Error for Wilmer 2006 is sqrt(sum(1/Vmax**2))
		LumFunc[i][j]=val
		NGal[i][j]=len(CMV[i][j])
		if err==0:
			LumFuncErr[i][j]=1
			LogErr[i][j]=1
		else:
			LumFuncErr[i][j]=np.sqrt(err)
			LogErr[i][j]=LumFuncErr[i][j]/(LumFunc[i][j]*np.log(10))
 #Calculating log(err)
	Density[i]=sum(LumFunc[i])


			###############    SEGMENT 9    ###############


ZBinMid=(zbinup+zbinlow)/2.
MBinMid=(MBinUp+MBinLow)/2.
MRange=(args.Mmax-args.Mmin)
	#Plotting! These commands create plots based on the number of z and M bins. Last few plots could be the same if the number of z bins cannot be easily split. (ex. if you only want 3 z bins,
	#program will create 4 plots, but the bottom two will be the same. 
pp=PdfPages('LuminosityFunctionPlot.pdf')
if args.zbin==1:
	fig1=plt.figure(1) 
	ax1 = fig1.add_subplot(111)
	ax1.errorbar(MBinMid,np.log10(LumFunc[0]),yerr=LogErr[0],fmt='s')
	plt.savefig(pp,format='pdf',orientation='landscape')
	if args.diagnostic:
		fig2=plt.figure(2)
		ax2=fig2.add_subplot(111)
		for i in range(0,len(LumFunc[0])):
			ax2.plot(CMV[0][i],M[LumFuncWhere[0][i]],'.')
	totalgals=0
	for i in range(0,len(CMV[0])):
		totalgals=totalgals+len(CMV[0][i])
	plt.savefig(pp,format='pdf',orientation='landscape')
if args.zbin==2: 
	f,axes=plt.subplots(1,2,sharey=True,sharex=True)	
	for i,a in enumerate(axes.flatten(),start=1):
		a.errorbar(MBinMid,np.log10(LumFunc[i-1]),yerr=LogErr[i-1],fmt='s')
		a.set(adjustable='box-forced')
		a.set_title('z=%.3f-%.3f' % (zbinlow[i-1],zbinup[i-1]))
		a.set_xlim([args.Mmin,args.Mmax])
		a.set_ylim([-5,-1])
	f.text(0.5,0.04,'M$_{B}$ (mag)',ha='center')
	f.text(0.04,0.5,'Log$_{10}(\Phi_{M}$) (Mpc$^{-3} dex^{-1}$)',va='center',rotation='vertical')
	plt.subplots_adjust(wspace=0)
	plt.savefig(pp,format='pdf',orientation='landscape')
	if args.diagnostic:
		f2,axes=plt.subplots(1,2,sharey=True,sharex=True)
		for i,a in enumerate(axes.flatten(),start=1):
			for j in range(0,len(LumFunc[i-1])):
				a.plot(CMV[i-1][j],M[LumFuncWhere[i-1][j]],'.')
			a.set(adjustable='box-forced')
			a.set_title('z=%.3f-%.3f' % (zbinlow[i-1],zbinup[i-1]))
		f2.text(0.5,0.04,'V$_{max}$',ha='center')
		f2.text(0.04,0.5,'M$_{B}$ (mag)',va='center',rotation='vertical')
		f2.tight_layout(rect=[0.05,0.05,1,1])		
		plt.subplots_adjust(wspace=0)
		plt.savefig(pp,format='pdf',orientation='landscape')

		NGAL,axes2=plt.subplots(1,2,sharex=True,sharey=True)
		for i,a in enumerate(axes2.flatten(),start=1):
			a.plot(MBinMid,NGal[i-1],'s')
			a.set(adjustable='box-forced')
			a.set_title('z=%.3f-%.3f' % (zbinlow[i-1],zbinup[i-1]))	
			a.set_xlim([args.Mmin,args.Mmax])		
		NGAL.text(0.5,0.04,'M$_{B}$ (mag)',ha='center')
		NGAL.text(0.04,0.5,'Number of Galaxies',va='center',rotation='vertical')
		NGAL.tight_layout(rect=[0.05,0.05,1,1])		
		plt.subplots_adjust(wspace=0)
		plt.savefig(pp,format='pdf',orientation='landscape')

		Figure=plt.figure()
		plt.plot(ZBinMid,Density,'s')
		plt.title('Density vs. Redshift')
		plt.tight_layout(rect=[0.05,0.05,1,1])
		plt.savefig(pp,format='pdf',orientation='landscape')

if args.zbin>2 and args.zbin<=4:
	f,axes=plt.subplots(2,2,sharex=True,sharey=True)
	i=0	
	if args.overplot:
		phi=np.array([0.00645,0.00490,0.00557,0.00715])
		Mparam=np.array([-20.73,-20.91,-21.14,-21.17])
		alpha2=np.array([-1.03,-1.03,-1.03,-1.03])
	for i,a in enumerate(axes.flatten(),start=1):
		if args.zbin==3 and i>3:
			a.errorbar(MBinMid,np.log10(LumFunc[2]),yerr=LogErr[2],fmt='s')
			a.set(adjustable='box-forced')
			a.set_title('z=%.3f-%.3f' % (zbinlow[2],zbinup[2]))
			a.set_xlim([args.Mmin,args.Mmax])
			a.set_ylim([-5,0])
			if args.overplot:
				schechter_range=np.linspace(args.Mmin,args.Mmax,10000)
				def schechter_fit(logM, phi=0.4*np.log(10)*phi[i+1], log_M0=Mparam[i+1], alpha=alpha2[i+1], e=2.718281828):
					schechter = phi*(10**(0.4*(alpha+1)*(log_M0-logM)))*(e**(-pow(10,(log_M0-logM)*0.4)))
					return schechter
				a.plot(schechter_range,np.log10(schechter_fit(schechter_range)))
		else:
			a.errorbar(MBinMid,np.log10(LumFunc[i-1]),yerr=LogErr[i-1],fmt='s')
			a.set(adjustable='box-forced')
			a.set_title('z=%.3f-%.3f' % (zbinlow[i-1],zbinup[i-1]))
			a.set_xlim([args.Mmin,args.Mmax])
			a.set_ylim([-6,0])
			if args.overplot:
				schechter_range=np.linspace(args.Mmin,args.Mmax,10000)
				def schechter_fit(logM, phi=0.4*np.log(10)*phi[i-1], log_M0=Mparam[i-1], alpha=alpha2[i-1], e=2.718281828):
					schechter = phi*(10**(0.4*(alpha+1)*(log_M0-logM)))*(e**(-pow(10,(log_M0-logM)*0.4)))
					return schechter
				a.plot(schechter_range,np.log10(schechter_fit(schechter_range)))
	f.text(0.5,0.04,'M$_{B}$ (mag)',ha='center')
	f.text(0.04,0.5,'Log$_{10}(\Phi_{M}$) (Mpc$^{-3} dex^{-1}$)',va='center',rotation='vertical')
	f.tight_layout(rect=[0.05,0.05,1,1])		
	plt.subplots_adjust(wspace=0)
	plt.savefig(pp,format='pdf',orientation='landscape')

	if args.diagnostic:
		VmaxvsM,axes1=plt.subplots(2,2,sharex=True,sharey=True)
		for i,a in enumerate(axes1.flatten(),start=1):
			if args.zbin==3 and i>3:
				for j in range(0,len(LumFunc[2])):
					a.plot(CMV[2][j],M[LumFuncWhere[2][j]],'.')
				a.set(adjustable='box-forced')
				a.set_title('z=%.3f-%.3f' % (zbinlow[2],zbinup[2]))
			else:
				for j in range(0,len(LumFunc[i-1])):
					a.plot(CMV[i-1][j],M[LumFuncWhere[i-1][j]],'.')
				a.set(adjustable='box-forced')
				a.set_title('z=%.3f-%.3f' % (zbinlow[i-1],zbinup[i-1]))
		VmaxvsM.text(0.5,0.04,'V$_{max}$',ha='center')
		VmaxvsM.text(0.04,0.5,'M$_{B}$ (mag)',va='center',rotation='vertical')
		VmaxvsM.tight_layout(rect=[0.05,0.05,1,1])		
		plt.subplots_adjust(wspace=0)
		plt.savefig(pp,format='pdf',orientation='landscape')

		NGAL,axes2=plt.subplots(2,2,sharex=True,sharey=True)
		for i,a in enumerate(axes2.flatten(),start=1):
			if args.zbin==3 and i>3:
				a.plot(MBinMid,NGal[2],'s')
				a.set(adjustable='box-forced')
				a.set_title('z=%.3f-%.3f' % (zbinlow[2],zbinup[2]))
			else:
				a.plot(MBinMid,NGal[i-1],'s')
				a.set(adjustable='box-forced')
				a.set_title('z=%.3f-%.3f' % (zbinlow[i-1],zbinup[i-1]))	
				a.set_xlim([args.Mmin,args.Mmax])		
		NGAL.text(0.5,0.04,'M$_{B}$ (mag)',ha='center')
		NGAL.text(0.04,0.5,'Number of Galaxies',va='center',rotation='vertical')
		NGAL.tight_layout(rect=[0.05,0.05,1,1])		
		plt.subplots_adjust(wspace=0)
		plt.savefig(pp,format='pdf',orientation='landscape')
		Figure=plt.figure()
		plt.plot(ZBinMid,Density,'s')
		plt.title('Density vs. Redshift')
		plt.tight_layout(rect=[0.05,0.05,1,1])
		plt.savefig(pp,format='pdf',orientation='landscape')
if args.zbin>4 and args.zbin<=6:
	f,axes=plt.subplots(2,3,sharex=True,sharey=True)
	i=0	
	for i,a in enumerate(axes.flatten(),start=1):
		if args.zbin<6 and i>5:
			a.errorbar(MBinMid,np.log10(LumFunc[4]),yerr=LogErr[4],fmt='s')
			a.set(adjustable='box-forced')
			a.set_title('z=%.3f-%.3f' % (zbinlow[4],zbinup[4]))	
		else:
			a.errorbar(MBinMid,np.log10(LumFunc[i-1]),yerr=LogErr[i-1],fmt='s')
			a.set(adjustable='box-forced')
			a.set_title('z=%.3f-%.3f' % (zbinlow[i-1],zbinup[i-1]))
	f.text(0.5,0.04,'M$_{B}$ (mag)',ha='center')
	f.text(0.04,0.5,'Log$_{10}(\Phi_{M}$) (Mpc$^{-3} dex^{-1}$)',va='center',rotation='vertical')
	f.tight_layout(rect=[0.05,0.05,1,1])
	plt.subplots_adjust(wspace=0)
	plt.savefig(pp,format='pdf',orientation='landscape')

	if args.diagnostic:
		VmaxvsM,axes1=plt.subplots(2,3,sharex=True,sharey=True)
		for i,a in enumerate(axes1.flatten(),start=1):
			if args.zbin<6 and i>5:
				for j in range(0,len(LumFunc[4])):
					a.plot(CMV[4][j],M[LumFuncWhere[4][j]],'.')
				a.set(adjustable='box-forced')
				a.set_title('z=%.3f-%.3f' % (zbinlow[4],zbinup[4]))
			else:
				for j in range(0,len(LumFunc[i-1])):
					a.plot(CMV[i-1][j],M[LumFuncWhere[i-1][j]],'.')
				a.set(adjustable='box-forced')
				a.set_title('z=%.3f-%.3f' % (zbinlow[i-1],zbinup[i-1]))
		VmaxvsM.text(0.5,0.04,'V$_{max}$',ha='center')
		VmaxvsM.text(0.04,0.5,'M$_{B}$ (mag)',va='center',rotation='vertical')
		VmaxvsM.tight_layout(rect=[0.05,0.05,1,1])		
		plt.subplots_adjust(wspace=0)
		plt.savefig(pp,format='pdf',orientation='landscape')

		NGAL,axes2=plt.subplots(2,3,sharex=True,sharey=True)
		for i,a in enumerate(axes2.flatten(),start=1):
			if args.zbin<6 and i>5:
				a.plot(MBinMid,NGal[4],'s')
				a.set(adjustable='box-forced')
				a.set_title('z=%.3f-%.3f' % (zbinlow[4],zbinup[4]))
			else:
				a.plot(MBinMid,NGal[i-1],'s')
				a.set(adjustable='box-forced')
				a.set_title('z=%.3f-%.3f' % (zbinlow[i-1],zbinup[i-1]))	
				a.set_xlim([args.Mmin,args.Mmax])		
		NGAL.text(0.5,0.04,'M$_{B}$ (mag)',ha='center')
		NGAL.text(0.04,0.5,'Number of Galaxies',va='center',rotation='vertical')
		NGAL.tight_layout(rect=[0.05,0.05,1,1])		
		plt.subplots_adjust(wspace=0)
		plt.savefig(pp,format='pdf',orientation='landscape')
		Figure=plt.figure()
		plt.plot(ZBinMid,Density,'s')
		plt.title('Density vs. Redshift')
		plt.tight_layout(rect=[0.05,0.05,1,1])
		plt.savefig(pp,format='pdf',orientation='landscape')
if args.zbin>6 and args.zbin<=9:
	f,axes=plt.subplots(3,3,sharex=True,sharey=True)
	i=0		
	for i,a in enumerate(axes.flatten(),start=1):
		if args.zbin<8 and i>7:
			a.errorbar(MBinMid,np.log10(LumFunc[6]),yerr=LogErr[6],fmt='s')
			a.set(adjustable='box-forced')
			a.set_title('z=%.3f-%.3f' % (zbinlow[6],zbinup[6]))
		elif args.zbin<9 and i>8:
			a.errorbar(MBinMid,np.log10(LumFunc[7]),yerr=LogErr[7],fmt='s')
			a.set(adjustable='box-forced')
			a.set_title('z=%.3f-%.3f' % (zbinlow[7],zbinup[7]))
		else:
			a.errorbar(MBinMid,np.log10(LumFunc[i-1]),yerr=LogErr[i-1],fmt='s')
			a.set(adjustable='box-forced')
			a.set_title('z=%.3f-%.3f' % (zbinlow[i-1],zbinup[i-1]))		
	f.text(0.5,0.04,'M$_{B}$ (mag)',ha='center')
	f.text(0.04,0.5,'Log$_{10}(\Phi_{M}$) (Mpc$^{-3} dex^{-1}$)',va='center',rotation='vertical')
	f.tight_layout(rect=[0.05,0.05,1,1])
	plt.subplots_adjust(wspace=0)
	plt.savefig(pp,format='pdf',orientation='landscape')

	if args.diagnostic:
		VmaxvsM,axes1=plt.subplots(3,3,sharex=True,sharey=True)
		for i,a in enumerate(axes1.flatten(),start=1):
			if args.zbin<8 and i>7:
				for j in range(0,len(LumFunc[6])):
					a.plot(CMV[6][j],M[LumFuncWhere[6][j]],'.')
				a.set(adjustable='box-forced')
				a.set_title('z=%.3f-%.3f' % (zbinlow[6],zbinup[6]))
			elif args.zbin<9 and i>8:
				for j in range(0,len(LumFunc[7])):
					a.plot(CMV[7][j],M[LumFuncWhere[7][j]],'.')
				a.set(adjustable='box-forced')
				a.set_title('z=%.3f-%.3f' % (zbinlow[7],zbinup[7]))
			else:
				for j in range(0,len(LumFunc[i-1])):
					a.plot(CMV[i-1][j],M[LumFuncWhere[i-1][j]],'.')
				a.set(adjustable='box-forced')
				a.set_title('z=%.3f-%.3f' % (zbinlow[i-1],zbinup[i-1]))
		VmaxvsM.text(0.5,0.04,'V$_{max}$',ha='center')
		VmaxvsM.text(0.04,0.5,'M$_{B}$ (mag)',va='center',rotation='vertical')
		VmaxvsM.tight_layout(rect=[0.05,0.05,1,1])		
		plt.subplots_adjust(wspace=0)
		plt.savefig(pp,format='pdf',orientation='landscape')

		NGAL,axes2=plt.subplots(3,3,sharex=True,sharey=True)
		for i,a in enumerate(axes2.flatten(),start=1):
			if args.zbin<8 and i>7:
				a.plot(MBinMid,NGal[6],'s')
				a.set(adjustable='box-forced')
				a.set_title('z=%.3f-%.3f' % (zbinlow[6],zbinup[6]))
			elif args.zbin<9 and i>8:
				a.plot(MBinMid,NGal[7],'s')
				a.set(adjustable='box-forced')
				a.set_title('z=%.3f-%.3f' % (zbinlow[7],zbinup[7]))
			else:
				a.plot(MBinMid,NGal[i-1],'s')
				a.set(adjustable='box-forced')
				a.set_title('z=%.3f-%.3f' % (zbinlow[i-1],zbinup[i-1]))	
				a.set_xlim([args.Mmin,args.Mmax])		
		NGAL.text(0.5,0.04,'M$_{B}$ (mag)',ha='center')
		NGAL.text(0.04,0.5,'Number of Galaxies',va='center',rotation='vertical')
		NGAL.tight_layout(rect=[0.05,0.05,1,1])		
		plt.subplots_adjust(wspace=0)
		plt.savefig(pp,format='pdf',orientation='landscape')
		Figure=plt.figure()
		plt.plot(ZBinMid,Density,'s')
		plt.title('Density')
		plt.tight_layout(rect=[0.05,0.05,1,1])
		plt.savefig(pp,format='pdf',orientation='landscape')
if args.zbin>9:
	print 'Too many redshift bins' 
	sys.exit()
pp.close()


