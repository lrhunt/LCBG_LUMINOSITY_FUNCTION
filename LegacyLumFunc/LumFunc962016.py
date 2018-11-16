# -*- coding: utf-8 -*-
import sys
import numpy as np
import astropy as ap
import matplotlib.pyplot as plt
import math as m
import argparse
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

args=parser.parse_args()

b06,b06err,v06,v06err,r06,r06err,i06,i06err,z06,z06err,zarr,classif,zuse=np.loadtxt(args.filein,skiprows=1,unpack=True)	#creates apparent and absolute magnitud arrays and redshift array

			###############    SEGMENT 2    ###############
zuse1=len(np.where(zuse==1)[0])*1.0
zuse2=len(np.where(zuse==2)[0])*1.0
zuse3=len(np.where(zuse==3)[0])*1.0
zuse4=len(np.where(zuse==4)[0])*1.0
TargetedForSpec=(zuse1+zuse2+zuse3)/(zuse1+zuse2+zuse3+zuse4)
WeightForSpec=1/TargetedForSpec
alpha=1.596*m.pi/180				#angular size of COSMOS pointing (Degrees) 
zdif=args.zmax-args.zmin			#Calculates difference between max and min redshift
zbinsize=zdif/args.zbin				#Creates bin size based on input for number of bins
Mdif=args.Mmax-args.Mmin			#Calculates difference between max and min Magnitude
Mbinsize=Mdif/args.mbin				#Creates bin size based on input for number of bins
print(Mdif,Mbinsize)
zVariable=args.zmin				#Creates a variable to use in loops through redshift
MPopsByz=[]					#List for galaxies binned by redshift
zBinWhere=[]					#Index. Allows us to index zarr and b06 later.
i=0						#Variable used in for loop
zbinup=np.arange(args.zbin)*1.			#Upper bound for zbin
zbinlow=np.arange(args.zbin)*1.			#Lower bound for zbin
zarrrm,b06rm,v06rm,r06rm,f81406rm,z06rm=np.loadtxt('outmaggies.txt',unpack=True)
zarr0,b06rm0,v06rm0,r06rm0,f81406rm0,z06rm0=np.loadtxt('outmaggies_0.txt',unpack=True)
srange=np.where(np.logical_and(zarrrm>0,zarrrm<1.3))
kb=np.zeros_like(b06,dtype=np.float)
kv=np.zeros_like(b06,dtype=np.float)
kr=np.zeros_like(b06,dtype=np.float)
ki=np.zeros_like(b06,dtype=np.float)
kz=np.zeros_like(b06,dtype=np.float)

kb[srange]=-2.5*np.log10(b06rm[srange]/b06rm0[srange])
kv[srange]=-2.5*np.log10(v06rm[srange]/v06rm0[srange])
kr[srange]=-2.5*np.log10(r06rm[srange]/r06rm0[srange])
ki[srange]=-2.5*np.log10(f81406rm[srange]/f81406rm0[srange])
kz[srange]=-2.5*np.log10(z06rm[srange]/z06rm0[srange])

#kB=np.empty_like(b06,dtype='float')
#kV=np.empty_like(b06,dtype='float')
#ibin=args.mbin
#colorbin=args.mbin
#for i in range(0,len(zarr)):
#	if zarr[i]>0:
#		if classif[i]==1 or classif[i]==0:
#			kB[i]=-1.9315*m.pow(zarr[i],4)+3.1222*m.pow(zarr[i],3)-1.208*m.pow(zarr[i],2)+4.476*zarr[i]+0.0536
#			kV[i]=-0.3551*m.pow(zarr[i],4)+1.7259*m.pow(zarr[i],3)-2.0001*m.pow(zarr[i],2)+4.3154*zarr[i]-0.2803
#		if classif[i]==2:
#			kB[i]=1.4862*m.pow(zarr[i],3)-4.4552*m.pow(zarr[i],2)+3.7959*zarr[i]-0.0422
#			kV[i]=3.2176*m.pow(zarr[i],4)-9.1893*m.pow(zarr[i],3)+7.3239*m.pow(zarr[i],2)-0.1406*zarr[i]+0.0189
#		if classif[i]==3:
#			kB[i]=-0.0957*m.pow(zarr[i],3)-1.9941*m.pow(zarr[i],2)+4.3893*zarr[i]-0.00005
#			kV[i]=-1.6856*m.pow(zarr[i],4)-5.3385*m.pow(zarr[i],3)+4.6353*m.pow(zarr[i],2)+1.4993*zarr[i]-0.0366
#
M=np.empty_like(b06,dtype='float')
i=0

for i in range(0,len(zarr)):
	if zarr[i]>0:
		if zarr[i]<=0.16:
			M[i]=b06[i]-5*(m.log10(100000*cosmo.luminosity_distance(zarr[i]).value))-kb[i]
		if zarr[i]<=0.31 and zarr[i]>0.16:
			M[i]=v06[i]-5*(m.log10(100000*cosmo.luminosity_distance(zarr[i]).value))-kv[i]
		if zarr[i]<=0.55 and zarr[i]>0.31:
			M[i]=r06[i]-5*(m.log10(100000*cosmo.luminosity_distance(zarr[i]).value))-kr[i]
		if zarr[i]<=0.9 and zarr[i]>0.55:
			M[i]=i06[i]-5*(m.log10(100000*cosmo.luminosity_distance(zarr[i]).value))-ki[i]
		if zarr[i]>0.9:
			M[i]=z06[i]-5*(m.log10(100000*cosmo.luminosity_distance(zarr[i]).value))-kz[i]

bv=b06-v06
for i in range(0,len(bv)):
	if (bv[i]==0 and b06[i]==-999.98999):
		bv[i]=-999.98999

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
	
	
			###############    SEGMENT 4    ###############


zVariable=args.zmin
MVariable=args.Mmin
zcut=np.arange(1000)		#Array that will be used to find max and minimum redshift each source can have and still exist in its z,M bin. 
findcut=1000./args.zmax
zcut=zcut/findcut		#making each element a redshift, difference between two redshifts is 1/findcut
zcutt=[]			
i=0
	#This loop will limit the number of digits in zcutt (5 digits)
for i in range(0,1000):
	zcutt.append(int(zcut[i]*1000+0.5)/1000.0)	

i=0


			###############    SEGMENT 5    ###############


	#Creating array for Luminosity Distance and k-correction for each z. Value for k-correction found by fitting a curve to k-correction data from Poggianti, 1997
dl=cosmo.luminosity_distance(zcutt).value

	#mnm is an array for distance modulus. Each element in this array is the distance modulus for the minute redshift bins in zcutt.
mnm1=[]
mnm1.append(0)
i=1
for i in range(1,1000):
	mnm1.append(5*np.log10(100000*dl[i]))


			###############    SEGMENT 6    ###############


i=0
j=0
k=0
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
				zupper[i][j][k]=min(zcutt[len(np.where((mnm1+kb[LumFuncWhere[i][j][k]])<(b06[LumFuncWhere[i][j][k]]-MBinUp[j]))[0])-1],zbinup[i])
				zlower[i][j][k]=max(zcutt[len(np.where((mnm1+kb[LumFuncWhere[i][j][k]])<(b06[LumFuncWhere[i][j][k]]-MBinLow[j]))[0])-1],zbinlow[i])
			if zarr[LumFuncWhere[i][j][k]]<=0.31 and zarr[LumFuncWhere[i][j][k]]>0.16:
				zupper[i][j][k]=min(zcutt[len(np.where((mnm1+kv[LumFuncWhere[i][j][k]])<(v06[LumFuncWhere[i][j][k]]-MBinUp[j]))[0])-1],zbinup[i])
				zlower[i][j][k]=max(zcutt[len(np.where((mnm1+kv[LumFuncWhere[i][j][k]])<(v06[LumFuncWhere[i][j][k]]-MBinLow[j]))[0])-1],zbinlow[i])
			if zarr[LumFuncWhere[i][j][k]]<=0.55 and zarr[LumFuncWhere[i][j][k]]>0.31:
				zupper[i][j][k]=min(zcutt[len(np.where((mnm1+kr[LumFuncWhere[i][j][k]])<(r06[LumFuncWhere[i][j][k]]-MBinUp[j]))[0])-1],zbinup[i])
				zlower[i][j][k]=max(zcutt[len(np.where((mnm1+kr[LumFuncWhere[i][j][k]])<(r06[LumFuncWhere[i][j][k]]-MBinLow[j]))[0])-1],zbinlow[i])
			if zarr[LumFuncWhere[i][j][k]]<=0.9 and zarr[LumFuncWhere[i][j][k]]>0.55:
				zupper[i][j][k]=min(zcutt[len(np.where((mnm1+ki[LumFuncWhere[i][j][k]])<(i06[LumFuncWhere[i][j][k]]-MBinUp[j]))[0])-1],zbinup[i])
				zlower[i][j][k]=max(zcutt[len(np.where((mnm1+ki[LumFuncWhere[i][j][k]])<(i06[LumFuncWhere[i][j][k]]-MBinLow[j]))[0])-1],zbinlow[i])
			if zarr[LumFuncWhere[i][j][k]]>0.9:
				zupper[i][j][k]=min(zcutt[len(np.where((mnm1+kz[LumFuncWhere[i][j][k]])<(z06[LumFuncWhere[i][j][k]]-MBinUp[j]))[0])-1],zbinup[i])
				zlower[i][j][k]=max(zcutt[len(np.where((mnm1+kz[LumFuncWhere[i][j][k]])<(z06[LumFuncWhere[i][j][k]]-MBinLow[j]))[0])-1],zbinlow[i])


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
			CMV[p][q]=cosmo.comoving_volume(zupper[p][q]).value/(4*m.pi/0.000312)-cosmo.comoving_volume(zlower[p][q]).value/(4*m.pi/(0.000312))		#Comoving volume from astropy calculates full comoving volume at z. Multiply by ratio of solid angle of full sky and solid angle of COSMOS field. Take difference of comoving volume at zmax and comoving volume at zmin for each source to find Max comoving volume the source could fall in and still be part of its bin. 


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
			#print(WeightArray[LumFuncWhere[i][j][k]],CMV[i][j][k])
			val=val+(WeightForSpec*WeightArray[LumFuncWhere[i][j][k]])/(CMV[i][j][k])	#LumFunc=Sum(1/Vmaxi) from i=0 to N  
			if CMV[i][j][k]==0:
				err=err+0		
			else:
				err=err+1./((CMV[i][j][k]*zbinsize)**2)		#Poission Error for Wilmer 2006 is sqrt(sum(1/Vmax**2))
		LumFunc[i][j]=val/zbinsize
		print(LumFunc[i][j])
		NGal[i][j]=len(CMV[i][j])
		if err==0:
			LumFuncErr[i][j]=1
			LogErr[i][j]=1
		else:
			LumFuncErr[i][j]=m.sqrt(err)
			LogErr[i][j]=LumFuncErr[i][j]/(LumFunc[i][j]*np.log(10)) #Calculating log(err)
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
		print(len(CMV[0][i]))
		totalgals=totalgals+len(CMV[0][i])
	print(totalgals)
	plt.savefig(pp,format='pdf',orientation='landscape')
if args.zbin==2: 
	f,axes=plt.subplots(1,2,sharey=True,sharex=True)	
	for i,a in enumerate(axes.flatten(),start=1):
		a.errorbar(MBinMid,np.log10(LumFunc[i-1]),yerr=LogErr[i-1],fmt='s')
		a.set(adjustable='box-forced')
		a.set_title('z=%.3f-%.3f' % (zbinlow[i-1],zbinup[i-1]))
		a.set_xlim([args.Mmin,args.Mmax])
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
	for i,a in enumerate(axes.flatten(),start=1):
		if args.zbin==3 and i>3:
			a.errorbar(MBinMid,np.log10(LumFunc[2]),yerr=LogErr[2],fmt='s')
			a.set(adjustable='box-forced')
			a.set_title('z=%.3f-%.3f' % (zbinlow[2],zbinup[2]))
		else:
			a.errorbar(MBinMid,np.log10(LumFunc[i-1]),yerr=LogErr[i-1],fmt='s')
			a.set(adjustable='box-forced')
			a.set_title('z=%.3f-%.3f' % (zbinlow[i-1],zbinup[i-1]))
			a.set_xlim([args.Mmin,args.Mmax])
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


