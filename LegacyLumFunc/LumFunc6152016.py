import sys
import numpy as np
import astropy as ap
import matplotlib.pyplot as plt
import math as m
import astropy as ap
from astropy.cosmology import WMAP9 as cosmo
from matplotlib.backends.backend_pdf import PdfPages
try:
	
	
			###############    SEGMENT 1    ###############
			
			
	if sys.argv[1] == '-h':
		print '''This is a program to make a luminosity function from a file of Absolute Magnitude and redshift. 
It takes an argument of an input file, number of bins for redshift, number of bins for luminosity, max redshift, min redshift, min Magnitude, max Magnitude.
It should create a plot (or plots) of a luminosity function using the 1/Vmax method.
Takes arguments Tab Seperated Variable File, Number of Redshift Bins (Default=4), Number of Luminosity Bins (Default=17).
Example python LumFunc5172016.py <FileIn> <#zbins> <#Mbins> <zmin> <zmax> <Mmin> <Mmax>
-d after calling the program will include plots showing Vmax vs Magnitude, Number of Galaxies per M bin, and Density (Sum of Luminosity Function) vs Redshift'''
		sys.exit()
	if sys.argv[1] == '-d':		
		diagnostic=1
		length=len(sys.argv)-1
	else:
		length=len(sys.argv)
		diagnostic=0	
	if length == 2:
		filein=sys.argv[1+diagnostic]
		b06,M,zarr=np.loadtxt(filein,skiprows=1,unpack=True)	#creates apparent and absolute magnitud arrays and redshift array
		zbin=4
		mbin=17
		zmin=0.5		#Hard redshift. Set to make sure there are enough galaxies in a bin or that our volume is large enough 
		zmax=1.3
		Mmin=-26.6231		#Hardcoded Absolute Magnitude. The highest calculated absolute magnitude for objects in current sample 	
		Mmax=-18.5211
	elif length == 3:
		filein=sys.argv[1+diagnostic]
		b06,M,zarr=np.loadtxt(filein,skiprows=1,unpack=True) 
		zbin=int(sys.argv[2+diagnostic])
		mbin=17
		zmin=0.5
		zmax=1.3
		Mmin=-26.6231
		Mmax=-18.5211
	elif length == 4:
		filein=sys.argv[1+diagnostic]
		b06,M,zarr=np.loadtxt(filein,skiprows=1,unpack=True)
		zbin=int(sys.argv[2+diagnostic])
		mbin=int(sys.argv[3+diagnostic])
		zmin=0.5
		zmax=1.3
		Mmin=-26.6231			
		Mmax=-18.5211
	elif length == 5:
		filein=sys.argv[1+diagnostic]
		b06,M,zarr=np.loadtxt(filein,skiprows=1,unpack=True)
		zbin=int(sys.argv[2+diagnostic])
		mbin=int(sys.argv[3+diagnostic])
		zmin=np.float32(sys.argv[4+diagnostic])
		zmax=1.3
		Mmin=-26.6231			
		Mmax=-18.5211
	elif length == 6:
		filein=sys.argv[1+diagnostic]
		b06,M,zarr=np.loadtxt(filein,skiprows=1,unpack=True)
		zbin=int(sys.argv[2+diagnostic])
		mbin=int(sys.argv[3+diagnostic])
		zmin=np.float32(sys.argv[4+diagnostic])
		zmax=np.float32(sys.argv[5+diagnostic])
		Mmin=-26.6231
		Mmax=-18.5211
	elif length == 7:
		filein=sys.argv[1+diagnostic]
		b06,M,zarr=np.loadtxt(filein,skiprows=1,unpack=True)
		zbin=int(sys.argv[2+diagnostic])
		mbin=int(sys.argv[3+diagnostic])
		zmin=np.float32(sys.argv[4+diagnostic])
		zmax=np.float32(sys.argv[5+diagnostic])
		Mmin=np.float32(sys.argv[6+diagnostic])
		Mmax=-18.5211
	#Can input all variables, number of redshift bins (zbin), number of Magnitude bins (mbin), minimum redshift (zmin), maximum redshift (zmax), minum Magnitude (Mmin), maximum Magnitude (Mmax) 
	elif length == 8:
		filein=sys.argv[1+diagnostic]
		b06,M,zarr=np.loadtxt(filein,skiprows=1,unpack=True)
		zbin=int(sys.argv[2+diagnostic])
		mbin=int(sys.argv[3+diagnostic])
		zmin=np.float32(sys.argv[4+diagnostic])
		zmax=np.float32(sys.argv[5+diagnostic])
		Mmin=np.float32(sys.argv[6+diagnostic])
		Mmax=np.float32(sys.argv[7+diagnostic])
	else:
		print 'Wrong number of variables'
		sys.exit()
	
	
			###############    SEGMENT 2    ###############
	
	
	
	alpha=1.596*m.pi/180		#angular size of COSMOS pointing (Degrees) 
	zdif=zmax-zmin			#Calculates difference between max and min redshift
	zbinsize=zdif/zbin		#Creates bin size based on input for number of bins
	Mdif=Mmax-Mmin			#Calculates difference between max and min Magnitude
	Mbinsize=Mdif/mbin		#Creates bin size based on input for number of bins
	zVariable=zmin			#Creates a variable to use in loops through redshift
	MPopsByz=[]			#List for galaxies binned by redshift
	zBinWhere=[]			#Index. Allows us to index zarr and b06 later.
	i=0				#Variable used in for loop
	zbinup=np.arange(zbin)*1.	#Upper bound for zbin
	zbinlow=np.arange(zbin)*1.	#Lower bound for zbin
	#This loop goes through all the sources and separates them based on redshift.
	#MPopsByz is filled with the Magnitude of the objects in a given redshift bin. zBinWhere gives the index for b06,M,zarr (b06[MPopsByz[0]],M[MPopsByz[0]],zarr[MPopsByz[0]] will give m,M,z for the same source)
	while (i<zbin):
		MPopsByz.append(M[np.where((zarr>=zVariable) & (zarr<zVariable+zbinsize))])
		zBinWhere.append(np.where((zarr>=zVariable) & (zarr<zVariable+zbinsize))[0])
		zbinlow[i]=zVariable				
		zVariable=zVariable+zbinsize
		zbinup[i]=zVariable
		i=i+1
	
	
			###############    SEGMENT 3    ###############
	
	
	i=0
	zVariable=zmin		
	LumFuncWhere=[]			#Index array. To point back to b06,zarr,M for respective values. 
	MBinLow=np.arange(mbin)*1.	#Lower bound for Mbin. If we have 12 bins and the upper and lower bounds for Magnitude are -18.5 and -24.5 then each bin is 0.5 magnitude. MBinLow[0]=-18.5
	MBinUp=np.arange(mbin)*1.	#Upper bound for Mbin. If we have 12 bins and the upper and lower bounds for Magnitude are -18.5 and -24.5 then each bin is 0.5 magnitude. MBinUp[0]=-19.0
	# Calculate index array. Will sort sources by redshift; then in each redshift bin, will sort by Magnitude. The numbers stored are indexes for b06,M,zarr Structure=LumFuncWhere[RedshiftBin][MagnitudeBin][Index pointing to b06,zarr,M] 
	while (i<zbin):
		LumFuncWhere.append([])
		j=0
		MVariable=Mmin
		while (j<mbin):
			LumFuncWhere[i].append(zBinWhere[i][np.where((MPopsByz[i]<=MVariable+Mbinsize) & (MPopsByz[i]>MVariable))])
			MBinUp[j]=MVariable
			MVariable=MVariable+Mbinsize
			MBinLow[j]=MVariable
			j=j+1
		i=i+1
		zVariable=zVariable+zbinsize
		
		
			###############    SEGMENT 4    ###############
			
			
	zVariable=zmin
	MVariable=Mmin
	zcut=np.arange(1000)		#Array that will be used to find max and minimum redshift each source can have and still exist in its z,M bin. 
	findcut=1000./zmax
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
	kcorr=[]
	for i in range(0,1000):
		kcorr.append(0.1055*m.pow(zcutt[i],6)-0.6393*m.pow(zcutt[i],5)+0.977*m.pow(zcutt[i],4)+1.1534*m.pow(zcutt[i],3)-4.6431*m.pow(zcutt[i],2)+3.8905*zcutt[i]-0.0483)
	
	#mnm is an array for distance modulus. Each element in this array is the distance modulus for the minute redshift bins in zcutt.
	mnm=[]
	mnm.append(0)
	i=1
	for i in range(1,1000):
		mnm.append(5*np.log10(100000*dl[i])+kcorr[i])
	
	
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
				zupper[i][j][k]=min(zcutt[len(np.where(mnm<(b06[LumFuncWhere[i][j][k]]-MBinUp[j]))[0])-1],zbinup[i])
				zlower[i][j][k]=max(zcutt[len(np.where(mnm<(b06[LumFuncWhere[i][j][k]]-MBinLow[j]))[0])-1],zbinlow[i])
	
	
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
				CMV[p][q]=cosmo.comoving_volume(zupper[p][q]).value/(4*m.pi/(2*m.pi*(1-m.cos(alpha))))-cosmo.comoving_volume(zlower[p][q]).value/(4*m.pi/(2*m.pi*(1-m.cos(alpha))))		#Comoving volume from astropy calculates full comoving volume at z. Multiply by ratio of solid angle of full sky and solid angle of COSMOS field. Take difference of comoving volume at zmax and comoving volume at zmin for each source to find Max comoving volume the source could fall in and still be part of its bin. 
	
	
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
				val=val+1./(CMV[i][j][k])	#LumFunc=Sum(1/Vmaxi) from i=0 to N  
				if CMV[i][j][k]==0:
					err=err+0		
				else:
					err=err+1./((CMV[i][j][k]*zbinsize)**2)		#Poission Error for Wilmer 2006 is sqrt(sum(1/Vmax**2))
			LumFunc[i][j]=val/zbinsize
			NGal[i][j]=len(CMV[i][j])
			if err==0:
				LumFuncErr[i][j]=0
				LogErr[i][j]=0
			else:
				LumFuncErr[i][j]=m.sqrt(err)
				LogErr[i][j]=LumFuncErr[i][j]/(LumFunc[i][j]*np.log(10)) #Calculating log(err)
		Density[i]=sum(LumFunc[i])
		
		
			###############    SEGMENT 9    ###############
	
	
	ZBinMid=(zbinup+zbinlow)/2.
	MBinMid=(MBinUp+MBinLow)/2.
	#Plotting! These commands create plots based on the number of z and M bins. Last few plots could be the same if the number of z bins cannot be easily split. (ex. if you only want 3 z bins,
	#program will create 4 plots, but the bottom two will be the same. 
	pp=PdfPages('LuminosityFunctionPlot.pdf')
	if zbin==1:
		fig1=plt.figure(1) 
		ax1 = fig1.add_subplot(111)
		ax1.errorbar(MBinMid,np.log10(LumFunc[0]),yerr=LogErr[0],fmt='s')
		plt.savefig(pp,format='pdf',orientation='landscape')
		if diagnostic==1:
			fig2=plt.figure(2)
			ax2=fig2.add_subplot(111)
			for i in range(0,len(LumFunc[0])):
				ax2.plot(CMV[0][i],M[LumFuncWhere[0][i]],'.')
		plt.savefig(pp,format='pdf',orientation='landscape')
	if zbin==2: 
		f,axes=plt.subplots(1,2,sharey=True,sharex=True)	
		for i,a in enumerate(axes.flatten(),start=1):
			a.errorbar(MBinMid,np.log10(LumFunc[i-1]),yerr=LogErr[i-1],fmt='s')
			a.set(adjustable='box-forced')
			a.set_title('z=%.3f-%.3f' % (zbinlow[i-1],zbinup[i-1]))
			a.set_xlim([-25,-18.5])
		f.text(0.5,0.04,'M$_{B}$ (mag)',ha='center')
		f.text(0.04,0.5,'Log$_{10}(\Phi_{M}$) (Mpc$^{-3} dex^{-1}$)',va='center',rotation='vertical')
		plt.subplots_adjust(wspace=0)
		plt.savefig(pp,format='pdf',orientation='landscape')
		if diagnostic==1:
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
				a.set_xlim([-25,-18.5])		
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
			
	if zbin>2 and zbin<=4:
		f,axes=plt.subplots(2,2,sharex=True,sharey=True)
		i=0	
		for i,a in enumerate(axes.flatten(),start=1):
			if zbin==3 and i>3:
				a.errorbar(MBinMid,np.log10(LumFunc[2]),yerr=LogErr[2],fmt='s')
    				a.set(adjustable='box-forced')
				a.set_title('z=%.3f-%.3f' % (zbinlow[2],zbinup[2]))
			else:
    				a.errorbar(MBinMid,np.log10(LumFunc[i-1]),yerr=LogErr[i-1],fmt='s')
    				a.set(adjustable='box-forced')
				a.set_title('z=%.3f-%.3f' % (zbinlow[i-1],zbinup[i-1]))
				a.set_xlim([-25,-18.5])
		f.text(0.5,0.04,'M$_{B}$ (mag)',ha='center')
		f.text(0.04,0.5,'Log$_{10}(\Phi_{M}$) (Mpc$^{-3} dex^{-1}$)',va='center',rotation='vertical')
		f.tight_layout(rect=[0.05,0.05,1,1])		
		plt.subplots_adjust(wspace=0)
		plt.savefig(pp,format='pdf',orientation='landscape')
		
		if diagnostic==1:

			VmaxvsM,axes1=plt.subplots(2,2,sharex=True,sharey=True)
			for i,a in enumerate(axes1.flatten(),start=1):
				if zbin==3 and i>3:
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
				if zbin==3 and i>3:
					a.plot(MBinMid,NGal[2],'s')
    					a.set(adjustable='box-forced')
					a.set_title('z=%.3f-%.3f' % (zbinlow[2],zbinup[2]))
				else:
					a.plot(MBinMid,NGal[i-1],'s')
    					a.set(adjustable='box-forced')
					a.set_title('z=%.3f-%.3f' % (zbinlow[i-1],zbinup[i-1]))	
					a.set_xlim([-25,-18.5])		
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
	if zbin>4 and zbin<=6:
		f,axes=plt.subplots(2,3,sharex=True,sharey=True)
		i=0		
		for i,a in enumerate(axes.flatten(),start=1):
			if zbin<6 and i>5:
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
		
		if diagnostic==1:

			VmaxvsM,axes1=plt.subplots(2,3,sharex=True,sharey=True)
			for i,a in enumerate(axes1.flatten(),start=1):
				if zbin<6 and i>5:
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
				if zbin<6 and i>5:
					a.plot(MBinMid,NGal[4],'s')
    					a.set(adjustable='box-forced')
					a.set_title('z=%.3f-%.3f' % (zbinlow[4],zbinup[4]))
				else:
					a.plot(MBinMid,NGal[i-1],'s')
    					a.set(adjustable='box-forced')
					a.set_title('z=%.3f-%.3f' % (zbinlow[i-1],zbinup[i-1]))	
					a.set_xlim([-25,-18.5])		
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
	if zbin>6 and zbin<=9:
		f,axes=plt.subplots(3,3,sharex=True,sharey=True)
		i=0		
		for i,a in enumerate(axes.flatten(),start=1):
			if zbin<8 and i>7:
				a.errorbar(MBinMid,np.log10(LumFunc[6]),yerr=LogErr[6],fmt='s')
    				a.set(adjustable='box-forced')
				a.set_title('z=%.3f-%.3f' % (zbinlow[6],zbinup[6]))
			elif zbin<9 and i>8:
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
		
		if diagnostic==1:

			VmaxvsM,axes1=plt.subplots(3,3,sharex=True,sharey=True)
			for i,a in enumerate(axes1.flatten(),start=1):
				if zbin<8 and i>7:
					for j in range(0,len(LumFunc[6])):
						a.plot(CMV[6][j],M[LumFuncWhere[6][j]],'.')
					a.set(adjustable='box-forced')
					a.set_title('z=%.3f-%.3f' % (zbinlow[6],zbinup[6]))
				elif zbin<9 and i>8:
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
				if zbin<8 and i>7:
					a.plot(MBinMid,NGal[6],'s')
    					a.set(adjustable='box-forced')
					a.set_title('z=%.3f-%.3f' % (zbinlow[6],zbinup[6]))
				elif zbin<9 and i>8:
					a.plot(MBinMid,NGal[7],'s')
    					a.set(adjustable='box-forced')
					a.set_title('z=%.3f-%.3f' % (zbinlow[7],zbinup[7]))
				else:
					a.plot(MBinMid,NGal[i-1],'s')
    					a.set(adjustable='box-forced')
					a.set_title('z=%.3f-%.3f' % (zbinlow[i-1],zbinup[i-1]))	
					a.set_xlim([-25,-18.5])		
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
	if zbin>9:
		print 'Too many redshift bins' 
		sys.exit()
	pp.close()
	
	
except IndexError:
	print 'Wrong number of variables'

