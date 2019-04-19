import numpy as np
import astropy as ap
import matplotlib.pyplot as plt
import math as m
import argparse
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import custom_model
from matplotlib.backends.backend_pdf import PdfPages

parser = argparse.ArgumentParser()
parser.add_argument("filein",help="File containing magnitudes and redshifts")
parser.add_argument("-d","--diagnostic",action="store_true",help="Include Diagnostic Plot of average Vmax in each bin")
parser.add_argument("-two","--twofiles",action="store_true",help="Do you to add another set of points to plot?")
parser.add_argument("-LCBG","--LCBGfile",type=str,help="input file for LCBG file", default="LCBG_10_35_0.txt")
parser.add_argument("-op","--overplot",type=int,help="Overplot the Schechter Function",default=0)
parser.add_argument("-ps","--phistar",type=float,help="Phi* value for input Schechter Function",default=0.00715)
parser.add_argument("-ms","--mstar",type=float,help="M* value for input Schechter Fucntion",default=-21.17)
parser.add_argument("-al","--schalpha",type=float,help="alpha value for input Schechter Function",default=-1.03)
parser.add_argument("-fo","--fileout",help="Filename of PDF you want to generate",default='LuminosityFunctionPlot.pdf')
parser.add_argument("-r","--Mmin",type=float,help="Minimum redshift to consider in luminosity function, default=--26.7", default=-24)
parser.add_argument("-s","--Mmax",type=float,help="Minimum redshift to consider in luminosity function, default=-18.5", default=-15)

args=parser.parse_args()

MBinMid,MBINAVE,LumFunc,LumFuncErr,LogErr,NGal,AveCMV,AveWeight=np.loadtxt(args.filein,unpack=True,skiprows=1)

if args.twofiles:
	LMBinMid,LMBINAVE,LLumFunc,LLumFuncErr,LLogErr,LNGal,LAveCMV,LAveWeight=np.loadtxt(args.LCBGfile,unpack=True,skiprows=1)
	with open(args.LCBGfile,'r') as lf:
		lspecvals=lf.readline().strip().split()
	lzmax=float(lspecvals[1])
	lzmin=float(lspecvals[2])
	lmbin=float(lspecvals[3])
	lMbinsize=float(lspecvals[4])
	fit=LevMarLSQFitter()
	LLumFuncErr[np.where(LLumFuncErr==0)[0]]=10000
	LCBG_Range=np.linspace(-24,-15,100)

with open(args.filein,'r') as f:
	specvals=f.readline().strip().split()

zmax=float(specvals[1])
zmin=float(specvals[2])
mbin=float(specvals[3])
Mbinsize=float(specvals[4])
maxvol=float(specvals[5])
print(zmax,zmin,mbin)


schechter_range=np.linspace(-24,-15,1000)



def schechter_fit(sample_M, phi=0.4*np.log(10)*args.phistar, M_star=args.mstar, alpha=args.schalpha, e=2.718281828):
	schechter = phi*(10**(0.4*(alpha+1)*(M_star-sample_M)))*(e**(-np.power(10,0.4*(M_star-sample_M))))
	return schechter


@custom_model
def schechter_func(x,phistar=0.0056,mstar=-21,alpha=-1.03):
    return (0.4*np.log(10)*phistar)*(10**(0.4*(alpha+1)*(mstar-x)))*(np.e**(-np.power(10,0.4*(mstar-x))))

print(np.log10(LumFunc))


def autolabel(rects,thecolor):
     for rect in rects:
          height=rect.get_height()
          print(height)
          if not m.isinf(height):
               axes[1].text(rect.get_x() + rect.get_width()/2.,0.7*height,'%d' % int(np.power(10,height)),ha='center',va='bottom',fontsize='small',color=thecolor)


with PdfPages(args.fileout) as pdf:
	f,axes=plt.subplots(nrows=2,ncols=1,sharex=True,gridspec_kw={'height_ratios':[3,1]})
	code=axes[0].errorbar(MBINAVE,np.log10(LumFunc),yerr=LogErr,fmt=',',label='1/V$_{MAX}$ code')
	if args.overplot==1:
		schech_func=axes[0].plot(schechter_range,np.log10(schechter_fit(schechter_range)),label='Schechter Function (Zucca,2009)')
		axes[0].errorbar(np.array([-21.383,-20.306,-19.191,-18.094,-16.947,-15.876]),np.array([-3.0654633,-2.5801028,-2.2969721,-2.2037608,-2.1003195,-1.7932905]),yerr=np.array([0.0685247,0.0444847,0.0316303,0.0338373,0.0604575,0.1100924]),fmt=',',label='Zucca Data')
	elif args.overplot==2:
		schech_func=axes[0].plot(schechter_range,np.log10(schechter_fit(schechter_range)),label='Schechter Function (Zucca,2009)')
		axes[0].errorbar(np.array([-22.097,-21.371,-20.620,-19.838,-19.147,-18.468]),np.array([-3.8243316,-3.0204007,-2.6223342,-2.4411843,-2.3885472,-2.3703618]),yerr=np.array([0.1161,0.0526,0.0361,0.0308,0.0367,0.0742]),fmt=',',label='Zucca Data')
	elif args.overplot==3:
		schech_func=axes[0].plot(schechter_range,np.log10(schechter_fit(schechter_range)),label='Schechter Function (Zucca,2009)')
		axes[0].errorbar(np.array([-22.874,-22.202,-21.538,-20.875,-20.257,-19.688]),np.array([-4.7407928,-3.5406160,-2.9240603,-2.6361774,-2.5375994,-2.8199852]),yerr=np.array([0.15487,0.0735861,0.03979,0.029427,0.028351,0.07468]),fmt=',',label='Zucca Data')
	elif args.overplot==4:
		schech_func=axes[0].plot(schechter_range,np.log10(schechter_fit(schechter_range)),label='Schechter Function (Zucca,2009)')
		axes[0].errorbar(np.array([-23.012,-22.185,-21.376,-20.735,-19.657,-18.616]),np.array([-4.5777,-3.3724774,-2.77235,-2.7633118,-4.567394,-5.085]),yerr=np.array([0.1375,0.04246,0.024944,0.061933,0.3123,0.27809]),fmt=',',label='Zucca Data')
	elif args.overplot==5:
		schech_func=axes[0].plot(schechter_range,np.log10(schechter_fit(schechter_range)),label='Schechter Function (Zucca,2009)')
		axes[0].errorbar(np.array([-22.591,-21.764,-20.934,-20.095,-19.182,-18.35]),np.array([-4.311,-3.218,-2.6232,-2.4194,-2.340,-2.1827]),yerr=np.array([0.0915,0.0406,0.0237,0.0230,0.0282,0.0522]),fmt=',',label='Zucca Data')
	if args.twofiles:
		LCBGFIT_init=schechter_func()
		LCBG_FIT=fit(LCBGFIT_init,LMBINAVE[np.where(LNGal>2)[0]],LLumFunc[np.where(LNGal>2)[0]],weights=1/LLumFuncErr[np.where(LNGal>2)[0]])
		print(LCBG_FIT)
		axes[0].errorbar(LMBINAVE,np.log10(LLumFunc),yerr=LLogErr,fmt=',',label='1/V$_{MAX}$ LCBG')
		axes[0].plot(LCBG_Range,np.log10(LCBG_FIT(LCBG_Range)),label='LCBG Schechter Fit')
	f.text(0.04,0.65,'Log$_{10}(\Phi_{M}$) (Mpc$^{-3} mag^{-1}$)',va='center',rotation='vertical')
	axes[0].legend(loc=4,fontsize='small')
	axes[0].set_ylim([-6,-1])
	axes[0].set_title('z=%.3f-%.3f' %(zmin,zmax))
	plt.xlim((args.Mmin,args.Mmax))
	ndist=axes[1].bar(MBinMid,np.log10(NGal),Mbinsize,align='center',label='Number of sources per Absolute Magnitude Bin',color='white')
	if args.twofiles:
		lcbg=axes[1].bar(LMBinMid,np.log10(LNGal),lMbinsize,align='center',label='Lumber of LCBGs per Absolute Magnitude Bin',color='gray')
		autolabel(lcbg,'black')
	autolabel(ndist,'black')
	plt.xlabel('Absolute Magnitude (M$_{B}$)')
	plt.ylabel('Log$_{10}$(N)',fontsize='small')
	plt.subplots_adjust(hspace=0)
	pdf.savefig(orientation='landhahape')
	if args.diagnostic:
		fig2=plt.figure(2)
		for i in range(0,len(AveCMV)):
			plt.plot(MBinMid[i],AveCMV[i],'.')
		plt.ylabel('Comoving Volume (Mpc$^{-3}$)')
		plt.xlabel('Absolute Magnitude (M)')
		plt.xlim((args.Mmin,args.Mmax))
		plt.plot(np.linspace(args.Mmin,args.Mmax,10),np.full(10,maxvol))
		pdf.savefig(orientation='landscape')
		fig3=plt.figure(3)
		for i in range(0,len(AveWeight)):
			plt.plot(MBinMid[i],AveWeight[i],'.')
		plt.ylabel('Weight')
		plt.xlabel('Absolute Magnitude (M)')
		plt.xlim((args.Mmin,args.Mmax))
		pdf.savefig(orientation='landscape')
		
