import numpy as np
import astropy as ap
import matplotlib.pyplot as plt
import math as m
import argparse
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import custom_model
from matplotlib.backends.backend_pdf import PdfPages
import scipy as sp
import uncertainties as unc
from uncertainties import unumpy as unp


#*******************************
#*******************************
#*******************************
#PLOTTING LUMINOSITY FUNCTION FOR PAPER
#*******************************
#*******************************
#*******************************



filein='/home/lrhunt/LUM_FUNC/LCBGLFOUT/LCBG_LF_0_20.txt'
sfilein='/home/lrhunt/LUM_FUNC/FULLLFOUT/LF_0_20_FULL.txt'
LMBinMid,LMBINAVE,LLumFunc,LLumFuncErr,LLogErr,LNGal,LAveCMV,LAveWeight=np.loadtxt(filein,unpack=True,skiprows=1)
MBinMid,MBINAVE,LumFunc,LumFuncErr,LogErr,NGal,AveCMV,AveWeight=np.loadtxt(sfilein,unpack=True,skiprows=1)

LFGAL020=sum(NGal)
LCBGGAL020=sum(LNGal)

with open(filein,'r') as lf:
    lspecvals=lf.readline().strip().split()
with open(sfilein,'r') as lf:
	specvals=lf.readline().strip().split()
lzmax=float(lspecvals[1])
lzmin=float(lspecvals[2])
lmbin=float(lspecvals[3])
lMbinsize=float(lspecvals[4])
zmax=float(specvals[1])
zmin=float(specvals[2])
mbin=float(specvals[3])
Mbinsize=float(specvals[4])
fit=LevMarLSQFitter()

def autolabel(rects,thecolor,row,col):
         for rect in rects:
                  height=rect.get_height()
                  print(height)
                  if not m.isinf(height):
                           axes[row][col].text(rect.get_x() + rect.get_width()/2.,height+0.05,'{}'.format(int(np.power(10,height))),ha='center',va='bottom',fontsize=7,color=thecolor)

@custom_model
def schechter_func_gal(x,phistar=0.0056,mstar=-21):
    return (0.4*np.log(10)*phistar)*(10**(0.4*(-1.053+1)*(mstar-x)))*(np.e**(-np.power(10,0.4*(mstar-x))))

@custom_model
def schechter_func_lcbg(x,phistar=0.0056,mstar=-21):
    return (0.4*np.log(10)*phistar)*(10**(0.4*(-0.63+1)*(mstar-x)))*(np.e**(-np.power(10,0.4*(mstar-x))))


def schechter_func_scipy_gal(x,phistar,mstar):
	return (0.4*np.log(10)*phistar)*(10**(0.4*(-1.053+1)*(mstar-x)))*(np.e**(-np.power(10,0.4*(mstar-x))))

def schechter_func_scipy_lcbg(x,phistar,mstar):
	return (0.4*np.log(10)*phistar)*(10**(0.4*(-0.63+1)*(mstar-x)))*(np.e**(-np.power(10,0.4*(mstar-x))))


LCBGFIT_init=schechter_func_lcbg()
GALFIT_init=schechter_func_gal()
LCBG_Range=np.linspace(-24,-15,30)

init_vals=[0.0056,-21]	#Best guess at initial values, needed for scipy fitting

#PLOTTING 0<z<0.2

#Creating Mask Arrays (to easily remove points that are generally off

LLumFunc2=np.ma.array(LLumFunc,mask=False)
LMBINAVE2=np.ma.array(LMBINAVE,mask=False)
LLumFuncErr2=np.ma.array(LLumFuncErr,mask=False)
LumFunc2=np.ma.array(LumFunc,mask=False)
MBINAVE2=np.ma.array(MBINAVE,mask=False)
LumFuncErr2=np.ma.array(LumFuncErr,mask=False)

#Masking zeros in LCBG Luminosity Function

LLumFunc2.mask[np.where(LLumFunc2==0)[0]]=True
LMBINAVE2.mask[np.where(LLumFunc2==0)[0]]=True
LLumFuncErr2.mask[np.where(LLumFunc2==0)[0]]=True

#Masking zeros in Luminosity Function

LumFunc2.mask[np.where(LumFunc2==0)[0]]=True
MBINAVE2.mask[np.where(LumFunc2==0)[0]]=True
LumFuncErr2.mask[np.where(LumFunc2==0)[0]]=True

#Masking errant points in LCBG Luminosity Function

LLumFunc2.mask[6]=True
LMBINAVE2.mask[6]=True
LLumFuncErr2.mask[6]=True
LLumFunc2.mask[8]=True
LMBINAVE2.mask[8]=True
LLumFuncErr2.mask[8]=True
LumFunc2.mask[2:4]=True
MBINAVE2.mask[2:4]=True
LumFuncErr2.mask[2:4]=True

#Astropy Modelling

LCBG_FIT020=fit(LCBGFIT_init,LMBINAVE2.compressed(),LLumFunc2.compressed(),weights=1/LLumFuncErr2.compressed())
LUMFUNC_FIT020=fit(GALFIT_init,MBINAVE2.compressed(),LumFunc2.compressed(),weights=1/LumFuncErr2.compressed())
LCBG_FIT020u=fit(LCBGFIT_init,LMBINAVE2.compressed(),LLumFunc2.compressed()+LLumFuncErr2.compressed(),weights=1/LLumFuncErr2.compressed())
LCBG_FIT020l=fit(LCBGFIT_init,LMBINAVE2.compressed(),LLumFunc2.compressed()-LLumFuncErr2.compressed(),weights=1/LLumFuncErr2.compressed())

#Scipy Modelling

scipy_LCBG_020_fit,scipy_LCBG_020_cov=sp.optimize.curve_fit(schechter_func_scipy_lcbg,LMBINAVE2.compressed(),LLumFunc2.compressed(),p0=init_vals,sigma=LLumFuncErr2.compressed())
scipy_LUMFUNC_020_fit,scipy_LUMFUNC_020_cov=sp.optimize.curve_fit(schechter_func_scipy_gal,MBINAVE2.compressed(),LumFunc2.compressed(),p0=init_vals,sigma=LumFuncErr2.compressed())

#Defining Errors on fit parameters from scipy

LCBG020ERRORS=np.array([np.sqrt(scipy_LCBG_020_cov[0][0]),np.sqrt(scipy_LCBG_020_cov[1][1])])
LUMFUNC020ERRORS=np.array([np.sqrt(scipy_LUMFUNC_020_cov[0][0]),np.sqrt(scipy_LUMFUNC_020_cov[1][1])])

#Plotting
x=np.linspace(-23,-16,8)
xminor=np.linspace(-23.5,-15.5,9)
y=np.linspace(-7,-1,7)
yminor=np.linspace(-6.5,-1.5,6)

f,axes=plt.subplots(nrows=4,ncols=3,sharex=True,gridspec_kw={'height_ratios':[3,1,3,1]})
LCBGcode=axes[0][0].errorbar(LMBINAVE,np.log10(LLumFunc),yerr=LLogErr,fmt=',',label='1/V$_{MAX}$ code',color='blue')
code=axes[0][0].errorbar(MBINAVE,np.log10(LumFunc),yerr=LogErr,fmt=',',label='1/V$_{MAX}$ code',color='green')
axes[0][0].errorbar(MBINAVE[LumFunc2.mask],np.log10(LumFunc[LumFunc2.mask]),yerr=LogErr[LumFunc2.mask],fmt='x',color='green')
axes[0][0].errorbar(LMBINAVE[LLumFunc2.mask],np.log10(LLumFunc[LLumFunc2.mask]),yerr=LLogErr[LLumFunc2.mask],fmt='x',label='1/V$_{MAX}$ code',color='blue')
gals=axes[1][0].bar(MBinMid,np.log10(NGal),Mbinsize,align='center',label='Number of Galaxies per Absolute Magnitude Bin',color='white')
lcbg=axes[1][0].bar(LMBinMid,np.log10(LNGal),lMbinsize,align='center',label='Number of LCBGs per Absolute Magnitude Bin',color='gray')
axes[0][0].plot(LCBG_Range,np.log10(LCBG_FIT020(LCBG_Range)))
axes[0][0].plot(LCBG_Range,np.log10(LUMFUNC_FIT020(LCBG_Range)))
axes[0][0].plot(LCBG_Range,np.log10(LCBG_FIT020u(LCBG_Range)),color='red')
axes[0][0].plot(LCBG_Range,np.log10(LCBG_FIT020l(LCBG_Range)),color='yellow')
plt.subplots_adjust(hspace=0,wspace=0)
axes[0][0].set_xticks(x)
axes[0][0].set_xticks(xminor,minor=True)
axes[0][0].set_yticks(y)
axes[0][0].set_yticks(yminor,minor=True)
axes[0][0].set_ylim([-7.5,-0.5])
axes[1][0].set_yticks([3,2,1,0])
axes[1][0].set_yticks([3.5,2.5,1.5,0.5],minor=True)
axes[1][0].set_ylim([0,4])
autolabel(lcbg,'black',1,0)
autolabel(gals,'black',1,0)

#PLOTTING 0.2<z<0.4

#Defining Files to Read in

filein='/home/lrhunt/LUM_FUNC/LCBGLFOUT/LCBG_LF_20_40.txt'
sfilein='/home/lrhunt/LUM_FUNC/FULLLFOUT/LF_20_40_FULL.txt'

#Reading in Luminosity Function Files

LMBinMid,LMBINAVE,LLumFunc,LLumFuncErr,LLogErr,LNGal,LAveCMV,LAveWeight=np.loadtxt(filein,unpack=True,skiprows=1)
MBinMid,MBINAVE,LumFunc,LumFuncErr,LogErr,NGal,AveCMV,AveWeight=np.loadtxt(sfilein,unpack=True,skiprows=1)

LFGAL2040=sum(NGal)
LCBGGAL2040=sum(LNGal)


#Reading in Header parameters

with open(filein,'r') as lf:
    lspecvals=lf.readline().strip().split()
with open(sfilein,'r') as lf:
	specvals=lf.readline().strip().split()
lzmax=float(lspecvals[1])
lzmin=float(lspecvals[2])
lmbin=float(lspecvals[3])
lMbinsize=float(lspecvals[4])
zmax=float(specvals[1])
zmin=float(specvals[2])
mbin=float(specvals[3])
Mbinsize=float(specvals[4])

#Creating Mask Arrays (to easily remove points that are generally off

LLumFunc2=np.ma.array(LLumFunc,mask=False)
LMBINAVE2=np.ma.array(LMBINAVE,mask=False)
LLumFuncErr2=np.ma.array(LLumFuncErr,mask=False)
LumFunc2=np.ma.array(LumFunc,mask=False)
MBINAVE2=np.ma.array(MBINAVE,mask=False)
LumFuncErr2=np.ma.array(LumFuncErr,mask=False)

#Masking zeros in LCBG Luminosity Function

LLumFunc2.mask[np.where(LLumFunc2==0)[0]]=True
LMBINAVE2.mask[np.where(LLumFunc2==0)[0]]=True
LLumFuncErr2.mask[np.where(LLumFunc2==0)[0]]=True

#Masking zeros in LCBG Luminosity Function

LumFunc2.mask[np.where(LumFunc2==0)[0]]=True
MBINAVE2.mask[np.where(LumFunc2==0)[0]]=True
LumFuncErr2.mask[np.where(LumFunc2==0)[0]]=True

#Masking Errant Points

LLumFunc2.mask[11:12]=True
LMBINAVE2.mask[11:12]=True
LLumFuncErr2.mask[11:12]=True
LLumFunc2.mask[2]=True
LMBINAVE2.mask[2]=True
LLumFuncErr2.mask[2]=True
LumFunc2.mask[14:16]=True
MBINAVE2.mask[14:16]=True
LumFuncErr2.mask[14:16]=True
LumFunc2.mask[1:3]=True
MBINAVE2.mask[1:3]=True
LumFuncErr2.mask[1:3]=True

#Astropy fitting

LCBG_FIT2040=fit(LCBGFIT_init,LMBINAVE2.compressed(),LLumFunc2.compressed(),weights=1/LLumFuncErr2.compressed())
LUMFUNC_FIT2040=fit(GALFIT_init,MBINAVE2.compressed(),LumFunc2.compressed(),weights=1/LumFuncErr2.compressed())
LCBG_FIT2040u=fit(LCBGFIT_init,LMBINAVE2.compressed(),LLumFunc2.compressed()+LLumFuncErr2.compressed(),weights=1/LLumFuncErr2.compressed())
LCBG_FIT2040l=fit(LCBGFIT_init,LMBINAVE2.compressed(),LLumFunc2.compressed()-LLumFuncErr2.compressed(),weights=1/LLumFuncErr2.compressed())

#Scipy Modelling

scipy_LCBG_2040_fit,scipy_LCBG_2040_cov=sp.optimize.curve_fit(schechter_func_scipy_lcbg,LMBINAVE2.compressed(),LLumFunc2.compressed(),p0=init_vals,sigma=LLumFuncErr2.compressed())
scipy_LUMFUNC_2040_fit,scipy_LUMFUNC_2040_cov=sp.optimize.curve_fit(schechter_func_scipy_gal,MBINAVE2.compressed(),LumFunc2.compressed(),p0=init_vals,sigma=LumFuncErr2.compressed())

#Defining Errors on fit parameters from scipy

LCBG2040ERRORS=np.array([np.sqrt(scipy_LCBG_2040_cov[0][0]),np.sqrt(scipy_LCBG_2040_cov[1][1])])
LUMFUNC2040ERRORS=np.array([np.sqrt(scipy_LUMFUNC_2040_cov[0][0]),np.sqrt(scipy_LUMFUNC_2040_cov[1][1])])

#Plotting


LCBGcode=axes[0][1].errorbar(LMBINAVE,np.log10(LLumFunc),yerr=LLogErr,fmt=',',label='1/V$_{MAX}$ code',color='blue')
code=axes[0][1].errorbar(MBINAVE,np.log10(LumFunc),yerr=LogErr,fmt=',',label='1/V$_{MAX}$ code',color='green')
axes[0][1].errorbar(MBINAVE[LumFunc2.mask],np.log10(LumFunc[LumFunc2.mask]),yerr=LogErr[LumFunc2.mask],fmt='x',color='green')
axes[0][1].errorbar(LMBINAVE[LLumFunc2.mask],np.log10(LLumFunc[LLumFunc2.mask]),yerr=LLogErr[LLumFunc2.mask],fmt='x',label='1/V$_{MAX}$ code',color='blue')
gals=axes[1][1].bar(MBinMid,np.log10(NGal),Mbinsize,align='center',label='Number of Galaxies per Absolute Magnitude Bin',color='white')
lcbg=axes[1][1].bar(LMBinMid,np.log10(LNGal),lMbinsize,align='center',label='Number of LCBGs per Absolute Magnitude Bin',color='gray')
axes[0][1].plot(LCBG_Range,np.log10(LCBG_FIT2040(LCBG_Range)))
axes[0][1].plot(LCBG_Range,np.log10(LUMFUNC_FIT2040(LCBG_Range)))
axes[0][1].plot(LCBG_Range,np.log10(LCBG_FIT2040u(LCBG_Range)),color='red')
axes[0][1].plot(LCBG_Range,np.log10(LCBG_FIT2040l(LCBG_Range)),color='yellow')
plt.subplots_adjust(hspace=0,wspace=0)
axes[0][1].set_yticks(y)
axes[0][1].set_yticks(yminor,minor=True)
axes[0][1].set_ylim([-7.5,-0.5])
axes[1][1].set_yticks([3,2,1,0])
axes[1][1].set_yticks([3.5,2.5,1.5,0.5],minor=True)
axes[1][1].set_ylim([0,4])
autolabel(lcbg,'black',1,1)
autolabel(gals,'black',1,1)

#PLOTTING 0.4<z<0.6

#New Files
filein='/home/lrhunt/LUM_FUNC/LCBGLFOUT/LCBG_LF_40_60.txt'
sfilein='/home/lrhunt/LUM_FUNC/FULLLFOUT/LF_40_60_FULL.txt'

#Reading in to array

LMBinMid,LMBINAVE,LLumFunc,LLumFuncErr,LLogErr,LNGal,LAveCMV,LAveWeight=np.loadtxt(filein,unpack=True,skiprows=1)
MBinMid,MBINAVE,LumFunc,LumFuncErr,LogErr,NGal,AveCMV,AveWeight=np.loadtxt(sfilein,unpack=True,skiprows=1)
with open(filein,'r') as lf:
    lspecvals=lf.readline().strip().split()
with open(sfilein,'r') as lf:
	specvals=lf.readline().strip().split()
lzmax=float(lspecvals[1])
lzmin=float(lspecvals[2])
lmbin=float(lspecvals[3])
lMbinsize=float(lspecvals[4])
zmax=float(specvals[1])
zmin=float(specvals[2])
mbin=float(specvals[3])
Mbinsize=float(specvals[4])
fit=LevMarLSQFitter()

LFGAL4060=sum(NGal)
LCBGGAL4060=sum(LNGal)


#Redefining arrays to mask them

LLumFunc2=np.ma.array(LLumFunc,mask=False)
LMBINAVE2=np.ma.array(LMBINAVE,mask=False)
LLumFuncErr2=np.ma.array(LLumFuncErr,mask=False)
LumFunc2=np.ma.array(LumFunc,mask=False)
MBINAVE2=np.ma.array(MBINAVE,mask=False)
LumFuncErr2=np.ma.array(LumFuncErr,mask=False)

#Masking zeros

LLumFunc2.mask[np.where(LLumFunc2==0)[0]]=True
LMBINAVE2.mask[np.where(LLumFunc2==0)[0]]=True
LLumFuncErr2.mask[np.where(LLumFunc2==0)[0]]=True
LumFunc2.mask[np.where(LumFunc2==0)[0]]=True
MBINAVE2.mask[np.where(LumFunc2==0)[0]]=True
LumFuncErr2.mask[np.where(LumFunc2==0)[0]]=True

#Masking errant points

LumFunc2.mask[11:15]=True
MBINAVE2.mask[11:15]=True
LumFuncErr2.mask[11:15]=True
#LumFunc2.mask[8]=True
#MBINAVE2.mask[8]=True
#LumFuncErr2.mask[8]=True
#LLumFunc2.mask[8]=True
#LMBINAVE2.mask[8]=True
#LLumFuncErr2.mask[8]=True

#Astropy Fitting

LCBG_FIT4060=fit(LCBGFIT_init,LMBINAVE2.compressed(),LLumFunc2.compressed(),weights=1/LLumFuncErr2.compressed())
LUMFUNC_FIT4060=fit(GALFIT_init,MBINAVE2.compressed(),LumFunc2.compressed(),weights=1/LumFuncErr2.compressed())
LCBG_FIT4060u=fit(LCBGFIT_init,LMBINAVE2.compressed(),LLumFunc2.compressed()+LLumFuncErr2.compressed(),weights=1/LLumFuncErr2.compressed())
LCBG_FIT4060l=fit(LCBGFIT_init,LMBINAVE2.compressed(),LLumFunc2.compressed()-LLumFuncErr2.compressed(),weights=1/LLumFuncErr2.compressed())

#Scipy Modelling

scipy_LCBG_4060_fit,scipy_LCBG_4060_cov=sp.optimize.curve_fit(schechter_func_scipy_lcbg,LMBINAVE2.compressed(),LLumFunc2.compressed(),p0=init_vals,sigma=LLumFuncErr2.compressed())
scipy_LUMFUNC_4060_fit,scipy_LUMFUNC_4060_cov=sp.optimize.curve_fit(schechter_func_scipy_gal,MBINAVE2.compressed(),LumFunc2.compressed(),p0=init_vals,sigma=LumFuncErr2.compressed())

#Defining Errors on fit parameters from scipy

LCBG4060ERRORS=np.array([np.sqrt(scipy_LCBG_4060_cov[0][0]),np.sqrt(scipy_LCBG_4060_cov[1][1])])
LUMFUNC4060ERRORS=np.array([np.sqrt(scipy_LUMFUNC_4060_cov[0][0]),np.sqrt(scipy_LUMFUNC_4060_cov[1][1])])


LCBGcode=axes[0][2].errorbar(LMBINAVE,np.log10(LLumFunc),yerr=LLogErr,fmt=',',label='1/V$_{MAX}$ code',color='blue')
code=axes[0][2].errorbar(MBINAVE,np.log10(LumFunc),yerr=LogErr,fmt=',',label='1/V$_{MAX}$ code',color='green')
axes[0][2].errorbar(MBINAVE[LumFunc2.mask],np.log10(LumFunc[LumFunc2.mask]),yerr=LogErr[LumFunc2.mask],fmt='x',color='green')
axes[0][2].errorbar(LMBINAVE[LLumFunc2.mask],np.log10(LLumFunc[LLumFunc2.mask]),yerr=LLogErr[LLumFunc2.mask],fmt='x',label='1/V$_{MAX}$ code',color='blue')
gals=axes[1][2].bar(MBinMid,np.log10(NGal),Mbinsize,align='center',label='Number of Galaxies per Absolute Magnitude Bin',color='white')
lcbg=axes[1][2].bar(LMBinMid,np.log10(LNGal),lMbinsize,align='center',label='Number of LCBGs per Absolute Magnitude Bin',color='gray')
axes[0][2].plot(LCBG_Range,np.log10(LCBG_FIT4060(LCBG_Range)))
axes[0][2].plot(LCBG_Range,np.log10(LUMFUNC_FIT4060(LCBG_Range)))
axes[0][2].plot(LCBG_Range,np.log10(LCBG_FIT4060u(LCBG_Range)),color='red')
axes[0][2].plot(LCBG_Range,np.log10(LCBG_FIT4060l(LCBG_Range)),color='yellow')
plt.subplots_adjust(hspace=0,wspace=0)
axes[0][2].set_yticks(y)
axes[0][2].set_yticks(yminor,minor=True)
axes[0][2].set_ylim([-7.5,-0.5])
axes[1][2].set_yticks([3,2,1,0])
axes[1][2].set_yticks([3.5,2.5,1.5,0.5],minor=True)
axes[1][2].set_ylim([0,4])
autolabel(lcbg,'black',1,2)
autolabel(gals,'black',1,2)

#PLOTTING 0.6<z<0.8

filein='/home/lrhunt/LUM_FUNC/LCBGLFOUT/LCBG_LF_60_80.txt'
sfilein='/home/lrhunt/LUM_FUNC/FULLLFOUT/LF_60_80_FULL.txt'


LMBinMid,LMBINAVE,LLumFunc,LLumFuncErr,LLogErr,LNGal,LAveCMV,LAveWeight=np.loadtxt(filein,unpack=True,skiprows=1)
MBinMid,MBINAVE,LumFunc,LumFuncErr,LogErr,NGal,AveCMV,AveWeight=np.loadtxt(sfilein,unpack=True,skiprows=1)
with open(filein,'r') as lf:
    lspecvals=lf.readline().strip().split()
with open(sfilein,'r') as lf:
	specvals=lf.readline().strip().split()
lzmax=float(lspecvals[1])
lzmin=float(lspecvals[2])
lmbin=float(lspecvals[3])
lMbinsize=float(lspecvals[4])
zmax=float(specvals[1])
zmin=float(specvals[2])
mbin=float(specvals[3])
Mbinsize=float(specvals[4])
fit=LevMarLSQFitter()

LFGAL6080=sum(NGal)
LCBGGAL6080=sum(LNGal)


LLumFunc2=np.ma.array(LLumFunc,mask=False)
LMBINAVE2=np.ma.array(LMBINAVE,mask=False)
LLumFuncErr2=np.ma.array(LLumFuncErr,mask=False)
LumFunc2=np.ma.array(LumFunc,mask=False)
MBINAVE2=np.ma.array(MBINAVE,mask=False)
LumFuncErr2=np.ma.array(LumFuncErr,mask=False)



LLumFunc2.mask[np.where(LLumFunc2==0)[0]]=True
LMBINAVE2.mask[np.where(LLumFunc2==0)[0]]=True
LLumFuncErr2.mask[np.where(LLumFunc2==0)[0]]=True
LumFunc2.mask[np.where(LumFunc2==0)[0]]=True
MBINAVE2.mask[np.where(LumFunc2==0)[0]]=True
LumFuncErr2.mask[np.where(LumFunc2==0)[0]]=True



LumFunc2.mask[9]=True
MBINAVE2.mask[9]=True
LumFuncErr2.mask[9]=True
LLumFunc2.mask[9]=True
LMBINAVE2.mask[9]=True
LLumFuncErr2.mask[9]=True
LLumFunc2.mask[0:3]=True
LMBINAVE2.mask[0:3]=True
LLumFuncErr2.mask[0:3]=True



LCBG_FIT6080=fit(LCBGFIT_init,LMBINAVE2.compressed(),LLumFunc2.compressed(),weights=1/LLumFuncErr2.compressed())
LUMFUNC_FIT6080=fit(GALFIT_init,MBINAVE2.compressed(),LumFunc2.compressed(),weights=1/LumFuncErr2.compressed())
LCBG_FIT6080u=fit(LCBGFIT_init,LMBINAVE2.compressed(),LLumFunc2.compressed()+LLumFuncErr2.compressed(),weights=1/LLumFuncErr2.compressed())
LCBG_FIT6080l=fit(LCBGFIT_init,LMBINAVE2.compressed(),LLumFunc2.compressed()-LLumFuncErr2.compressed(),weights=1/LLumFuncErr2.compressed())

#Scipy Modelling

scipy_LCBG_6080_fit,scipy_LCBG_6080_cov=sp.optimize.curve_fit(schechter_func_scipy_lcbg,LMBINAVE2.compressed(),LLumFunc2.compressed(),p0=init_vals,sigma=LLumFuncErr2.compressed())
scipy_LUMFUNC_6080_fit,scipy_LUMFUNC_6080_cov=sp.optimize.curve_fit(schechter_func_scipy_gal,MBINAVE2.compressed(),LumFunc2.compressed(),p0=init_vals,sigma=LumFuncErr2.compressed())

#Defining Errors on fit parameters from scipy

LCBG6080ERRORS=np.array([np.sqrt(scipy_LCBG_6080_cov[0][0]),np.sqrt(scipy_LCBG_6080_cov[1][1])])
LUMFUNC6080ERRORS=np.array([np.sqrt(scipy_LUMFUNC_6080_cov[0][0]),np.sqrt(scipy_LUMFUNC_6080_cov[1][1])])


LCBGcode=axes[2][0].errorbar(LMBINAVE,np.log10(LLumFunc),yerr=LLogErr,fmt=',',label='1/V$_{MAX}$ code',color='blue')
code=axes[2][0].errorbar(MBINAVE,np.log10(LumFunc),yerr=LogErr,fmt=',',label='1/V$_{MAX}$ code',color='green')
axes[2][0].errorbar(MBINAVE[LumFunc2.mask],np.log10(LumFunc[LumFunc2.mask]),yerr=LogErr[LumFunc2.mask],fmt='x',color='green')
axes[2][0].errorbar(LMBINAVE[LLumFunc2.mask],np.log10(LLumFunc[LLumFunc2.mask]),yerr=LLogErr[LLumFunc2.mask],fmt='x',label='1/V$_{MAX}$ code',color='blue')
gals=axes[3][0].bar(MBinMid,np.log10(NGal),Mbinsize,align='center',label='Number of Galaxies per Absolute Magnitude Bin',color='white')
lcbg=axes[3][0].bar(LMBinMid,np.log10(LNGal),lMbinsize,align='center',label='Number of LCBGs per Absolute Magnitude Bin',color='gray')
axes[2][0].plot(LCBG_Range,np.log10(LCBG_FIT6080(LCBG_Range)))
axes[2][0].plot(LCBG_Range,np.log10(LUMFUNC_FIT6080(LCBG_Range)))
axes[2][0].plot(LCBG_Range,np.log10(LCBG_FIT6080u(LCBG_Range)),color='red')
axes[2][0].plot(LCBG_Range,np.log10(LCBG_FIT6080l(LCBG_Range)),color='yellow')
plt.subplots_adjust(hspace=0,wspace=0)
axes[2][0].set_yticks(y)
axes[2][0].set_yticks(yminor,minor=True)
axes[2][0].set_ylim([-7.5,-0.5])
axes[3][0].set_yticks([3,2,1,0])
axes[3][0].set_yticks([3.5,2.5,1.5,0.5],minor=True)
axes[3][0].set_ylim([0,4])
autolabel(lcbg,'black',3,0)
autolabel(gals,'black',3,0)

#PLOTTING 0.8<z<1

filein='/home/lrhunt/LUM_FUNC/LCBGLFOUT/LCBG_LF_80_100.txt'
sfilein='/home/lrhunt/LUM_FUNC/FULLLFOUT/LF_80_100_FULL.txt'
LMBinMid,LMBINAVE,LLumFunc,LLumFuncErr,LLogErr,LNGal,LAveCMV,LAveWeight=np.loadtxt(filein,unpack=True,skiprows=1)
MBinMid,MBINAVE,LumFunc,LumFuncErr,LogErr,NGal,AveCMV,AveWeight=np.loadtxt(sfilein,unpack=True,skiprows=1)
with open(filein,'r') as lf:
    lspecvals=lf.readline().strip().split()
with open(sfilein,'r') as lf:
	specvals=lf.readline().strip().split()
lzmax=float(lspecvals[1])
lzmin=float(lspecvals[2])
lmbin=float(lspecvals[3])
lMbinsize=float(lspecvals[4])
zmax=float(specvals[1])
zmin=float(specvals[2])
mbin=float(specvals[3])
Mbinsize=float(specvals[4])
fit=LevMarLSQFitter()

LFGAL80100=sum(NGal)
LCBGGAL80100=sum(LNGal)

LLumFunc2=np.ma.array(LLumFunc,mask=False)
LMBINAVE2=np.ma.array(LMBINAVE,mask=False)
LLumFuncErr2=np.ma.array(LLumFuncErr,mask=False)
LumFunc2=np.ma.array(LumFunc,mask=False)
MBINAVE2=np.ma.array(MBINAVE,mask=False)
LumFuncErr2=np.ma.array(LumFuncErr,mask=False)

LLumFunc2.mask[np.where(LLumFunc2==0)[0]]=True
LMBINAVE2.mask[np.where(LLumFunc2==0)[0]]=True
LLumFuncErr2.mask[np.where(LLumFunc2==0)[0]]=True
LumFunc2.mask[np.where(LumFunc2==0)[0]]=True
MBINAVE2.mask[np.where(LumFunc2==0)[0]]=True
LumFuncErr2.mask[np.where(LumFunc2==0)[0]]=True

LumFunc2.mask[7]=True
MBINAVE2.mask[7]=True
LumFuncErr2.mask[7]=True
LLumFunc2.mask[7]=True
LMBINAVE2.mask[7]=True
LLumFuncErr2.mask[7]=True

LumFunc2.mask[0]=True
MBINAVE2.mask[0]=True
LumFuncErr2.mask[0]=True

LCBG_FIT80100=fit(LCBGFIT_init,LMBINAVE2.compressed(),LLumFunc2.compressed(),weights=1/LLumFuncErr2.compressed())
LUMFUNC_FIT80100=fit(GALFIT_init,MBINAVE2.compressed(),LumFunc2.compressed(),weights=1/LumFuncErr2.compressed())
LCBG_FIT80100u=fit(LCBGFIT_init,LMBINAVE2.compressed(),LLumFunc2.compressed()+LLumFuncErr2.compressed(),weights=1/LLumFuncErr2.compressed())
LCBG_FIT80100l=fit(LCBGFIT_init,LMBINAVE2.compressed(),LLumFunc2.compressed()-LLumFuncErr2.compressed(),weights=1/LLumFuncErr2.compressed())

#Scipy Modelling

scipy_LCBG_80100_fit,scipy_LCBG_80100_cov=sp.optimize.curve_fit(schechter_func_scipy_lcbg,LMBINAVE2.compressed(),LLumFunc2.compressed(),p0=init_vals,sigma=LLumFuncErr2.compressed())
scipy_LUMFUNC_80100_fit,scipy_LUMFUNC_80100_cov=sp.optimize.curve_fit(schechter_func_scipy_gal,MBINAVE2.compressed(),LumFunc2.compressed(),p0=init_vals,sigma=LumFuncErr2.compressed())

#Defining Errors on fit parameters from scipy

LCBG80100ERRORS=np.array([np.sqrt(scipy_LCBG_80100_cov[0][0]),np.sqrt(scipy_LCBG_80100_cov[1][1])])
LUMFUNC80100ERRORS=np.array([np.sqrt(scipy_LUMFUNC_80100_cov[0][0]),np.sqrt(scipy_LUMFUNC_80100_cov[1][1])])



LCBGcode=axes[2][1].errorbar(LMBINAVE,np.log10(LLumFunc),yerr=LLogErr,fmt=',',label='1/V$_{MAX}$ code',color='blue')
code=axes[2][1].errorbar(MBINAVE,np.log10(LumFunc),yerr=LogErr,fmt=',',label='1/V$_{MAX}$ code',color='green')
axes[2][1].errorbar(MBINAVE[LumFunc2.mask],np.log10(LumFunc[LumFunc2.mask]),yerr=LogErr[LumFunc2.mask],fmt='x',color='green')
axes[2][1].errorbar(LMBINAVE[LLumFunc2.mask],np.log10(LLumFunc[LLumFunc2.mask]),yerr=LLogErr[LLumFunc2.mask],fmt='x',label='1/V$_{MAX}$ code',color='blue')
gals=axes[3][1].bar(MBinMid,np.log10(NGal),Mbinsize,align='center',label='Number of Galaxies per Absolute Magnitude Bin',color='white')
lcbg=axes[3][1].bar(LMBinMid,np.log10(LNGal),lMbinsize,align='center',label='Number of LCBGs per Absolute Magnitude Bin',color='gray')
axes[2][1].plot(LCBG_Range,np.log10(LCBG_FIT80100(LCBG_Range)))
axes[2][1].plot(LCBG_Range,np.log10(LUMFUNC_FIT80100(LCBG_Range)))
axes[2][1].plot(LCBG_Range,np.log10(LCBG_FIT80100u(LCBG_Range)),color='red')
axes[2][1].plot(LCBG_Range,np.log10(LCBG_FIT80100l(LCBG_Range)),color='yellow')
plt.subplots_adjust(hspace=0,wspace=0)
axes[2][1].set_yticks(y)
axes[2][1].set_yticks(yminor,minor=True)
axes[2][1].set_ylim([-7.5,-0.5])
axes[3][1].set_yticks([3,2,1,0])
axes[3][1].set_yticks([3.5,2.5,1.5,0.5],minor=True)
axes[3][1].set_ylim([0,4])
autolabel(lcbg,'black',3,1)
autolabel(gals,'black',3,1)

#0.3<z<0.8

filein='/home/lrhunt/LUM_FUNC/LCBGLFOUT/LCBG_LF_30_80.txt'
sfilein='/home/lrhunt/LUM_FUNC/FULLLFOUT/LF_30_80_FULL.txt'
LMBinMid,LMBINAVE,LLumFunc,LLumFuncErr,LLogErr,LNGal,LAveCMV,LAveWeight=np.loadtxt(filein,unpack=True,skiprows=1)
MBinMid,MBINAVE,LumFunc,LumFuncErr,LogErr,NGal,AveCMV,AveWeight=np.loadtxt(sfilein,unpack=True,skiprows=1)
with open(filein,'r') as lf:
    lspecvals=lf.readline().strip().split()
with open(sfilein,'r') as lf:
	specvals=lf.readline().strip().split()
lzmax=float(lspecvals[1])
lzmin=float(lspecvals[2])
lmbin=float(lspecvals[3])
lMbinsize=float(lspecvals[4])
zmax=float(specvals[1])
zmin=float(specvals[2])
mbin=float(specvals[3])
Mbinsize=float(specvals[4])
fit=LevMarLSQFitter()

LFGAL3080=sum(NGal)
LCBGGAL3080=sum(LNGal)

LLumFunc2=np.ma.array(LLumFunc,mask=False)
LMBINAVE2=np.ma.array(LMBINAVE,mask=False)
LLumFuncErr2=np.ma.array(LLumFuncErr,mask=False)
LumFunc2=np.ma.array(LumFunc,mask=False)
MBINAVE2=np.ma.array(MBINAVE,mask=False)
LumFuncErr2=np.ma.array(LumFuncErr,mask=False)

LLumFunc2.mask[np.where(LLumFunc2==0)[0]]=True
LMBINAVE2.mask[np.where(LLumFunc2==0)[0]]=True
LLumFuncErr2.mask[np.where(LLumFunc2==0)[0]]=True
LumFunc2.mask[np.where(LumFunc2==0)[0]]=True
MBINAVE2.mask[np.where(LumFunc2==0)[0]]=True
LumFuncErr2.mask[np.where(LumFunc2==0)[0]]=True

LumFunc2.mask[11:17]=True
MBINAVE2.mask[11:17]=True
LumFuncErr2.mask[11:17]=True
LLumFunc2.mask[11:17]=True
LMBINAVE2.mask[11:17]=True
LLumFuncErr2.mask[11:17]=True
LLumFunc2.mask[0:3]=True
LMBINAVE2.mask[0:3]=True
LLumFuncErr2.mask[0:3]=True

LCBG_FIT3080=fit(LCBGFIT_init,LMBINAVE2.compressed(),LLumFunc2.compressed(),weights=1/LLumFuncErr2.compressed())
LUMFUNC_FIT3080=fit(GALFIT_init,MBINAVE2.compressed(),LumFunc2.compressed(),weights=1/LumFuncErr2.compressed())
LCBG_FIT3080u=fit(LCBGFIT_init,LMBINAVE2.compressed(),LLumFunc2.compressed()+LLumFuncErr2.compressed(),weights=1/LLumFuncErr2.compressed())
LCBG_FIT3080l=fit(LCBGFIT_init,LMBINAVE2.compressed(),LLumFunc2.compressed()-LLumFuncErr2.compressed(),weights=1/LLumFuncErr2.compressed())

#Scipy Modelling

scipy_LCBG_3080_fit,scipy_LCBG_3080_cov=sp.optimize.curve_fit(schechter_func_scipy_lcbg,LMBINAVE2.compressed(),LLumFunc2.compressed(),p0=init_vals,sigma=LLumFuncErr2.compressed())
scipy_LUMFUNC_3080_fit,scipy_LUMFUNC_3080_cov=sp.optimize.curve_fit(schechter_func_scipy_gal,MBINAVE2.compressed(),LumFunc2.compressed(),p0=init_vals,sigma=LumFuncErr2.compressed())

#Defining Errors on fit parameters from scipy

LCBG3080ERRORS=np.array([np.sqrt(scipy_LCBG_3080_cov[0][0]),np.sqrt(scipy_LCBG_3080_cov[1][1])])
LUMFUNC3080ERRORS=np.array([np.sqrt(scipy_LUMFUNC_3080_cov[0][0]),np.sqrt(scipy_LUMFUNC_3080_cov[1][1])])


LCBGcode=axes[2][2].errorbar(LMBINAVE,np.log10(LLumFunc),yerr=LLogErr,fmt=',',label='1/V$_{MAX}$ code',color='blue')
code=axes[2][2].errorbar(MBINAVE,np.log10(LumFunc),yerr=LogErr,fmt=',',label='1/V$_{MAX}$ code',color='green')
axes[2][2].errorbar(MBINAVE[LumFunc2.mask],np.log10(LumFunc[LumFunc2.mask]),yerr=LogErr[LumFunc2.mask],fmt='x',color='green')
axes[2][2].errorbar(LMBINAVE[LLumFunc2.mask],np.log10(LLumFunc[LLumFunc2.mask]),yerr=LLogErr[LLumFunc2.mask],fmt='x',label='1/V$_{MAX}$ code',color='blue')
gals=axes[3][2].bar(MBinMid,np.log10(NGal),Mbinsize,align='center',label='Number of Galaxies per Absolute Magnitude Bin',color='white')
lcbg=axes[3][2].bar(LMBinMid,np.log10(LNGal),lMbinsize,align='center',label='Number of LCBGs per Absolute Magnitude Bin',color='gray')
axes[2][2].plot(LCBG_Range,np.log10(LCBG_FIT3080(LCBG_Range)))
axes[2][2].plot(LCBG_Range,np.log10(LUMFUNC_FIT3080(LCBG_Range)))
axes[2][2].plot(LCBG_Range,np.log10(LCBG_FIT3080u(LCBG_Range)),color='red')
axes[2][2].plot(LCBG_Range,np.log10(LCBG_FIT3080l(LCBG_Range)),color='yellow')
plt.subplots_adjust(hspace=0,wspace=0)
axes[2][2].set_yticks(y)
axes[2][2].set_yticks(yminor,minor=True)
axes[2][2].set_ylim([-7.5,-0.5])
axes[3][2].set_yticks([3,2,1,0])
axes[3][2].set_yticks([3.5,2.5,1.5,0.5],minor=True)
axes[3][2].set_ylim([0,4.0])
autolabel(lcbg,'black',3,2)
autolabel(gals,'black',3,2)
axes[0][1].set_yticklabels([])
axes[0][2].set_yticklabels([])
axes[1][1].set_yticklabels([])
axes[1][2].set_yticklabels([])
axes[2][1].set_yticklabels([])
axes[2][2].set_yticklabels([])
axes[3][1].set_yticklabels([])
axes[3][2].set_yticklabels([])
axes[0][0].text(-23.5,-1,'z=0.01-0.2',verticalalignment='center',fontsize=12)
axes[0][1].text(-23.5,-1,'z=0.2-0.4',verticalalignment='center',fontsize=12)
axes[0][2].text(-23.5,-1,'z=0.4-0.6',verticalalignment='center',fontsize=12)
axes[2][0].text(-23.5,-1,'z=0.6-0.8',verticalalignment='center',fontsize=12)
axes[2][2].text(-23.5,-1,'z=0.3-0.8',verticalalignment='center',fontsize=12)
axes[2][1].text(-23.5,-1,'z=0.8-1.0',verticalalignment='center',fontsize=12)
f.text(0.52,0.05,'M$_{B}$-5log$_{10}$(h$_{70}$)',ha='center',va='center',fontsize=12)
f.text(0.05,0.75,'Log$_{10}$($\Phi_{M}$) ($h^{3}_{70}$ Mpc$^{-3}$ mag$^{-1}$)',ha='center',va='center',rotation='vertical',fontsize=12)
f.text(0.05,0.35,'Log$_{10}$($\Phi_{M}$) ($h^{3}_{70}$ Mpc$^{-3}$ mag$^{-1}$)',ha='center',va='center',rotation='vertical',fontsize=12)
f.text(0.05,0.55,'Log$_{10}$(N)',ha='center',va='center',rotation='vertical',fontsize=12)
f.text(0.05,0.15,'Log$_{10}$(N)',ha='center',va='center',rotation='vertical',fontsize=12)




#*******************************
#*******************************
#*******************************
#GENERATING OUTPUT 
#*******************************
#*******************************
#*******************************



redshiftrange=np.array([0.1,0.3,0.5,0.7,0.9,0.55])

allngal=np.array([LFGAL020,LFGAL2040,LFGAL4060,LFGAL6080,LFGAL80100,LFGAL3080])

allphistar=1000*np.array([LUMFUNC_FIT020.phistar.value,LUMFUNC_FIT2040.phistar.value,LUMFUNC_FIT4060.phistar.value,LUMFUNC_FIT6080.phistar.value,LUMFUNC_FIT80100.phistar.value,LUMFUNC_FIT3080.phistar.value])

allphistarerr=1000*np.array([LUMFUNC020ERRORS[0],LUMFUNC2040ERRORS[0],LUMFUNC4060ERRORS[0],LUMFUNC6080ERRORS[0],LUMFUNC80100ERRORS[0],LUMFUNC3080ERRORS[0]])

allmstar=np.array([LUMFUNC_FIT020.mstar.value,LUMFUNC_FIT2040.mstar.value,LUMFUNC_FIT4060.mstar.value,LUMFUNC_FIT6080.mstar.value,LUMFUNC_FIT80100.mstar.value,LUMFUNC_FIT3080.mstar.value])

allmstarerr=np.array([LUMFUNC020ERRORS[1],LUMFUNC2040ERRORS[1],LUMFUNC4060ERRORS[1],LUMFUNC6080ERRORS[1],LUMFUNC80100ERRORS[1],LUMFUNC3080ERRORS[1]])

#allalpha=np.array([LUMFUNC_FIT020.alpha.value,LUMFUNC_FIT2040.alpha.value,LUMFUNC_FIT4060.alpha.value,LUMFUNC_FIT6080.alpha.value,LUMFUNC_FIT80100.alpha.value,LUMFUNC_FIT3080.alpha.value])

#allalphaerr=np.array([LUMFUNC020ERRORS[2],LUMFUNC2040ERRORS[2],LUMFUNC4060ERRORS[2],LUMFUNC6080ERRORS[2],LUMFUNC80100ERRORS[2],LUMFUNC3080ERRORS[2]])

lcbgngal=np.array([LCBGGAL020,LCBGGAL2040,LCBGGAL4060,LCBGGAL6080,LCBGGAL80100,LCBGGAL3080])

lcbgphistar=1000*np.array([LCBG_FIT020.phistar.value,LCBG_FIT2040.phistar.value,LCBG_FIT4060.phistar.value,LCBG_FIT6080.phistar.value,LCBG_FIT80100.phistar.value,LCBG_FIT3080.phistar.value])

lcbgphistarerr=1000*np.array([LCBG020ERRORS[0],LCBG2040ERRORS[0],LCBG4060ERRORS[0],LCBG6080ERRORS[0],LCBG80100ERRORS[0],LCBG3080ERRORS[0]])

lcbgmstar=np.array([LCBG_FIT020.mstar.value,LCBG_FIT2040.mstar.value,LCBG_FIT4060.mstar.value,LCBG_FIT6080.mstar.value,LCBG_FIT80100.mstar.value,LCBG_FIT3080.mstar.value])

lcbgmstarerr=np.array([LCBG020ERRORS[1],LCBG2040ERRORS[1],LCBG4060ERRORS[1],LCBG6080ERRORS[1],LCBG80100ERRORS[1],LCBG3080ERRORS[1]])

#lcbgalpha=np.array([LCBG_FIT020.alpha.value,LCBG_FIT2040.alpha.value,LCBG_FIT4060.alpha.value,LCBG_FIT6080.alpha.value,LCBG_FIT80100.alpha.value,LCBG_FIT3080.alpha.value])

#lcbgalphaerr=np.array([LCBG020ERRORS[2],LCBG2040ERRORS[2],LCBG4060ERRORS[2],LCBG6080ERRORS[2],LCBG80100ERRORS[2],LCBG3080ERRORS[2]])

galdensityeighteen=np.array([sp.integrate.quad(LUMFUNC_FIT020,-100,-18.5)[0],sp.integrate.quad(LUMFUNC_FIT2040,-100,-18.5)[0],sp.integrate.quad(LUMFUNC_FIT4060,-100,-18.5)[0],sp.integrate.quad(LUMFUNC_FIT6080,-100,-18.5)[0],sp.integrate.quad(LUMFUNC_FIT80100,-100,-18.5)[0],sp.integrate.quad(LUMFUNC_FIT3080,-100,-18.5)[0]])

galdensityfifteen=np.array([sp.integrate.quad(LUMFUNC_FIT020,-100,-15)[0],sp.integrate.quad(LUMFUNC_FIT2040,-100,-15)[0],sp.integrate.quad(LUMFUNC_FIT4060,-100,-15)[0],sp.integrate.quad(LUMFUNC_FIT6080,-100,-15)[0],sp.integrate.quad(LUMFUNC_FIT80100,-100,-15)[0],sp.integrate.quad(LUMFUNC_FIT3080,-100,-15)[0]])

lcbgdensity=np.array([sp.integrate.quad(LCBG_FIT020,-100,-18.5)[0],sp.integrate.quad(LCBG_FIT2040,-100,-18.5)[0],sp.integrate.quad(LCBG_FIT4060,-100,-18.5)[0],sp.integrate.quad(LCBG_FIT6080,-100,-18.5)[0],sp.integrate.quad(LCBG_FIT80100,-100,-18.5)[0],sp.integrate.quad(LCBG_FIT3080,-100,-18.5)[0]])

FracGals=np.stack((redshiftrange,galdensityeighteen,galdensityfifteen,lcbgdensity),axis=-1)

#np.savetxt('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/TXTFILES/Fracfromlumfunc.txt',FracGals,header='#z	galden	lcbgden')

LFfittingparams=np.stack((redshiftrange,allngal,allmstar,allmstarerr,allphistar,allphistarerr),axis=-1)
LCBGfittingparams=np.stack((redshiftrange,lcbgngal,lcbgmstar,lcbgmstarerr,lcbgphistar,lcbgphistarerr),axis=-1)
#np.savetxt('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/TXTFILES/LFfitparams.txt',LFfittingparams,header='#zupper	alpha	alphaerr	mstar	mstarerr	phistar	phistarerr',fmt='%5.3f')
#np.savetxt('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/TXTFILES/LCBGfitparams.txt',LCBGfittingparams,header='#zupper	alpha	alphaerr	mstar	mstarerr	phistar	phistarerr',fmt='%5.3f')

WILLMER=np.loadtxt('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/TXTFILES/OTHERSTUDIES/WILLMER.txt')
FABER=np.loadtxt('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/TXTFILES/OTHERSTUDIES/FABER.txt')
COOL=np.loadtxt('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/TXTFILES/OTHERSTUDIES/COOL.txt')
ZUCCA=np.loadtxt('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/TXTFILES/OTHERSTUDIES/ZUCCA.txt')
FRITZ=np.loadtxt('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/TXTFILES/OTHERSTUDIES/FRITZ.txt')
BEARE=np.loadtxt('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/TXTFILES/OTHERSTUDIES/BEARE.txt')


LCBGDENUP020ps=sp.integrate.quad(schechter_func(alpha=-0.61,phistar=0.00098,mstar=-20.33),-100,-18.5)
LCBGDENLOW020ps=sp.integrate.quad(schechter_func(alpha=-0.61,phistar=0.00044,mstar=-20.33),-100,-18.5)
LCBGDENUP2040ps=sp.integrate.quad(schechter_func(alpha=-1.06,phistar=0.00196,mstar=-20.08),-100,-18.5)
LCBGDENLOW2040ps=sp.integrate.quad(schechter_func(alpha=-1.06,phistar=0.00106,mstar=-20.08),-100,-18.5)
LCBGDENUP4060ps=sp.integrate.quad(schechter_func(alpha=-0.86,phistar=0.0018,mstar=-20.55),-100,-18.5)
LCBGDENLOW4060ps=sp.integrate.quad(schechter_func(alpha=-0.86,phistar=0.00126,mstar=-20.55),-100,-18.5)
LCBGDENUP6080ps=sp.integrate.quad(schechter_func(alpha=-0.88,phistar=0.00405,mstar=-20.56),-100,-18.5)
LCBGDENLOW6080ps=sp.integrate.quad(schechter_func(alpha=-0.88,phistar=0.00243,mstar=-20.56),-100,-18.5)
LCBGDENUP80100ps=sp.integrate.quad(schechter_func(alpha=-1.12,phistar=0.00479,mstar=-20.86),-100,-18.5)
LCBGDENLOW80100ps=sp.integrate.quad(schechter_func(alpha=-1.12,phistar=0.00381,mstar=-20.86),-100,-18.5)
LCBGDENUP3080ps=sp.integrate.quad(schechter_func(alpha=-0.52,phistar=0.00296,mstar=-20.37),-100,-18.5)
LCBGDENLOW3080ps=sp.integrate.quad(schechter_func(alpha=-0.52,phistar=0.0024,mstar=-20.37),-100,-18.5)

LCBGDENUP020ms=sp.integrate.quad(schechter_func(alpha=-0.61,phistar=0.00071,mstar=-19.92),-100,-18.5)
LCBGDENLOW020ms=sp.integrate.quad(schechter_func(alpha=-0.61,phistar=0.00071,mstar=-20.74),-100,-18.5)
LCBGDENUP2040ms=sp.integrate.quad(schechter_func(alpha=-1.06,phistar=0.00151,mstar=-19.83),-100,-18.5)
LCBGDENLOW2040ms=sp.integrate.quad(schechter_func(alpha=-1.06,phistar=0.00151,mstar=-20.33),-100,-18.5)
LCBGDENUP4060ms=sp.integrate.quad(schechter_func(alpha=-0.86,phistar=0.00153,mstar=-20.38),-100,-18.5)
LCBGDENLOW4060ms=sp.integrate.quad(schechter_func(alpha=-0.86,phistar=0.00153,mstar=-20.72),-100,-18.5)
LCBGDENUP6080ms=sp.integrate.quad(schechter_func(alpha=-0.88,phistar=0.00324,mstar=-20.3),-100,-18.5)
LCBGDENLOW6080ms=sp.integrate.quad(schechter_func(alpha=-0.88,phistar=0.00324,mstar=-20.82),-100,-18.5)
LCBGDENUP80100ms=sp.integrate.quad(schechter_func(alpha=-1.12,phistar=0.0043,mstar=-20.98),-100,-18.5)
LCBGDENLOW80100ms=sp.integrate.quad(schechter_func(alpha=-1.12,phistar=0.0043,mstar=-20.74),-100,-18.5)
LCBGDENUP3080ms=sp.integrate.quad(schechter_func(alpha=-0.52,phistar=0.00268,mstar=-20.24),-100,-18.5)
LCBGDENLOW3080ms=sp.integrate.quad(schechter_func(alpha=-0.52,phistar=0.00268,mstar=-20.5),-100,-18.5)

#LCBGDENUP020al=sp.integrate.quad(schechter_func(alpha=-0.98,phistar=0.00071,mstar=-20.33),-100,-18.5)
#LCBGDENLOW020al=sp.integrate.quad(schechter_func(alpha=-0.24,phistar=0.00071,mstar=-20.33),-100,-18.5)
#LCBGDENUP2040al=sp.integrate.quad(schechter_func(alpha=-1.3,phistar=0.00151,mstar=-20.08),-100,-18.5)
#LCBGDENLOW2040al=sp.integrate.quad(schechter_func(alpha=-0.82,phistar=0.00151,mstar=-20.08),-100,-18.5)
#LCBGDENUP4060al=sp.integrate.quad(schechter_func(alpha=-1,phistar=0.00153,mstar=-20.55),-100,-18.5)
#LCBGDENLOW4060al=sp.integrate.quad(schechter_func(alpha=-0.72,phistar=0.00153,mstar=-20.55),-100,-18.5)
#LCBGDENUP6080al=sp.integrate.quad(schechter_func(alpha=-1.24,phistar=0.00324,mstar=-20.56),-100,-18.5)
#LCBGDENLOW6080al=sp.integrate.quad(schechter_func(alpha=-0.52,phistar=0.00324,mstar=-20.56),-100,-18.5)
#LCBGDENUP80100al=sp.integrate.quad(schechter_func(alpha=-1.33,phistar=0.0043,mstar=-20.86),-100,-18.5)
#LCBGDENLOW80100al=sp.integrate.quad(schechter_func(alpha=-0.91,phistar=0.0043,mstar=-20.86),-100,-18.5)
#LCBGDENUP3080al=sp.integrate.quad(schechter_func(alpha=-0.67,phistar=0.00268,mstar=-20.37),-100,-18.5)
#LCBGDENLOW3080al=sp.integrate.quad(schechter_func(alpha=-0.37,phistar=0.00268,mstar=-20.37),-100,-18.5)

LCBGDENUPps=np.stack((LCBGDENUP020ps[0],LCBGDENUP2040ps[0],LCBGDENUP4060ps[0],LCBGDENUP6080ps[0],LCBGDENUP80100ps[0],LCBGDENUP3080ps[0]),axis=-1)

LCBGDENLOWms=np.stack((LCBGDENUP020ms[0],LCBGDENUP2040ms[0],LCBGDENUP4060ms[0],LCBGDENUP6080ms[0],LCBGDENUP80100ms[0],LCBGDENUP3080ms[0]),axis=-1)

#LCBGDENUPal=np.stack((LCBGDENUP020al[0],LCBGDENUP2040al[0],LCBGDENUP4060al[0],LCBGDENUP6080al[0],LCBGDENUP80100al[0],LCBGDENUP3080al[0]),axis=-1)

LCBGDENLOWps=np.stack((LCBGDENLOW020ps[0],LCBGDENLOW2040ps[0],LCBGDENLOW4060ps[0],LCBGDENLOW6080ps[0],LCBGDENLOW80100ps[0],LCBGDENLOW3080ps[0]),axis=-1)

LCBGDENUPms=np.stack((LCBGDENLOW020ms[0],LCBGDENLOW2040ms[0],LCBGDENLOW4060ms[0],LCBGDENLOW6080ms[0],LCBGDENLOW80100ms[0],LCBGDENLOW3080ms[0]),axis=-1)

#LCBGDENLOWal=np.stack((LCBGDENLOW020al[0],LCBGDENLOW2040al[0],LCBGDENLOW4060al[0],LCBGDENLOW6080al[0],LCBGDENLOW80100al[0],LCBGDENLOW3080al[0]),axis=-1)

#*******************************
#*******************************
#*******************************
# Plotting Evolution of Parameters
#*******************************
#*******************************
#*******************************


f,axes=plt.subplots(nrows=2,ncols=1,sharex=True,sharey=True,gridspec_kw={'height_ratios':[1,1]},figsize=(8,10))
axes[0].errorbar(LFfittingparams[0:5,0],LFfittingparams[0:5,2],yerr=LFfittingparams[0:5,3],fmt='-0',color='black',label='This Work')
axes[0].errorbar(WILLMER[:,0],WILLMER[:,4],yerr=WILLMER[:,5],color='yellow',fmt='->',label='Willmer, 2006')
axes[0].errorbar(FABER[:,0],FABER[:,4],yerr=FABER[:,5],color='green',fmt='-<',label='Faber, 2007')
axes[0].errorbar(ZUCCA[:,0],ZUCCA[:,4],yerr=ZUCCA[:,5],color='grey',fmt='-8',label='Zucca, 2009')
axes[0].errorbar(COOL[:,0],COOL[:,4],yerr=COOL[:,5],color='red',fmt='-s',label='Cool, 2012')
axes[0].errorbar(FRITZ[:,0],FRITZ[:,4],yerr=FRITZ[:,5],color='purple',fmt='-*',label='Fritz, 2014')
axes[0].errorbar(BEARE[:,0],BEARE[:,4],yerr=BEARE[:,5],color='slateblue',fmt='-d',label='Beare, 2015')
axes[1].errorbar(LCBGfittingparams[0:5,0],LCBGfittingparams[0:5,2],yerr=LCBGfittingparams[0:5,3],color='blue')
axes[1].set_xlim([0,1.5])
axes[1].set_xticks([0.2,0.4,0.6,0.8,1.0])
axes[1].set_ylim([-19.6,-21.8])
axes[1].set_yticks([-20,-20.5,-21,-21.5])
plt.subplots_adjust(hspace=0,left=0.17)
axes[0].legend(fontsize='small')
axes[0].grid()
axes[1].grid()
axes[0].text(0.5,-21.6,'All',fontsize=12,ha='center',va='center')
axes[1].text(0.5,-21.6,'LCBG',fontsize=12,ha='center',va='center')
f.text(0.05,0.5,'M$^{*}$-5log(h$_{70}$)',ha='center',va='center',rotation='vertical',fontsize=14)
f.text(0.55,0.05,'z',ha='center',va='center',fontsize=14)
#plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/EVMSTAR.pdf')

f,axes=plt.subplots(nrows=2,ncols=1,sharex=True,sharey=True,gridspec_kw={'height_ratios':[1,1]},figsize=(8,10))
axes[0].errorbar(LFfittingparams[0:5,0],LFfittingparams[0:5,4]/1000.,yerr=LFfittingparams[0:5,5]/1000.,color='black',label='This Work',fmt='-o')
axes[0].errorbar(WILLMER[:,0],WILLMER[:,6]/1000.,yerr=WILLMER[:,7]/1000.,color='yellow',fmt='->',label='Willmer, 2006')
axes[0].errorbar(FABER[:,0],FABER[:,6]/1000.,yerr=FABER[:,7]/1000.,color='green',fmt='-<',label='Faber, 2007')
axes[0].errorbar(ZUCCA[:,0],ZUCCA[:,6]/1000.,yerr=ZUCCA[:,7]/1000.,color='grey',fmt='-8',label='Zucca, 2009')
axes[0].errorbar(COOL[:,0],COOL[:,6]/1000.,yerr=COOL[:,7]/1000.,color='red',fmt='-s',label='Cool, 2012')
axes[0].errorbar(FRITZ[:,0],FRITZ[:,6]/1000.,yerr=FRITZ[:,7]/1000.,color='purple',fmt='-*',label='Fritz, 2014')
axes[0].errorbar(BEARE[:,0],BEARE[:,6]/1000.,yerr=BEARE[:,7]/1000.,color='slateblue',fmt='-d',label='Beare, 2015')
axes[1].errorbar(LCBGfittingparams[0:5,0],LCBGfittingparams[0:5,4]/1000.,yerr=LCBGfittingparams[0:5,5]/1000.,color='blue')
axes[1].set_xlim([0,1.5])
axes[1].set_xticks([0.2,0.4,0.6,0.8,1.0])
axes[1].set_ylim([0,0.01])
axes[1].set_yticks([0.001,0.003,0.005,0.007,0.009])
plt.subplots_adjust(hspace=0,left=0.17)
axes[0].legend(fontsize='small')
axes[0].grid()
axes[1].grid()
axes[0].text(0.5,0.009,'All',fontsize=12,ha='center',va='center')
axes[1].text(0.5,0.009,'LCBG',fontsize=12,ha='center',va='center')
f.text(0.05,0.5,'$\Phi^{*}$ ($h_{70}^{3}Mpc^{-3} mag^{-1}$)',ha='center',va='center',rotation='vertical',fontsize=14)
f.text(0.55,0.05,'z',ha='center',va='center',fontsize=14)
#plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/EVPHISTAR.pdf')

lcbgmstaru=unp.uarray([lcbgmstar,lcbgmstarerr])
lcbgphistaru=unp.uarray([lcbgphistar,lcbgphistarerr])
allmstaru=unp.uarray([allmstar,allmstarerr])
allphistaru=unp.uarray([allphistar,allphistarerr])
BEAREmstaru=unp.uarray([BEARE[:,4],BEARE[:,5]])
BEAREphistaru=unp.uarray([BEARE[:,6],BEARE[:,7]])
WILLMERmstaru=unp.uarray([WILLMER[:,4],WILLMER[:,5]])
WILLMERphistaru=unp.uarray([WILLMER[:,6],WILLMER[:,7]])
FABERmstaru=unp.uarray([FABER[:,4],FABER[:,5]])
FABERphistaru=unp.uarray([FABER[:,6],FABER[:,7]])
COOLmstaru=unp.uarray([COOL[:,4],COOL[:,5]])
COOLphistaru=unp.uarray([COOL[:,6],COOL[:,7]])
ZUCCAmstaru=unp.uarray([ZUCCA[:,4],ZUCCA[:,5]])
ZUCCAphistaru=unp.uarray([ZUCCA[:,6],ZUCCA[:,7]])
FRITZmstaru=unp.uarray([FRITZ[:,4],FRITZ[:,5]])
FRITZphistaru=unp.uarray([FRITZ[:,6],FRITZ[:,7]])

lcbgju=lcbgphistaru/1000*np.power(10,(lcbgmstaru-5.48)/-2.5)*sp.special.gammainc(-0.5+2,np.power(10,(lcbgmstar+18.5)/2.5))
galju=allphistaru/1000*np.power(10,(allmstaru-5.48)/-2.5)*sp.special.gammainc(2+-1.053,np.power(10,(allmstar+18.5)/2.5))
BEAREju=BEAREphistaru/1000*np.power(10,(BEAREmstaru-5.48)/-2.5)*sp.special.gammainc(2+BEARE[:,2],np.power(10,(BEARE[:,4]+18.5)/2.5))
WILLMERju=WILLMERphistaru/1000*np.power(10,(WILLMERmstaru-5.48)/-2.5)*sp.special.gammainc(2+WILLMER[:,2],np.power(10,(WILLMER[:,4]+18.5)/2.5))
FABERju=FABERphistaru/1000*np.power(10,(FABERmstaru-5.48)/-2.5)*sp.special.gammainc(2+FABER[:,2],np.power(10,(FABER[:,4]+18.5)/2.5))
COOLju=COOLphistaru/1000*np.power(10,(COOLmstaru-5.48)/-2.5)*sp.special.gammainc(2+COOL[:,2],np.power(10,(COOL[:,4]+18.5)/2.5))
ZUCCAju=ZUCCAphistaru/1000*np.power(10,(ZUCCAmstaru-5.48)/-2.5)*sp.special.gammainc(2+ZUCCA[:,2],np.power(10,(ZUCCA[:,4]+18.5)/2.5))
FRITZju=FRITZphistaru/1000*np.power(10,(FRITZmstaru-5.48)/-2.5)*sp.special.gammainc(2+FRITZ[:,2],np.power(10,(FRITZ[:,4]+18.5)/2.5))

f,axes=plt.subplots(nrows=2,ncols=1,sharex=True,sharey=True,gridspec_kw={'height_ratios':[1,1]},figsize=(8,10))
axes[0].errorbar(LFfittingparams[0:5,0],unp.nominal_values(galju)[0:5]/10000000,yerr=unp.std_devs(galju)[0:5]/10000000,color='black',label='This Work',fmt='-0')
axes[0].errorbar(BEARE[:,0],unp.nominal_values(BEAREju)/10000000,yerr=unp.std_devs(BEAREju)/10000000,color='slateblue',label='Beare, 2015')
axes[0].errorbar(WILLMER[:,0],unp.nominal_values(WILLMERju)/10000000,yerr=unp.std_devs(WILLMERju)/10000000,color='yellow',label='Willmer, 2006')
axes[0].errorbar(FABER[:,0],unp.nominal_values(FABERju)/10000000,yerr=unp.std_devs(FABERju)/10000000,color='green', label='Faber, 2007')
axes[0].errorbar(COOL[:,0],unp.nominal_values(COOLju)/10000000,yerr=unp.std_devs(COOLju)/10000000,color='red',label='Cool, 2012')
axes[0].errorbar(ZUCCA[:,0],unp.nominal_values(ZUCCAju)/10000000,yerr=unp.std_devs(ZUCCAju)/10000000,color='grey',label='Zucca, 2009')
axes[0].errorbar(FRITZ[:,0],unp.nominal_values(FRITZju)/10000000,yerr=unp.std_devs(FRITZju)/10000000,color='purple', label='Fritz, 2014')
axes[1].errorbar(LFfittingparams[0:5,0],unp.nominal_values(lcbgju)[0:5]/10000000,yerr=unp.std_devs(lcbgju)[0:5]/10000000,color='blue')
axes[1].set_xlim([0,1.5])
axes[1].set_ylim([0,10])
axes[1].set_yticks([1,2,3,4,5,6,7,8,9])
axes[1].set_xticks([0.2,0.4,0.6,0.8,1.0])
plt.subplots_adjust(hspace=0,left=0.17)
axes[0].grid()
axes[1].grid()
axes[0].legend(fontsize='small')
axes[0].text(0.5,4,'All',fontsize=12,ha='center',va='center')
axes[1].text(0.5,4,'LCBG',fontsize=12,ha='center',va='center')
f.text(0.05,0.5,'$j_{B}$ (10$^{7}$h$_{70}$L$_{\odot}$Mpc$^{-3}$)',ha='center',va='center',rotation='vertical',fontsize=14)
f.text(0.55,0.05,'z',ha='center',va='center',fontsize=14)
#plt.savefig('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/PLOTS/LUMDENSEV.pdf')

galju2=allphistaru/1000*np.power(10,(allmstaru-5.48)/-2.5)*sp.special.gamma(2+allalpha)
BEAREju2=BEAREphistaru/1000*np.power(10,(BEAREmstaru-5.48)/-2.5)*sp.special.gamma(2+BEARE[:,2])
WILLMERju2=WILLMERphistaru/1000*np.power(10,(WILLMERmstaru-5.48)/-2.5)*sp.special.gamma(2+WILLMER[:,2])
FABERju2=FABERphistaru/1000*np.power(10,(FABERmstaru-5.48)/-2.5)*sp.special.gamma(2+FABER[:,2])
COOLju2=COOLphistaru/1000*np.power(10,(COOLmstaru-5.48)/-2.5)*sp.special.gamma(2+COOL[:,2])
ZUCCAju2=ZUCCAphistaru/1000*np.power(10,(ZUCCAmstaru-5.48)/-2.5)*sp.special.gamma(2+ZUCCA[:,2])
FRITZju2=FRITZphistaru/1000*np.power(10,(FRITZmstaru-5.48)/-2.5)*sp.special.gamma(2+FRITZ[:,2])

f,axes=plt.subplots(nrows=1,ncols=1,sharex=True,sharey=True)
axes.errorbar(LFfittingparams[0:5,0],unp.nominal_values(galju2)[0:5]/100000000,yerr=unp.std_devs(galju2)[0:5]/100000000,color='black')
axes.errorbar(BEARE[:,0],unp.nominal_values(BEAREju2)/100000000,yerr=unp.std_devs(BEAREju2)/100000000,color='yellow')
axes.errorbar(WILLMER[:,0],unp.nominal_values(WILLMERju2)/100000000,yerr=unp.std_devs(WILLMERju2)/100000000,color='green')
axes.errorbar(FABER[:,0],unp.nominal_values(FABERju2)/100000000,yerr=unp.std_devs(FABERju2)/100000000,color='gray')
axes.errorbar(COOL[:,0],unp.nominal_values(COOLju2)/100000000,yerr=unp.std_devs(COOLju2)/100000000,color='red')
axes.errorbar(ZUCCA[:,0],unp.nominal_values(ZUCCAju2)/100000000,yerr=unp.std_devs(ZUCCAju2)/100000000,color='purple')
axes.errorbar(FRITZ[:,0],unp.nominal_values(FRITZju2)/100000000,yerr=unp.std_devs(FRITZju2)/100000000,color='slateblue')
axes.errorbar(LFfittingparams[0:5,0],unp.nominal_values(galju)[0:5]/10000000,yerr=unp.std_devs(galju)[0:5]/10000000,color='black',ls='--')
axes.errorbar(BEARE[:,0],unp.nominal_values(BEAREju)/10000000,yerr=unp.std_devs(BEAREju)/10000000,color='yellow',ls='--')
axes.errorbar(WILLMER[:,0],unp.nominal_values(WILLMERju)/10000000,yerr=unp.std_devs(WILLMERju)/10000000,color='green',ls='--')
axes.errorbar(FABER[:,0],unp.nominal_values(FABERju)/10000000,yerr=unp.std_devs(FABERju)/10000000,color='gray',ls='--')
axes.errorbar(COOL[:,0],unp.nominal_values(COOLju)/10000000,yerr=unp.std_devs(COOLju)/10000000,color='red',ls='--')
axes.errorbar(ZUCCA[:,0],unp.nominal_values(ZUCCAju)/10000000,yerr=unp.std_devs(ZUCCAju)/10000000,color='purple',ls='--')
axes.errorbar(FRITZ[:,0],unp.nominal_values(FRITZju)/10000000,yerr=unp.std_devs(FRITZju)/10000000,color='slateblue',ls='--')



#*******************************
#*******************************
#*******************************
#CALCULATING LCBG PHI IN SAME BINS AS WERK PAPER
#*******************************
#*******************************
#*******************************

#0.4<z<0.7

filein='/home/lrhunt/LUM_FUNC/LCBGLFOUT/LCBG_LF_40_70.txt'
LMBinMid,LMBINAVE,LLumFunc,LLumFuncErr,LLogErr,LNGal,LAveCMV,LAveWeight=np.loadtxt(filein,unpack=True,skiprows=1)

with open(filein,'r') as lf:
    lspecvals=lf.readline().strip().split()

lzmax=float(lspecvals[1])
lzmin=float(lspecvals[2])
lmbin=float(lspecvals[3])
lMbinsize=float(lspecvals[4])

fit=LevMarLSQFitter()

LCBGGAL4070=sum(LNGal)

LLumFunc2=np.ma.array(LLumFunc,mask=False)
LMBINAVE2=np.ma.array(LMBINAVE,mask=False)
LLumFuncErr2=np.ma.array(LLumFuncErr,mask=False)

LLumFunc2.mask[np.where(LLumFunc2==0)[0]]=True
LMBINAVE2.mask[np.where(LLumFunc2==0)[0]]=True
LLumFuncErr2.mask[np.where(LLumFunc2==0)[0]]=True


LLumFunc2.mask[7:17]=True
LMBINAVE2.mask[7:17]=True
LLumFuncErr2.mask[7:17]=True

LCBG_FIT4070=fit(LCBGFIT_init,LMBINAVE2.compressed(),LLumFunc2.compressed(),weights=1/LLumFuncErr2.compressed())


#Scipy Modelling

scipy_LCBG_4070_fit,scipy_LCBG_4070_cov=sp.optimize.curve_fit(schechter_func_scipy,LMBINAVE2.compressed(),LLumFunc2.compressed(),p0=init_vals,sigma=LLumFuncErr2.compressed())

#Defining Errors on fit parameters from scipy

LCBG4070ERRORS=np.array([np.sqrt(scipy_LCBG_4070_cov[0][0]),np.sqrt(scipy_LCBG_4070_cov[1][1]),np.sqrt(scipy_LCBG_4070_cov[2][2])])

LCBGcode=plt.errorbar(LMBINAVE,np.log10(LLumFunc),yerr=LLogErr,fmt=',',label='1/V$_{MAX}$ code',color='blue')
plt.errorbar(LMBINAVE[LLumFunc2.mask],np.log10(LLumFunc[LLumFunc2.mask]),yerr=LLogErr[LLumFunc2.mask],fmt='x',label='1/V$_{MAX}$ code',color='blue')
plt.plot(LCBG_Range,np.log10(LCBG_FIT4070(LCBG_Range)))

#0.7<z<1.0


filein='/home/lrhunt/LUM_FUNC/LCBGLFOUT/LCBG_LF_70_100.txt'
LMBinMid,LMBINAVE,LLumFunc,LLumFuncErr,LLogErr,LNGal,LAveCMV,LAveWeight=np.loadtxt(filein,unpack=True,skiprows=1)

with open(filein,'r') as lf:
    lspecvals=lf.readline().strip().split()

lzmax=float(lspecvals[1])
lzmin=float(lspecvals[2])
lmbin=float(lspecvals[3])
lMbinsize=float(lspecvals[4])

fit=LevMarLSQFitter()

LCBGGAL70100=sum(LNGal)

LLumFunc2=np.ma.array(LLumFunc,mask=False)
LMBINAVE2=np.ma.array(LMBINAVE,mask=False)
LLumFuncErr2=np.ma.array(LLumFuncErr,mask=False)

LLumFunc2.mask[np.where(LLumFunc2==0)[0]]=True
LMBINAVE2.mask[np.where(LLumFunc2==0)[0]]=True
LLumFuncErr2.mask[np.where(LLumFunc2==0)[0]]=True


LLumFunc2.mask[6:17]=True
LMBINAVE2.mask[6:17]=True
LLumFuncErr2.mask[6:17]=True

LCBG_FIT70100=fit(LCBGFIT_init,LMBINAVE2.compressed(),LLumFunc2.compressed(),weights=1/LLumFuncErr2.compressed())

LCBGcode=plt.errorbar(LMBINAVE,np.log10(LLumFunc),yerr=LLogErr,fmt=',',label='1/V$_{MAX}$ code',color='blue')
plt.errorbar(LMBINAVE[LLumFunc2.mask],np.log10(LLumFunc[LLumFunc2.mask]),yerr=LLogErr[LLumFunc2.mask],fmt='x',label='1/V$_{MAX}$ code',color='blue')
plt.plot(LCBG_Range,np.log10(LCBG_FIT70100(LCBG_Range)))


#Scipy Modelling

scipy_LCBG_70100_fit,scipy_LCBG_70100_cov=sp.optimize.curve_fit(schechter_func_scipy,LMBINAVE2.compressed(),LLumFunc2.compressed(),p0=init_vals,sigma=LLumFuncErr2.compressed())

#Defining Errors on fit parameters from scipy

LCBG70100ERRORS=np.array([np.sqrt(scipy_LCBG_70100_cov[0][0]),np.sqrt(scipy_LCBG_70100_cov[1][1]),np.sqrt(scipy_LCBG_70100_cov[2][2])])

LCBGcode=plt.errorbar(LMBINAVE,np.log10(LLumFunc),yerr=LLogErr,fmt=',',label='1/V$_{MAX}$ code',color='blue')
plt.errorbar(LMBINAVE[LLumFunc2.mask],np.log10(LLumFunc[LLumFunc2.mask]),yerr=LLogErr[LLumFunc2.mask],fmt='x',label='1/V$_{MAX}$ code',color='blue')
plt.plot(LCBG_Range,np.log10(LCBG_FIT70100(LCBG_Range)))


GaldensityPhillips=np.array([sp.integrate.quad(LCBG_FIT4070,-100,-18.5)[0],sp.integrate.quad(LCBG_FIT70100,-100,-18.5)[0]])

LCBGDENUP4070ps=sp.integrate.quad(schechter_func(alpha=LCBG_FIT4070.alpha.value,phistar=LCBG_FIT4070.phistar.value+LCBG4070ERRORS[0],mstar=LCBG_FIT4070.mstar.value),-100,-18.5)[0]
LCBGDENDOWN4070ps=sp.integrate.quad(schechter_func(alpha=LCBG_FIT4070.alpha.value,phistar=LCBG_FIT4070.phistar.value-LCBG4070ERRORS[0],mstar=LCBG_FIT4070.mstar.value),-100,-18.5)[0]
LCBGDENUP4070ms=sp.integrate.quad(schechter_func(alpha=LCBG_FIT4070.alpha.value,phistar=LCBG_FIT4070.phistar.value,mstar=LCBG_FIT4070.mstar.value+LCBG4070ERRORS[1]),-100,-18.5)[0]
LCBGDENDOWN4070ms=sp.integrate.quad(schechter_func(alpha=LCBG_FIT4070.alpha.value,phistar=LCBG_FIT4070.phistar.value,mstar=LCBG_FIT4070.mstar.value-LCBG4070ERRORS[1]),-100,-18.5)[0]
LCBGDENUP4070al=sp.integrate.quad(schechter_func(alpha=LCBG_FIT4070.alpha.value+LCBG4070ERRORS[2],phistar=LCBG_FIT4070.phistar.value,mstar=LCBG_FIT4070.mstar.value),-100,-18.5)[0]
LCBGDENDOWN4070al=sp.integrate.quad(schechter_func(alpha=LCBG_FIT4070.alpha.value-LCBG4070ERRORS[2],phistar=LCBG_FIT4070.phistar.value,mstar=LCBG_FIT4070.mstar.value),-100,-18.5)[0]
LCBGDENUP70100ps=sp.integrate.quad(schechter_func(alpha=LCBG_FIT4070.alpha.value,phistar=LCBG_FIT4070.phistar.value+LCBG70100ERRORS[0],mstar=LCBG_FIT4070.mstar.value),-100,-18.5)[0]
LCBGDENDOWN70100ps=sp.integrate.quad(schechter_func(alpha=LCBG_FIT4070.alpha.value,phistar=LCBG_FIT4070.phistar.value-LCBG70100ERRORS[0],mstar=LCBG_FIT4070.mstar.value),-100,-18.5)[0]
LCBGDENUP70100ms=sp.integrate.quad(schechter_func(alpha=LCBG_FIT4070.alpha.value,phistar=LCBG_FIT4070.phistar.value,mstar=LCBG_FIT4070.mstar.value+LCBG70100ERRORS[1]),-100,-18.5)[0]
LCBGDENDOWN70100ms=sp.integrate.quad(schechter_func(alpha=LCBG_FIT4070.alpha.value,phistar=LCBG_FIT4070.phistar.value,mstar=LCBG_FIT4070.mstar.value-LCBG70100ERRORS[1]),-100,-18.5)[0]
LCBGDENUP70100al=sp.integrate.quad(schechter_func(alpha=LCBG_FIT4070.alpha.value+LCBG70100ERRORS[2],phistar=LCBG_FIT4070.phistar.value,mstar=LCBG_FIT4070.mstar.value),-100,-18.5)[0]
LCBGDENDOWN70100al=sp.integrate.quad(schechter_func(alpha=LCBG_FIT4070.alpha.value-LCBG70100ERRORS[2],phistar=LCBG_FIT4070.phistar.value,mstar=LCBG_FIT4070.mstar.value),-100,-18.5)[0]

LCBGDENUPphps=np.stack((LCBGDENUP4070ps,LCBGDENUP70100ps),axis=-1)
LCBGDENLOWphps=np.stack((LCBGDENDOWN4070ps,LCBGDENDOWN70100ps),axis=-1)
LCBGDENUPphms=np.stack((LCBGDENUP4070ms,LCBGDENUP70100ms),axis=-1)
LCBGDENLOWphms=np.stack((LCBGDENDOWN4070ms,LCBGDENDOWN70100ms),axis=-1)
LCBGDENUPphal=np.stack((LCBGDENUP4070al,LCBGDENUP70100al),axis=-1)
LCBGDENLOWphal=np.stack((LCBGDENDOWN4070al,LCBGDENDOWN70100al),axis=-1)

yupphps=abs(GaldensityPhillips-LCBGDENUPphps)
ylowphps=abs(GaldensityPhillips-LCBGDENLOWphps)
yupphms=abs(GaldensityPhillips-LCBGDENUPphms)
ylowphms=abs(GaldensityPhillips-LCBGDENLOWphms)
yupphal=abs(GaldensityPhillips-LCBGDENUPphal)
ylowphal=abs(GaldensityPhillips-LCBGDENLOWphal)

yupps=abs(lcbgdensity[0:5]-LCBGDENUPps[0:5])
ylowps=abs(lcbgdensity[0:5]-LCBGDENLOWps[0:5])
yupms=abs(lcbgdensity[0:5]-LCBGDENUPms[0:5])
ylowms=abs(lcbgdensity[0:5]-LCBGDENLOWms[0:5])
yupal=abs(lcbgdensity[0:5]-LCBGDENUPal[0:5])
ylowal=abs(lcbgdensity[0:5]-LCBGDENLOWal[0:5])

yup=np.array([yupps[0],yupps[1],yupps[2],yupal[3],yupal[4]])
ylow=np.array([ylowps[0],ylowps[1],ylowps[2],ylowal[3],ylowal[4]])

phz=np.array([0.55,0.85])

f,ax=plt.subplots()
ax.errorbar(LCBGfittingparams[0:5,0],lcbgdensity[0:5],yerr=[yup,ylow],xerr=np.array([0.1,0.1,0.1,0.1,0.1]),color='black',marker='s',ls='none',label='COSMOS')
ax.errorbar(0.55,0.0022*(70./75.)**3,color='blue',marker='x',xerr=0.15,ls='none',label='Phillips, 1997')
ax.errorbar(0.85,0.0088*(70./75.)**3,color='blue',marker='x',xerr=0.15,ls='none')
ax.errorbar(0.023,0.00054*(70./75.)**3,color='red',marker='*',xerr=0.022,ls='none',label='Werk, 2004')
ax.errorbar(phz,GaldensityPhillips,color='green',marker='o',xerr=0.15,yerr=[yupphps,ylowphps],ls='none',label='COSMOS in Phillips bin')
ax.set_xlim([0,1])
plt.xlabel('Redshift')
plt.ylabel('N (Mpc$^{-3}$)')
plt.legend(fontsize='small',loc='2')

f,ax=plt.subplots(3,1,sharex=True,sharey=True)
ax[0].errorbar(LCBGfittingparams[0:5,0],lcbgdensity[0:5],yerr=[yupps,ylowps],color='red',marker='s',ls='none',label='$\phi^{*}$')
ax[0].errorbar(phz,GaldensityPhillips,yerr=[yupphps,ylowphps],color='red',marker='s',ls='none',label='$\phi^{*}$')
ax[1].errorbar(LCBGfittingparams[0:5,0],lcbgdensity[0:5],yerr=[yupms,ylowms],color='blue',marker='s',ls='none',label='$M^{*}$')
ax[1].errorbar(phz,GaldensityPhillips,yerr=[yupphms,ylowphms],color='blue',marker='s',ls='none',label='$M^{*}$')
ax[2].errorbar(LCBGfittingparams[0:5,0],lcbgdensity[0:5],yerr=[yupal,ylowal],color='green',marker='s',ls='none',label='$\alpha$')
ax[2].errorbar(phz,GaldensityPhillips,yerr=[yupphal,ylowphal],color='green',marker='s',ls='none',label='$\alpha$')
ax[0].set_xlim(0,1)

galdensity195=np.array([sp.integrate.quad(LUMFUNC_FIT020,-100,-19.5)[0],sp.integrate.quad(LUMFUNC_FIT2040,-100,-19.5)[0],sp.integrate.quad(LUMFUNC_FIT4060,-100,-19.5)[0],sp.integrate.quad(LUMFUNC_FIT6080,-100,-19.5)[0],sp.integrate.quad(LUMFUNC_FIT80100,-100,-19.5)[0],sp.integrate.quad(LUMFUNC_FIT3080,-100,-19.5)[0]])

lcbgdensity195=np.array([sp.integrate.quad(LCBG_FIT020,-100,-19.5)[0],sp.integrate.quad(LCBG_FIT2040,-100,-19.5)[0],sp.integrate.quad(LCBG_FIT4060,-100,-19.5)[0],sp.integrate.quad(LCBG_FIT6080,-100,-19.5)[0],sp.integrate.quad(LCBG_FIT80100,-100,-19.5)[0],sp.integrate.quad(LCBG_FIT3080,-100,-19.5)[0]])

