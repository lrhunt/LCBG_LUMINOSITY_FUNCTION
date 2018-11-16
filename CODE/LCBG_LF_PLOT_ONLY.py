import numpy as np
import astropy as ap
import matplotlib.pyplot as plt
import math as m
import argparse
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import custom_model
from matplotlib.backends.backend_pdf import PdfPages
import scipy as sp

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
def schechter_func(x,phistar=0.0056,mstar=-21,alpha=-1.03):
    return (0.4*np.log(10)*phistar)*(10**(0.4*(alpha+1)*(mstar-x)))*(np.e**(-np.power(10,0.4*(mstar-x))))

def schechter_func_scipy(x,phistar,mstar,alpha):
	return (0.4*np.log(10)*phistar)*(10**(0.4*(alpha+1)*(mstar-x)))*(np.e**(-np.power(10,0.4*(mstar-x))))

LCBGFIT_init=schechter_func()
LCBG_Range=np.linspace(-24,-15,30)

init_vals=[0.0056,-21,-1.03]	#Best guess at initial values, needed for scipy fitting

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
LUMFUNC_FIT020=fit(LCBGFIT_init,MBINAVE2.compressed(),LumFunc2.compressed(),weights=1/LumFuncErr2.compressed())

#Scipy Modelling

scipy_LCBG_020_fit,scipy_LCBG_020_cov=sp.optimize.curve_fit(schechter_func_scipy,LMBINAVE2.compressed(),LLumFunc2.compressed(),p0=init_vals,sigma=LLumFuncErr2.compressed())
scipy_LUMFUNC_020_fit,scipy_LUMFUNC_020_cov=sp.optimize.curve_fit(schechter_func_scipy,MBINAVE2.compressed(),LumFunc2.compressed(),p0=init_vals,sigma=LumFuncErr2.compressed())

#Defining Errors on fit parameters from scipy

LCBG020ERRORS=np.array([np.sqrt(scipy_LCBG_020_cov[0][0]),np.sqrt(scipy_LCBG_020_cov[1][1]),np.sqrt(scipy_LCBG_020_cov[2][2])])
LUMFUNC020ERRORS=np.array([np.sqrt(scipy_LUMFUNC_020_cov[0][0]),np.sqrt(scipy_LUMFUNC_020_cov[1][1]),np.sqrt(scipy_LUMFUNC_020_cov[2][2])])

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
LUMFUNC_FIT2040=fit(LCBGFIT_init,MBINAVE2.compressed(),LumFunc2.compressed(),weights=1/LumFuncErr2.compressed())

#Scipy Modelling

scipy_LCBG_2040_fit,scipy_LCBG_2040_cov=sp.optimize.curve_fit(schechter_func_scipy,LMBINAVE2.compressed(),LLumFunc2.compressed(),p0=init_vals,sigma=LLumFuncErr2.compressed())
scipy_LUMFUNC_2040_fit,scipy_LUMFUNC_2040_cov=sp.optimize.curve_fit(schechter_func_scipy,MBINAVE2.compressed(),LumFunc2.compressed(),p0=init_vals,sigma=LumFuncErr2.compressed())

#Defining Errors on fit parameters from scipy

LCBG2040ERRORS=np.array([np.sqrt(scipy_LCBG_2040_cov[0][0]),np.sqrt(scipy_LCBG_2040_cov[1][1]),np.sqrt(scipy_LCBG_2040_cov[2][2])])
LUMFUNC2040ERRORS=np.array([np.sqrt(scipy_LUMFUNC_2040_cov[0][0]),np.sqrt(scipy_LUMFUNC_2040_cov[1][1]),np.sqrt(scipy_LUMFUNC_2040_cov[2][2])])

#Plotting


LCBGcode=axes[0][1].errorbar(LMBINAVE,np.log10(LLumFunc),yerr=LLogErr,fmt=',',label='1/V$_{MAX}$ code',color='blue')
code=axes[0][1].errorbar(MBINAVE,np.log10(LumFunc),yerr=LogErr,fmt=',',label='1/V$_{MAX}$ code',color='green')
axes[0][1].errorbar(MBINAVE[LumFunc2.mask],np.log10(LumFunc[LumFunc2.mask]),yerr=LogErr[LumFunc2.mask],fmt='x',color='green')
axes[0][1].errorbar(LMBINAVE[LLumFunc2.mask],np.log10(LLumFunc[LLumFunc2.mask]),yerr=LLogErr[LLumFunc2.mask],fmt='x',label='1/V$_{MAX}$ code',color='blue')
gals=axes[1][1].bar(MBinMid,np.log10(NGal),Mbinsize,align='center',label='Number of Galaxies per Absolute Magnitude Bin',color='white')
lcbg=axes[1][1].bar(LMBinMid,np.log10(LNGal),lMbinsize,align='center',label='Number of LCBGs per Absolute Magnitude Bin',color='gray')
axes[0][1].plot(LCBG_Range,np.log10(LCBG_FIT2040(LCBG_Range)))
axes[0][1].plot(LCBG_Range,np.log10(LUMFUNC_FIT2040(LCBG_Range)))
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
LUMFUNC_FIT4060=fit(LCBGFIT_init,MBINAVE2.compressed(),LumFunc2.compressed(),weights=1/LumFuncErr2.compressed())

#Scipy Modelling

scipy_LCBG_4060_fit,scipy_LCBG_4060_cov=sp.optimize.curve_fit(schechter_func_scipy,LMBINAVE2.compressed(),LLumFunc2.compressed(),p0=init_vals,sigma=LLumFuncErr2.compressed())
scipy_LUMFUNC_4060_fit,scipy_LUMFUNC_4060_cov=sp.optimize.curve_fit(schechter_func_scipy,MBINAVE2.compressed(),LumFunc2.compressed(),p0=init_vals,sigma=LumFuncErr2.compressed())

#Defining Errors on fit parameters from scipy

LCBG4060ERRORS=np.array([np.sqrt(scipy_LCBG_4060_cov[0][0]),np.sqrt(scipy_LCBG_4060_cov[1][1]),np.sqrt(scipy_LCBG_4060_cov[2][2])])
LUMFUNC4060ERRORS=np.array([np.sqrt(scipy_LUMFUNC_4060_cov[0][0]),np.sqrt(scipy_LUMFUNC_4060_cov[1][1]),np.sqrt(scipy_LUMFUNC_4060_cov[2][2])])


LCBGcode=axes[0][2].errorbar(LMBINAVE,np.log10(LLumFunc),yerr=LLogErr,fmt=',',label='1/V$_{MAX}$ code',color='blue')
code=axes[0][2].errorbar(MBINAVE,np.log10(LumFunc),yerr=LogErr,fmt=',',label='1/V$_{MAX}$ code',color='green')
axes[0][2].errorbar(MBINAVE[LumFunc2.mask],np.log10(LumFunc[LumFunc2.mask]),yerr=LogErr[LumFunc2.mask],fmt='x',color='green')
axes[0][2].errorbar(LMBINAVE[LLumFunc2.mask],np.log10(LLumFunc[LLumFunc2.mask]),yerr=LLogErr[LLumFunc2.mask],fmt='x',label='1/V$_{MAX}$ code',color='blue')
gals=axes[1][2].bar(MBinMid,np.log10(NGal),Mbinsize,align='center',label='Number of Galaxies per Absolute Magnitude Bin',color='white')
lcbg=axes[1][2].bar(LMBinMid,np.log10(LNGal),lMbinsize,align='center',label='Number of LCBGs per Absolute Magnitude Bin',color='gray')
axes[0][2].plot(LCBG_Range,np.log10(LCBG_FIT4060(LCBG_Range)))
axes[0][2].plot(LCBG_Range,np.log10(LUMFUNC_FIT4060(LCBG_Range)))
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
LUMFUNC_FIT6080=fit(LCBGFIT_init,MBINAVE2.compressed(),LumFunc2.compressed(),weights=1/LumFuncErr2.compressed())

#Scipy Modelling

scipy_LCBG_6080_fit,scipy_LCBG_6080_cov=sp.optimize.curve_fit(schechter_func_scipy,LMBINAVE2.compressed(),LLumFunc2.compressed(),p0=init_vals,sigma=LLumFuncErr2.compressed())
scipy_LUMFUNC_6080_fit,scipy_LUMFUNC_6080_cov=sp.optimize.curve_fit(schechter_func_scipy,MBINAVE2.compressed(),LumFunc2.compressed(),p0=init_vals,sigma=LumFuncErr2.compressed())

#Defining Errors on fit parameters from scipy

LCBG6080ERRORS=np.array([np.sqrt(scipy_LCBG_6080_cov[0][0]),np.sqrt(scipy_LCBG_6080_cov[1][1]),np.sqrt(scipy_LCBG_6080_cov[2][2])])
LUMFUNC6080ERRORS=np.array([np.sqrt(scipy_LUMFUNC_6080_cov[0][0]),np.sqrt(scipy_LUMFUNC_6080_cov[1][1]),np.sqrt(scipy_LUMFUNC_6080_cov[2][2])])


LCBGcode=axes[2][0].errorbar(LMBINAVE,np.log10(LLumFunc),yerr=LLogErr,fmt=',',label='1/V$_{MAX}$ code',color='blue')
code=axes[2][0].errorbar(MBINAVE,np.log10(LumFunc),yerr=LogErr,fmt=',',label='1/V$_{MAX}$ code',color='green')
axes[2][0].errorbar(MBINAVE[LumFunc2.mask],np.log10(LumFunc[LumFunc2.mask]),yerr=LogErr[LumFunc2.mask],fmt='x',color='green')
axes[2][0].errorbar(LMBINAVE[LLumFunc2.mask],np.log10(LLumFunc[LLumFunc2.mask]),yerr=LLogErr[LLumFunc2.mask],fmt='x',label='1/V$_{MAX}$ code',color='blue')
gals=axes[3][0].bar(MBinMid,np.log10(NGal),Mbinsize,align='center',label='Number of Galaxies per Absolute Magnitude Bin',color='white')
lcbg=axes[3][0].bar(LMBinMid,np.log10(LNGal),lMbinsize,align='center',label='Number of LCBGs per Absolute Magnitude Bin',color='gray')
axes[2][0].plot(LCBG_Range,np.log10(LCBG_FIT6080(LCBG_Range)))
axes[2][0].plot(LCBG_Range,np.log10(LUMFUNC_FIT6080(LCBG_Range)))
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
LUMFUNC_FIT80100=fit(LCBGFIT_init,MBINAVE2.compressed(),LumFunc2.compressed(),weights=1/LumFuncErr2.compressed())

#Scipy Modelling

scipy_LCBG_80100_fit,scipy_LCBG_80100_cov=sp.optimize.curve_fit(schechter_func_scipy,LMBINAVE2.compressed(),LLumFunc2.compressed(),p0=init_vals,sigma=LLumFuncErr2.compressed())
scipy_LUMFUNC_80100_fit,scipy_LUMFUNC_80100_cov=sp.optimize.curve_fit(schechter_func_scipy,MBINAVE2.compressed(),LumFunc2.compressed(),p0=init_vals,sigma=LumFuncErr2.compressed())

#Defining Errors on fit parameters from scipy

LCBG80100ERRORS=np.array([np.sqrt(scipy_LCBG_80100_cov[0][0]),np.sqrt(scipy_LCBG_80100_cov[1][1]),np.sqrt(scipy_LCBG_80100_cov[2][2])])
LUMFUNC80100ERRORS=np.array([np.sqrt(scipy_LUMFUNC_80100_cov[0][0]),np.sqrt(scipy_LUMFUNC_80100_cov[1][1]),np.sqrt(scipy_LUMFUNC_80100_cov[2][2])])


LCBGcode=axes[2][1].errorbar(LMBINAVE,np.log10(LLumFunc),yerr=LLogErr,fmt=',',label='1/V$_{MAX}$ code',color='blue')
code=axes[2][1].errorbar(MBINAVE,np.log10(LumFunc),yerr=LogErr,fmt=',',label='1/V$_{MAX}$ code',color='green')
axes[2][1].errorbar(MBINAVE[LumFunc2.mask],np.log10(LumFunc[LumFunc2.mask]),yerr=LogErr[LumFunc2.mask],fmt='x',color='green')
axes[2][1].errorbar(LMBINAVE[LLumFunc2.mask],np.log10(LLumFunc[LLumFunc2.mask]),yerr=LLogErr[LLumFunc2.mask],fmt='x',label='1/V$_{MAX}$ code',color='blue')
gals=axes[3][1].bar(MBinMid,np.log10(NGal),Mbinsize,align='center',label='Number of Galaxies per Absolute Magnitude Bin',color='white')
lcbg=axes[3][1].bar(LMBinMid,np.log10(LNGal),lMbinsize,align='center',label='Number of LCBGs per Absolute Magnitude Bin',color='gray')
axes[2][1].plot(LCBG_Range,np.log10(LCBG_FIT80100(LCBG_Range)))
axes[2][1].plot(LCBG_Range,np.log10(LUMFUNC_FIT80100(LCBG_Range)))
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
LUMFUNC_FIT3080=fit(LCBGFIT_init,MBINAVE2.compressed(),LumFunc2.compressed(),weights=1/LumFuncErr2.compressed())

#Scipy Modelling

scipy_LCBG_3080_fit,scipy_LCBG_3080_cov=sp.optimize.curve_fit(schechter_func_scipy,LMBINAVE2.compressed(),LLumFunc2.compressed(),p0=init_vals,sigma=LLumFuncErr2.compressed())
scipy_LUMFUNC_3080_fit,scipy_LUMFUNC_3080_cov=sp.optimize.curve_fit(schechter_func_scipy,MBINAVE2.compressed(),LumFunc2.compressed(),p0=init_vals,sigma=LumFuncErr2.compressed())

#Defining Errors on fit parameters from scipy

LCBG3080ERRORS=np.array([np.sqrt(scipy_LCBG_3080_cov[0][0]),np.sqrt(scipy_LCBG_3080_cov[1][1]),np.sqrt(scipy_LCBG_3080_cov[2][2])])
LUMFUNC3080ERRORS=np.array([np.sqrt(scipy_LUMFUNC_3080_cov[0][0]),np.sqrt(scipy_LUMFUNC_3080_cov[1][1]),np.sqrt(scipy_LUMFUNC_3080_cov[2][2])])


LCBGcode=axes[2][2].errorbar(LMBINAVE,np.log10(LLumFunc),yerr=LLogErr,fmt=',',label='1/V$_{MAX}$ code',color='blue')
code=axes[2][2].errorbar(MBINAVE,np.log10(LumFunc),yerr=LogErr,fmt=',',label='1/V$_{MAX}$ code',color='green')
axes[2][2].errorbar(MBINAVE[LumFunc2.mask],np.log10(LumFunc[LumFunc2.mask]),yerr=LogErr[LumFunc2.mask],fmt='x',color='green')
axes[2][2].errorbar(LMBINAVE[LLumFunc2.mask],np.log10(LLumFunc[LLumFunc2.mask]),yerr=LLogErr[LLumFunc2.mask],fmt='x',label='1/V$_{MAX}$ code',color='blue')
gals=axes[3][2].bar(MBinMid,np.log10(NGal),Mbinsize,align='center',label='Number of Galaxies per Absolute Magnitude Bin',color='white')
lcbg=axes[3][2].bar(LMBinMid,np.log10(LNGal),lMbinsize,align='center',label='Number of LCBGs per Absolute Magnitude Bin',color='gray')
axes[2][2].plot(LCBG_Range,np.log10(LCBG_FIT3080(LCBG_Range)))
axes[2][2].plot(LCBG_Range,np.log10(LUMFUNC_FIT3080(LCBG_Range)))
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

#GENERATING OUTPUT 

redshiftrange=np.array([0.1,0.3,0.5,0.7,0.9,0.55])

allngal=np.array([LFGAL020,LFGAL2040,LFGAL4060,LFGAL6080,LFGAL80100,LFGAL3080])

allphistar=1000*np.array([LUMFUNC_FIT020.phistar.value,LUMFUNC_FIT2040.phistar.value,LUMFUNC_FIT4060.phistar.value,LUMFUNC_FIT6080.phistar.value,LUMFUNC_FIT80100.phistar.value,LUMFUNC_FIT3080.phistar.value])

allphistarerr=1000*np.array([LUMFUNC020ERRORS[0],LUMFUNC2040ERRORS[0],LUMFUNC4060ERRORS[0],LUMFUNC6080ERRORS[0],LUMFUNC80100ERRORS[0],LUMFUNC3080ERRORS[0]])

allmstar=np.array([LUMFUNC_FIT020.mstar.value,LUMFUNC_FIT2040.mstar.value,LUMFUNC_FIT4060.mstar.value,LUMFUNC_FIT6080.mstar.value,LUMFUNC_FIT80100.mstar.value,LUMFUNC_FIT3080.mstar.value])

allmstarerr=np.array([LUMFUNC020ERRORS[1],LUMFUNC2040ERRORS[1],LUMFUNC4060ERRORS[1],LUMFUNC6080ERRORS[1],LUMFUNC80100ERRORS[1],LUMFUNC3080ERRORS[1]])

allalpha=np.array([LUMFUNC_FIT020.alpha.value,LUMFUNC_FIT2040.alpha.value,LUMFUNC_FIT4060.alpha.value,LUMFUNC_FIT6080.alpha.value,LUMFUNC_FIT80100.alpha.value,LUMFUNC_FIT3080.alpha.value])

allalphaerr=np.array([LUMFUNC020ERRORS[2],LUMFUNC2040ERRORS[2],LUMFUNC4060ERRORS[2],LUMFUNC6080ERRORS[2],LUMFUNC80100ERRORS[2],LUMFUNC3080ERRORS[2]])

lcbgngal=np.array([LCBGGAL020,LCBGGAL2040,LCBGGAL4060,LCBGGAL6080,LCBGGAL80100,LCBGGAL3080])

lcbgphistar=1000*np.array([LCBG_FIT020.phistar.value,LCBG_FIT2040.phistar.value,LCBG_FIT4060.phistar.value,LCBG_FIT6080.phistar.value,LCBG_FIT80100.phistar.value,LCBG_FIT3080.phistar.value])

lcbgphistarerr=1000*np.array([LCBG020ERRORS[0],LCBG2040ERRORS[0],LCBG4060ERRORS[0],LCBG6080ERRORS[0],LCBG80100ERRORS[0],LCBG3080ERRORS[0]])

lcbgmstar=np.array([LCBG_FIT020.mstar.value,LCBG_FIT2040.mstar.value,LCBG_FIT4060.mstar.value,LCBG_FIT6080.mstar.value,LCBG_FIT80100.mstar.value,LCBG_FIT3080.mstar.value])

lcbgmstarerr=np.array([LCBG020ERRORS[1],LCBG2040ERRORS[1],LCBG4060ERRORS[1],LCBG6080ERRORS[1],LCBG80100ERRORS[1],LCBG3080ERRORS[1]])

lcbgalpha=np.array([LCBG_FIT020.alpha.value,LCBG_FIT2040.alpha.value,LCBG_FIT4060.alpha.value,LCBG_FIT6080.alpha.value,LCBG_FIT80100.alpha.value,LCBG_FIT3080.alpha.value])

lcbgalphaerr=np.array([LCBG020ERRORS[2],LCBG2040ERRORS[2],LCBG4060ERRORS[2],LCBG6080ERRORS[2],LCBG80100ERRORS[2],LCBG3080ERRORS[2]])

galdensityeighteen=np.array([sp.integrate.quad(LUMFUNC_FIT020,-100,-18.5)[0],sp.integrate.quad(LUMFUNC_FIT2040,-100,-18.5)[0],sp.integrate.quad(LUMFUNC_FIT4060,-100,-18.5)[0],sp.integrate.quad(LUMFUNC_FIT6080,-100,-18.5)[0],sp.integrate.quad(LUMFUNC_FIT80100,-100,-18.5)[0],sp.integrate.quad(LUMFUNC_FIT3080,-100,-18.5)[0]])

galdensityfifteen=np.array([sp.integrate.quad(LUMFUNC_FIT020,-100,-15)[0],sp.integrate.quad(LUMFUNC_FIT2040,-100,-15)[0],sp.integrate.quad(LUMFUNC_FIT4060,-100,-15)[0],sp.integrate.quad(LUMFUNC_FIT6080,-100,-15)[0],sp.integrate.quad(LUMFUNC_FIT80100,-100,-15)[0],sp.integrate.quad(LUMFUNC_FIT3080,-100,-15)[0]])

lcbgdensity=np.array([sp.integrate.quad(LCBG_FIT020,-100,-18.5)[0],sp.integrate.quad(LCBG_FIT2040,-100,-18.5)[0],sp.integrate.quad(LCBG_FIT4060,-100,-18.5)[0],sp.integrate.quad(LCBG_FIT6080,-100,-18.5)[0],sp.integrate.quad(LCBG_FIT80100,-100,-18.5)[0],sp.integrate.quad(LCBG_FIT3080,-100,-18.5)[0]])

FracGals=np.stack((redshiftrange,galdensityeighteen,galdensityfifteen,lcbgdensity),axis=-1)

np.savetxt('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/TXTFILES/Fracfromlumfunc.txt',FracGals,header='#z	galden	lcbgden')

LFfittingparams=np.stack((redshiftrange,allngal,allalpha,allalphaerr,allmstar,allmstarerr,allphistar,allphistarerr),axis=-1)
LCBGfittingparams=np.stack((redshiftrange,lcbgngal,lcbgalpha,lcbgalphaerr,lcbgmstar,lcbgmstarerr,lcbgphistar,lcbgphistarerr),axis=-1)
np.savetxt('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/TXTFILES/LFfitparams.txt',LFfittingparams,header='#zupper	alpha	alphaerr	mstar	mstarerr	phistar	phistarerr',fmt='%5.3f')
np.savetxt('/home/lrhunt/Documents/LCBG_LUM_FUNC_PAPER/TXTFILES/LCBGfitparams.txt',LCBGfittingparams,header='#zupper	alpha	alphaerr	mstar	mstarerr	phistar	phistarerr',fmt='%5.3f')

# Plotting Evolution of Parameters


f,axes=plt.subplots(nrows=2,ncols=1,sharex=True,sharey=True,gridspec_kw={'height_ratios':[1,1]})
axes[0].errorbar(LFfittingparams[0:5,0],LFfittingparams[0:5,4],yerr=LFfittingparams[0:5,5],color='black')
axes[1].errorbar(LCBGfittingparams[0:5,0],LCBGfittingparams[0:5,4],yerr=LCBGfittingparams[0:5,5],color='blue')
axes[1].set_xlim([0,1])
axes[1].set_ylim([-19.6,-21.2])
axes[1].set_yticks([-19.7,-19.9,-20.1,-20.3,-20.5,-20.7,-20.9,-21.1])
plt.subplots_adjust(hspace=0,left=0.17)
axes[0].grid()
axes[1].grid()
axes[0].text(0.5,-21,'All',fontsize=12,ha='center',va='center')
axes[1].text(0.5,-21,'LCBG',fontsize=12,ha='center',va='center')
f.text(0.05,0.5,'M$^{*}$-5log(h$_{70}$))',ha='center',va='center',rotation='vertical',fontsize=14)
f.text(0.55,0.05,'z',ha='center',va='center',fontsize=14)

f,axes=plt.subplots(nrows=2,ncols=1,sharex=True,sharey=True,gridspec_kw={'height_ratios':[1,1]})
axes[0].errorbar(LFfittingparams[0:5,0],LFfittingparams[0:5,6]/1000.,yerr=LFfittingparams[0:5,7]/1000.,color='black')
axes[1].errorbar(LCBGfittingparams[0:5,0],LCBGfittingparams[0:5,6]/1000.,yerr=LCBGfittingparams[0:5,7]/1000.,color='blue')
axes[1].set_xlim([0,1])
axes[1].set_ylim([0,0.01])
axes[1].set_yticks([0.001,0.003,0.005,0.007,0.009])
plt.subplots_adjust(hspace=0,left=0.17)
axes[0].grid()
axes[1].grid()
axes[0].text(0.5,0.009,'All',fontsize=12,ha='center',va='center')
axes[1].text(0.5,0.009,'LCBG',fontsize=12,ha='center',va='center')
f.text(0.05,0.5,'$\Phi^{*}$ ($h_{70}^{3}Mpc^{-3} mag^{-1}$)',ha='center',va='center',rotation='vertical',fontsize=14)
f.text(0.55,0.05,'z',ha='center',va='center',fontsize=14)

