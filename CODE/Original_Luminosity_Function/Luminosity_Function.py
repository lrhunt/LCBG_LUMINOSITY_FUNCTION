
# coding: utf-8

# In[1]:


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
import pandas as pd


# python Luminosity_Function.py /Users/lucashunt/ASTRODATA/LCBG_LUMINOSITY_FUNCTION/COSMOS_CATALOGS/Final_Catalogs/Laigle_LCBG_Catalog.csv -mbin 18 -zmin 0.01 -zmax 0.2 -mmin -24 -mmax -15 -ama 23 -ami 15 -om 0.3 -ho 70

# In[2]:


parser = argparse.ArgumentParser()
parser.add_argument("filein",help="File containing magnitudes and redshifts")
parser.add_argument("-mbin","--mbin",type=int,help="Number of Absolute Magnitude bins. Default=19",default=18)
parser.add_argument("-zmin","--zmin",type=float,help="Minimum redshift to consider in luminosity function, default=0.01", default=0.01)
parser.add_argument("-zmax","--zmax",type=float,help="Maximum redshift to consider in luminosity function, default=2", default=0.2)
parser.add_argument("-mmin","--Mmin",type=float,help="Minumum absolute magnitude to consider in luminosity function, default=-24", default=-24)
parser.add_argument("-mmax","--Mmax",type=float,help="Maximum absolute magnitude to consider in luminosity function, default=-15", default=-15)
parser.add_argument("-ama","--appmax",type=float,help='Maximum apparent magnitude to consider part of the survey COSMOS i<22.5',default=22.5)
parser.add_argument("-ami","--appmin",type=float,help='Minimum apparent magnitude to consider part of the survey COSMOS i>15',default=15)
parser.add_argument("-om","--OmegaMatter",type=float,help="Omega Matter, if you want to define your own cosmology", default=0.3)
parser.add_argument("-ho","--HubbleConstant",type=float,help="Hubble Constant if you want to define your own cosmology",default=70)
parser.add_argument("-lfdir","--lfdir",help="Directroy you want to put the Luminosity Function txt file and the underlying catalog in",default='/home/lrhunt/Projects/LCBG_LUMINOSITY_FUNCTION/FULLLFOUT/')
parser.add_argument("-fileout","--fileout",type=str,help="Set a string if you want to give a filename if not it will default to LF_zmin_zmax.csv",default='')
parser.add_argument("-LCBG","--LCBGLIST",action="store_true",help="Make Luminosity Function with LCBGs only?")
parser.add_argument("-nv","--novega",action="store_true",help="Do not apply correction to switch from AB to Vega magnitudes")
args=parser.parse_args()


# Setting values needed throughout the code

# In[3]:


cosmo=FlatLambdaCDM(H0=args.HubbleConstant,Om0=args.OmegaMatter)
mbinsize=float(args.Mmax-args.Mmin)/args.mbin
Magnitude_Loop_Array=np.stack((np.arange(args.appmin,args.appmax+0.5,0.5)[0:len(np.arange(args.appmin,args.appmax+0.5,0.5))-1],np.arange(args.appmin,args.appmax+0.5,0.5)[1:len(np.arange(args.appmin,args.appmax+0.5,0.5))]),axis=-1)
kcordir=os.environ["KCORRECT_DIR"]


# In[5]:


print('********READING FILE********')
CATALOG=pd.read_csv(args.filein)


# In[6]:


print('********CALCULATING WEIGHTS********')
tf=open('weights.txt','w')
CATALOG['Spec_Weight']=np.nan
CATALOG['Color_Weight']=np.nan
CATALOG['Surface_Brightness_Weight']=np.nan
for magrange in Magnitude_Loop_Array:
    Num_Good_Spec=len(CATALOG.loc[(CATALOG.subaru_i_mag>magrange[0])&
                              (CATALOG.subaru_i_mag<magrange[1])&
                              (CATALOG.SG_MASTER==0)&
                              ((CATALOG.Z_USE==1)|
                               (CATALOG.Z_USE==2))])
    Num_Bad_Spec=len(CATALOG.loc[(CATALOG.subaru_i_mag>magrange[0])&
                             (CATALOG.subaru_i_mag<magrange[1])&
                             (CATALOG.SG_MASTER==0)&
                             ((CATALOG.Z_USE==3)|
                              (CATALOG.Z_USE==4))])
    Num_Good_Color=len(CATALOG.loc[(CATALOG.subaru_i_mag>magrange[0])&
                               (CATALOG.subaru_i_mag<magrange[1])&
                               (CATALOG.SG_MASTER==0)&
                               ((CATALOG.subaru_B_mag<100)&
                                (CATALOG.subaru_V_mag<100))])
    Num_Bad_Color=len(CATALOG.loc[(CATALOG.subaru_i_mag>magrange[0])&
                              (CATALOG.subaru_i_mag<magrange[1])&
                              (CATALOG.SG_MASTER==0)&
                              ((CATALOG.subaru_B_mag>100)|
                               (CATALOG.subaru_V_mag>100))])
    Num_Bad_Rh=len(CATALOG.loc[(CATALOG.subaru_i_mag>magrange[0])&
                           (CATALOG.subaru_i_mag<magrange[1])&
                           (CATALOG.SG_MASTER==0)&
                           (np.isnan(CATALOG.R_HALF_PIXELS))])
    Num_Good_Rh=len(CATALOG.loc[(CATALOG.subaru_i_mag>magrange[0])&
                            (CATALOG.subaru_i_mag<magrange[1])&
                            (CATALOG.SG_MASTER==0)&
                            (~np.isnan(CATALOG.R_HALF_PIXELS))])
    CATALOG.loc[(CATALOG.subaru_i_mag>magrange[0])&
                (CATALOG.subaru_i_mag<magrange[1])&
                (CATALOG.SG_MASTER==0),
                'Spec_Weight']=float(Num_Good_Spec+Num_Bad_Spec)/float(Num_Good_Spec)
    CATALOG.loc[(CATALOG.subaru_i_mag>magrange[0])&
                (CATALOG.subaru_i_mag<magrange[1])&
                (CATALOG.SG_MASTER==0),
                'Color_Weight']=float(Num_Good_Color+Num_Bad_Color)/float(Num_Good_Color)
    CATALOG.loc[(CATALOG.subaru_i_mag>magrange[0])&
                (CATALOG.subaru_i_mag<magrange[1])&
                (CATALOG.SG_MASTER==0),
                'Surface_Brightness_Weight']=float(Num_Good_Rh+Num_Bad_Rh)/float(Num_Good_Rh)
    print('Spec Weight = {} | Color Weight = {} | Surface Brightness Weight = {}'.format(
        np.round(float(Num_Good_Spec+Num_Bad_Spec)/float(Num_Good_Spec),4),
        np.round(float(Num_Good_Color+Num_Bad_Color)/float(Num_Good_Color),4),
        np.round(float(Num_Good_Rh+Num_Bad_Rh)/float(Num_Good_Rh),4)))

    


# ***********************************
# Starting the Luminosity Function part of the code! 
# ***********************************

# Breaking up the large catalog into a smaller one containing the sources over the correct redshift range and apparent magnitude range

# In[ ]:


CATALOG['is_LCBG']=0
CATALOG.loc[(CATALOG.BJ0_vega_absmag.values<-18.5)&(CATALOG.BJ0_vega_surface_brightness.values<21)&(CATALOG['rest_frame_B-V'].values<0.6),'is_LCBG']=1


# In[7]:


print('********LOOKING FOR LCBGS********')
if args.LCBGLIST:
    LUMFUNC_CATALOG=CATALOG.loc[(CATALOG.Z_USE<3)&
                            (CATALOG.subaru_i_mag<=args.appmax)&
                            (CATALOG.subaru_i_mag>=args.appmin)&
                            (CATALOG.SG_MASTER==0)&
                            (CATALOG.Z_BEST>=args.zmin)&
                            (CATALOG.Z_BEST<=args.zmax)&
                            (CATALOG.is_LCBG==1)]
else:
    LUMFUNC_CATALOG=CATALOG.loc[(CATALOG.Z_USE<3)&
                            (CATALOG.subaru_i_mag<=args.appmax)&
                            (CATALOG.subaru_i_mag>=args.appmin)&
                            (CATALOG.SG_MASTER==0)&
                            (CATALOG.Z_BEST>=args.zmin)&
                            (CATALOG.Z_BEST<=args.zmax)]


# Setting apparent magnitude limits to calculate the range over which this source would be detected. 

# In[43]:


print('********FINDING UPPER AND LOWER REDSHIFTS********')
kcorrect.load_templates()
kcorrect.load_filters(kcordir+'/data/templates/subaru_i.dat')
zlookup=np.linspace(0,1,1000)
for x in LUMFUNC_CATALOG.index:
    rmarrlookup=np.ndarray(1000)
    for j in range(0,1000):
        rmarrlookup[j]=kcorrect.reconstruct_maggies(LUMFUNC_CATALOG.loc[x,'c1':'c6'],redshift=zlookup[j])[1:]
    AbsMag=LUMFUNC_CATALOG.subaru_i_mag.loc[x]-cosmo.distmod(LUMFUNC_CATALOG.Z_BEST.loc[x]).value-LUMFUNC_CATALOG.subaru_i_synthetic_mag.loc[x]-2.5*np.log10(rmarrlookup[0])
    ilookup=AbsMag+cosmo.distmod(zlookup).value-2.5*np.log10(rmarrlookup)+2.5*np.log10(rmarrlookup[0])
    LUMFUNC_CATALOG.loc[x,'upper_redshift']=round(zlookup[np.abs(ilookup-args.appmax).argmin()],4)
    LUMFUNC_CATALOG.loc[x,'lower_redshift']=round(zlookup[np.abs(ilookup-args.appmin).argmin()],4)


# In[48]:


LUMFUNC_CATALOG.loc[LUMFUNC_CATALOG.upper_redshift>args.zmax,'upper_redshift']=args.zmax
LUMFUNC_CATALOG.loc[LUMFUNC_CATALOG.lower_redshift<args.zmin,'lower_redshift']=args.zmin
LUMFUNC_CATALOG['comoving_volume']=cosmo.comoving_volume(LUMFUNC_CATALOG.upper_redshift).value/(4*np.pi/0.0003116)-cosmo.comoving_volume(LUMFUNC_CATALOG.lower_redshift).value/(4*np.pi/0.0003116)


# In[49]:


print('********CALCULATING VALUES FOR LUMINOSITY FUNCTION********')
Abs_Magnitude_Loop_Array=np.stack((np.arange(args.Mmin,args.Mmax+mbinsize,mbinsize)[0:len(np.arange(args.Mmin,args.Mmax,mbinsize))],np.arange(args.Mmin,args.Mmax+mbinsize,mbinsize)[1:len(np.arange(args.Mmin,args.Mmax,mbinsize))+1]),axis=-1)
Abs_Mags=[]
NUMB_DENS_LIST=[]
NUMB_DENS_Err=[]
NGal=[]
AveCMV=[]
AveWeight=[]
for rng in Abs_Magnitude_Loop_Array:
    if args.LCBGLIST:
        NUMB_DENS_LIST.append(((
            LUMFUNC_CATALOG.loc[(LUMFUNC_CATALOG.BJ0_vega_absmag>rng[0])&
                                (LUMFUNC_CATALOG.BJ0_vega_absmag<rng[1]),
                                'Spec_Weight']
            *LUMFUNC_CATALOG.loc[(LUMFUNC_CATALOG.BJ0_vega_absmag>rng[0])&
                                (LUMFUNC_CATALOG.BJ0_vega_absmag<rng[1]),
                                'Color_Weight']
            *LUMFUNC_CATALOG.loc[(LUMFUNC_CATALOG.BJ0_vega_absmag>rng[0])&
                                (LUMFUNC_CATALOG.BJ0_vega_absmag<rng[1]),
                                'Surface_Brightness_Weight'])
            /(LUMFUNC_CATALOG.loc[(LUMFUNC_CATALOG.BJ0_vega_absmag>rng[0])&
                                  (LUMFUNC_CATALOG.BJ0_vega_absmag<rng[1]),
                                  'comoving_volume']*mbinsize)).sum())
        AveWeight.append(((
            LUMFUNC_CATALOG.loc[(LUMFUNC_CATALOG.BJ0_vega_absmag>rng[0])&
                                (LUMFUNC_CATALOG.BJ0_vega_absmag<rng[1]),
                                'Spec_Weight']
            *LUMFUNC_CATALOG.loc[(LUMFUNC_CATALOG.BJ0_vega_absmag>rng[0])&
                                 (LUMFUNC_CATALOG.BJ0_vega_absmag<rng[1]),
                                 'Color_Weight']
            *LUMFUNC_CATALOG.loc[(LUMFUNC_CATALOG.BJ0_vega_absmag>rng[0])&
                                 (LUMFUNC_CATALOG.BJ0_vega_absmag<rng[1]),
                                 'Surface_Brightness_Weight']).sum()
            /len(LUMFUNC_CATALOG.loc[(LUMFUNC_CATALOG.BJ0_vega_absmag>rng[0])&
                                     (LUMFUNC_CATALOG.BJ0_vega_absmag<rng[1]),
                                     'Spec_Weight'])))
        NUMB_DENS_Err.append(
            np.sqrt(
                (
                    (LUMFUNC_CATALOG.loc[(LUMFUNC_CATALOG.BJ0_vega_absmag>rng[0])&
                                         (LUMFUNC_CATALOG.BJ0_vega_absmag<rng[1]),
                                         'Spec_Weight']*
                     LUMFUNC_CATALOG.loc[(LUMFUNC_CATALOG.BJ0_vega_absmag>rng[0])&
                                         (LUMFUNC_CATALOG.BJ0_vega_absmag<rng[1]),
                                         'Color_Weight']*
                     LUMFUNC_CATALOG.loc[(LUMFUNC_CATALOG.BJ0_vega_absmag>rng[0])&
                                         (LUMFUNC_CATALOG.BJ0_vega_absmag<rng[1]),
                                         'Surface_Brightness_Weight'])
                    /(LUMFUNC_CATALOG.loc[(LUMFUNC_CATALOG.BJ0_vega_absmag>rng[0])&
                                          (LUMFUNC_CATALOG.BJ0_vega_absmag<rng[1])
                                          ,'comoving_volume']*mbinsize)**2).sum()))
    else:
        NUMB_DENS_LIST.append(
            (LUMFUNC_CATALOG.loc[(LUMFUNC_CATALOG.BJ0_vega_absmag>rng[0])&
                                 (LUMFUNC_CATALOG.BJ0_vega_absmag<rng[1]),
                                 'Spec_Weight']
             /(LUMFUNC_CATALOG.loc[(LUMFUNC_CATALOG.BJ0_vega_absmag>rng[0])&
                                   (LUMFUNC_CATALOG.BJ0_vega_absmag<rng[1]),
                                   'comoving_volume']*
               mbinsize)).sum())
        NUMB_DENS_Err.append(np.sqrt(
            (LUMFUNC_CATALOG.loc[(LUMFUNC_CATALOG.BJ0_vega_absmag>rng[0])&
                                 (LUMFUNC_CATALOG.BJ0_vega_absmag<rng[1]),
                                 'Spec_Weight']
             /(LUMFUNC_CATALOG.loc[(LUMFUNC_CATALOG.BJ0_vega_absmag>rng[0])&
                                   (LUMFUNC_CATALOG.BJ0_vega_absmag<rng[1]),
                                   'comoving_volume']
               *mbinsize)**2).sum()))
        AveWeight.append(
            LUMFUNC_CATALOG.loc[(LUMFUNC_CATALOG.BJ0_vega_absmag>rng[0])&
                                             (LUMFUNC_CATALOG.BJ0_vega_absmag<rng[1]),
                                'Spec_Weight'].sum()/
            len(LUMFUNC_CATALOG.loc[(LUMFUNC_CATALOG.BJ0_vega_absmag>rng[0])&
                                    (LUMFUNC_CATALOG.BJ0_vega_absmag<rng[1]),
                                    'Spec_Weight']))
    NGal.append(
        len(LUMFUNC_CATALOG.loc[(LUMFUNC_CATALOG.BJ0_vega_absmag>rng[0])&
                                (LUMFUNC_CATALOG.BJ0_vega_absmag<rng[1]),
                                'Spec_Weight']))
    AveCMV.append(LUMFUNC_CATALOG.loc[(LUMFUNC_CATALOG.BJ0_vega_absmag>rng[0])&
                                      (LUMFUNC_CATALOG.BJ0_vega_absmag<rng[1]),
                                      'comoving_volume'].sum()
                  /len(LUMFUNC_CATALOG.loc[(LUMFUNC_CATALOG.BJ0_vega_absmag>rng[0])&
                                           (LUMFUNC_CATALOG.BJ0_vega_absmag<rng[1]),
                                           'Spec_Weight']))
    Abs_Mags.append(LUMFUNC_CATALOG.loc[(LUMFUNC_CATALOG.BJ0_vega_absmag>rng[0])&
                                        (LUMFUNC_CATALOG.BJ0_vega_absmag<rng[1]),
                                        'BJ0_vega_absmag'].mean())


# In[51]:


Luminosity_Function=pd.DataFrame(
    {'Number_Of_Gals':NGal,'Absolute_Magnitude_Bin':np.average(Abs_Magnitude_Loop_Array,axis=-1),
     'Average_Absolute_Magnitude':Abs_Mags,
     'Number_Density':NUMB_DENS_LIST,
     'Number_Density_Error':NUMB_DENS_Err,
     'Average_Comoving_Volume':AveCMV,
     'Average_Weight':AveWeight})


# In[52]:


Luminosity_Function['Log10Phi']=np.log10(Luminosity_Function['Number_Density'])
Luminosity_Function['Log10Err']=Luminosity_Function.Number_Density_Error/(Luminosity_Function.Number_Density*np.log(10))


# In[55]:


print('********SAVING FILE********')
if args.LCBGLIST:
    Luminosity_Function.to_csv(args.lfdir+'LF_{}_{}_LCBG.csv'.format(int(args.zmin*100),int(args.zmax*100)))
else:
    Luminosity_Function.to_csv(args.lfdir+'LF_{}_{}.csv'.format(int(args.zmin*100),int(args.zmax*100)))

