#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import astropy as ap
from astropy import units as u
from astropy.coordinates import SkyCoord
import kcorrect
import kcorrect.utils as ut
from astropy.cosmology import FlatLambdaCDM
import os
import matplotlib.pyplot as plt
from itertools import combinations


# Defining Cosmology

# In[2]:


cosmo=FlatLambdaCDM(H0=70,Om0=0.3)


# Reading in Catalogs (LAMBDAR and TASCA)

# In[37]:


def make_kcorr_filt_template(dataframe):
    '''This task will make a kcorrect filter template from a dataframe that optimizes the number of objects with detections in a subset of filters. In this case the dataframe should contain cfht, subaru, and irac wideband filters. '''
    kcordir=os.environ["KCORRECT_DIR"]
    lambdar_to_kcorr={'mag_cfht_u':'capak_cfht_megaprime_sagem_u.par','mag_subaru_B':'capak_subaru_suprimecam_B.par','mag_subaru_V':'capak_subaru_suprimecam_V.par','mag_subaru_g':'capak_subaru_suprimecam_g.par','mag_subaru_r':'capak_subaru_suprimecam_r.par','mag_subaru_i':'capak_subaru_suprimecam_i.par','mag_subaru_z':'capak_subaru_suprimecam_z.par','mag_irac_1':'spitzer_irac_ch1.par','mag_irac_2':'spitzer_irac_ch2.par','mag_irac_3':'spitzer_irac_ch3.par','mag_irac_4':'spitzer_irac_ch4.par'}
    numb1=0
    numb2=0
    numb3=0
    numb4=0
    numb5=0
    flist1=[]
    flist2=[]
    flist3=[]
    flist4=[]
    flist5=[]
    ilist1=[]
    ilist2=[]
    ilist3=[]
    ilist4=[]
    ilist5=[]
    kcor_template=kcordir+'/data/templates/temp_filt_list.dat'
    for x in combinations(list(dataframe),5):
        if len(dataframe[(dataframe[x[0]]<40)&(dataframe[x[1]]<40)&(dataframe[x[2]]<40)&(dataframe[x[3]]<40)&(dataframe[x[4]]<40)]) > numb1:
            ilist5=ilist4
            ilist4=ilist3
            ilist3=ilist2
            ilist2=ilist1
            ilist1=dataframe[(dataframe[x[0]]<40)&(dataframe[x[1]]<40)&(dataframe[x[2]]<40)&(dataframe[x[3]]<40)&(dataframe[x[4]]<40)].index.tolist()
            numb5=numb4
            numb4=numb3
            numb3=numb2
            numb2=numb1
            numb1=len(ilist1)
            flist5=flist4
            flist4=flist3
            flist3=flist2
            flist2=flist1
            flist1=x
    with open(kcor_template,'w') as file:
        file.write('KCORRECT_DIR\n')
        for filt in flist1:
            file.write('data/filters/'+lambdar_to_kcorr[filt]+'\n')
    return flist1,kcor_template


# In[4]:


print('Reading in Catalogs')
COSMOS_PHOT_LAMBDAR=pd.read_csv('/home/lrhunt/Astrodata/LCBG_Luminosity_Function/Original_Catalogs/G10CosmosLAMBDARCatv05.csv')
TASCA_COSMOS_MORPH=pd.read_csv('/home/lrhunt/Astrodata/LCBG_Luminosity_Function/Original_Catalogs/tasca_morphology.tbl',delim_whitespace=True,header=30,dtype=float,error_bad_lines=False,skiprows=[31,32,33])
CASSATA_COSMOS_MORPH=pd.read_csv('/home/lrhunt/Astrodata/LCBG_Luminosity_Function/Original_Catalogs/cassata_morphology.tbl',delim_whitespace=True,header=30,dtype=float,error_bad_lines=False,skiprows=[31,32,33])
ZURICH_COSMOS_MORPH=pd.read_csv('/home/lrhunt/Astrodata/LCBG_Luminosity_Function/Original_Catalogs/zurich_morphology.tbl',delim_whitespace=True,header=32,error_bad_lines=False,skiprows=[33,34,35])


# In[5]:


ZURICH_COSMOS_MORPH=ZURICH_COSMOS_MORPH.drop(columns='|')


# In[6]:


ZURICH_COSMOS_MORPH.columns=ZURICH_COSMOS_MORPH.columns.str.strip('|')
CASSATA_COSMOS_MORPH.columns=CASSATA_COSMOS_MORPH.columns.str.strip('|')
TASCA_COSMOS_MORPH.columns=TASCA_COSMOS_MORPH.columns.str.strip('|')


# In[8]:


HALF_LIGHT_RADII=CASSATA_COSMOS_MORPH.copy()


# In[9]:


HALF_LIGHT_RADII=HALF_LIGHT_RADII.rename(columns={'r_half':'r_half_cassata'})


# Setting Variables

# In[10]:


kcordir=os.environ["KCORRECT_DIR"]


# *********************************************
# Matching Catalogs
# *********************************************

# In[11]:


print('Matching Catalogs')


# Generating Astropy skycoord objects to easily match catalogs based on position

# In[12]:


CASSATA_COORD=SkyCoord(ra=CASSATA_COSMOS_MORPH['ra'].values*u.degree,dec=CASSATA_COSMOS_MORPH['dec'].values*u.degree)
ZURICH_COORD=SkyCoord(ra=ZURICH_COSMOS_MORPH['ra'].values*u.degree,dec=ZURICH_COSMOS_MORPH['dec'].values*u.degree)
TASCA_COORD=SkyCoord(ra=TASCA_COSMOS_MORPH['ra'].values*u.degree,dec=TASCA_COSMOS_MORPH['dec'].values*u.degree)
G10_COORD=SkyCoord(ra=COSMOS_PHOT_LAMBDAR['RA'].values*u.degree,dec=COSMOS_PHOT_LAMBDAR['DEC'].values*u.degree)
RH_COORD=SkyCoord(ra=HALF_LIGHT_RADII['ra'].values*u.degree,dec=HALF_LIGHT_RADII['dec'].values*u.degree)


# 
# Matching Catalogs

# In[13]:


idxCASSATA_t,idxTASCA_c,d2d_ct,d3d_tc=TASCA_COORD.search_around_sky(CASSATA_COORD,0.1*u.arcsecond)
idxCASSATA_z,idxZURICH_c,d2d_cz,d3d_zc=ZURICH_COORD.search_around_sky(CASSATA_COORD,0.1*u.arcsecond)


# In[14]:


mask=np.ones(len(idxCASSATA_z),dtype=bool)
separation_list=[]
three_obj_within_1_arcsec=[]
for i in range(0,len(idxCASSATA_z)-2):
    if idxZURICH_c[i]==idxZURICH_c[i+1]:
        separation_list.append([d2d_cz[i].arcsecond,d2d_cz[i+1].arcsecond])
        if d2d_cz[i]>d2d_cz[i+1]:
            mask[i]=False
        if d2d_cz[i]<d2d_cz[i+1]:
            mask[i+1]=False
        if d2d_cz[i]==d2d_cz[i+1]:
            mask[i+1]=False
        print(mask[i])
        print(mask[i+1])
    if idxZURICH_c[i]==idxZURICH_c[i+2]:
        three_obj_within_1_arcsec.append([d2d_cz[i].arcsecond,d2d_cz[i+1].arcsecond,d2d_zc[i+2].arcsecond])


# In[15]:


separation_list=[]
three_obj_within_1_arcsec=[]
for i in range(0,len(idxCASSATA_t)-2):
    if idxTASCA_c[i]==idxTASCA_c[i+1]:
        separation_list.append([d2d_ct[i].arcsecond,d2d_ct[i+1].arcsecond])
    if idxTASCA_c[i]==idxTASCA_c[i+2]:
        three_obj_within_1_arcsec.append([d2d_ct[i].arcsecond,d2d_ct[i+1].arcsecond,d2d_zt[i+2].arcsecond])


# In[16]:


HALF_LIGHT_RADII.loc[idxCASSATA_t,'r_half_tasca']=np.array(TASCA_COSMOS_MORPH.r_half[idxTASCA_c])
HALF_LIGHT_RADII.loc[idxCASSATA_t,'acs_mag_tasca']=np.array(TASCA_COSMOS_MORPH.acs_mag_auto[idxTASCA_c])
HALF_LIGHT_RADII.loc[idxCASSATA_z,'r_half_zurich']=np.array(ZURICH_COSMOS_MORPH.r50[idxZURICH_c])
HALF_LIGHT_RADII.loc[idxCASSATA_z,'acs_mag_zurich']=np.array(TASCA_COSMOS_MORPH.acs_mag_auto[idxZURICH_c])


# In[17]:


HALF_LIGHT_RADII['CASSATA_SB']=HALF_LIGHT_RADII.mag_auto_acs+0.753+2.5*np.log10(np.pi*(HALF_LIGHT_RADII.r_half_cassata*0.03)**2)


# In[18]:


HALF_LIGHT_RADII['TASCA_SB']=HALF_LIGHT_RADII.acs_mag_tasca+0.753+2.5*np.log10(np.pi*(HALF_LIGHT_RADII.r_half_tasca*0.03)**2)


# In[19]:


HALF_LIGHT_RADII['ZURICH_SB']=HALF_LIGHT_RADII.acs_mag_zurich+0.753+2.5*np.log10(np.pi*(HALF_LIGHT_RADII.r_half_zurich*0.03)**2)


# In[22]:


idxRH,idxG10,d2dG10,d3dG10=G10_COORD.search_around_sky(RH_COORD,0.1*u.arcsecond)


# In[23]:


COSMOS_PHOT_LAMBDAR['r_half_cassata']=np.nan
COSMOS_PHOT_LAMBDAR['r_half_tasca']=np.nan
COSMOS_PHOT_LAMBDAR['r_half_zurich']=np.nan
COSMOS_PHOT_LAMBDAR.loc[idxG10,['r_half_cassata','r_half_tasca','r_half_zurich']]=np.array(HALF_LIGHT_RADII.loc[idxRH,['r_half_cassata','r_half_tasca','r_half_zurich']])*0.03


# In[24]:


COSMOS_FLUXES=COSMOS_PHOT_LAMBDAR.copy()


# *********************************************
# Calculating apparent magnitudes
# *********************************************

# In[25]:


print('Calculating apparent magnitudes')


# Determining what data is in each column

# In[26]:


column_headers=COSMOS_FLUXES.columns.values
telescope_flux_headers=[s for s in column_headers if (('subaru' in s) or ('uvista' in s) or ('galex' in s) or ('cfht' in s) or ('irac' in s) or ('mips' in s) or ('pacs' in s) or ('spire' in s)) and ('err' not in s)]


# Calculating magnitudes (And creating column for maggies, see kcorrect.org for explanation)

# In[27]:


for i in telescope_flux_headers:
    COSMOS_FLUXES['mag_'+i]=-2.5*np.log10(COSMOS_FLUXES[i]/3631)
    COSMOS_FLUXES['mag_'+i+'_err']=2.5/np.log(10)*COSMOS_FLUXES[i+'_err']/COSMOS_FLUXES[i]
    COSMOS_FLUXES['maggies_'+i]=np.nan
    COSMOS_FLUXES['invervar_'+i]=np.nan

COSMOS_FLUXES.loc[COSMOS_FLUXES.isnull().mag_galex_fuv,'mag_galex_fuv_err']=np.nan
COSMOS_FLUXES.loc[COSMOS_FLUXES.isnull().mag_galex_nuv,'mag_galex_nuv_err']=np.nan


# In[28]:


optimize_kcorr_filts=COSMOS_FLUXES[[filt for filt in COSMOS_FLUXES.columns.values if ('mag' in filt) & ('maggies' not in filt) & (('subaru' in filt) | ('irac' in filt) | ('cfht' in filt)) & ('ia' not in filt) & ('nb' not in filt) & ('err' not in filt)]]


# In[38]:


kcorr_filt_list,kcor_template_filter=make_kcorr_filt_template(optimize_kcorr_filts)


# In[39]:


kcorr_filt_list


# In[50]:


kcor_template_filter


# *********************************************
# Getting k-correction
# *********************************************

# Making columns for kcorrect output values

# In[40]:


numbers=np.arange(1,7,1)
for string in ['c'+str(number) for number in numbers]:
    COSMOS_FLUXES[string]=np.nan


# Making list that contains filters used for k-correction, making columns for synthetic filters. 

# Converting to maggies and invervar

# In[41]:


column_headers=COSMOS_FLUXES.columns.values


# In[42]:


print('Converting to maggies')
column_headers=COSMOS_FLUXES.columns.values
for column in column_headers:
    if 'maggies' in column:
        COSMOS_FLUXES[column]=ut.mag2maggies(COSMOS_FLUXES['mag_'+column.split('maggies_')[1]])
        COSMOS_FLUXES['invervar_'+column.split('maggies_')[1]]=ut.invariance(COSMOS_FLUXES[column],COSMOS_FLUXES['mag_'+column.split('maggies_')[1]+'_err'])


# Here we make a list of the filters we want to use for the k-correct code

# In[43]:


for f in kcorr_filt_list:
    COSMOS_FLUXES['maggies_'+f.split('mag_')[1]+'_synthetic']=np.nan
    COSMOS_FLUXES['maggies_'+f.split('mag_')[1]+'0_synthetic']=np.nan


# Loading Filters

# In[44]:


kcorrect.load_templates()
kcorrect.load_filters(kcor_template_filter)


# Creating an index array so that we step through each source to calculate the coefficients needed for k-correct

# In[45]:


cmm_ind=COSMOS_FLUXES.index.values


# In[46]:


for i in cmm_ind:
    COSMOS_FLUXES.loc[i,'c1':'c6']=kcorrect.fit_nonneg(np.array(COSMOS_FLUXES.loc[i,'Z_BEST'],dtype=float),np.array(COSMOS_FLUXES.loc[i,['maggies_'+col.split('mag_')[1] for col in kcorr_filt_list]],dtype=float),np.array(COSMOS_FLUXES.loc[i,['invervar_'+col.split('mag_')[1] for col in kcorr_filt_list]],dtype=float))


# Recalculating the "maggies" for each object based on the best guess of the SED

# In[47]:


kcorrect.load_templates()
kcorrect.load_filters(kcor_template_filter)


# In[ ]:


for i in cmm_ind:
    COSMOS_FLUXES.loc[i,['maggies_'+f.split('mag_')[1]+'_synthetic' for f in kcorr_filt_list]]=kcorrect.reconstruct_maggies(COSMOS_FLUXES.loc[i,'c1':'c6'])[1:]
    COSMOS_FLUXES.loc[i,['maggies_'+f.split('mag_')[1]+'0_synthetic' for f in kcorr_filt_list]]=kcorrect.reconstruct_maggies(COSMOS_FLUXES.loc[i,'c1':'c6'],redshift=0)[1:]


# Converting the synthetic maggies to apparent magnitude

# In[ ]:


COSMOS_FLUXES[[f+'_synthetic' for f in kcorr_filt_list]]=-2.5*np.log10(COSMOS_FLUXES[['maggies_'+f.split('mag_')[1]+'_synthetic' for f in kcorr_filt_list]])
COSMOS_FLUXES[[f+'0_synthetic' for f in kcorr_filt_list]]=-2.5*np.log10(COSMOS_FLUXES[['maggies_'+f.split('mag_')[1]+'0_synthetic' for f in kcorr_filt_list]])


# Calculating Johnson U/B/V from the SED

# In[ ]:


COSMOS_FLUXES['U0_synthetic']=np.nan
COSMOS_FLUXES['B0_synthetic']=np.nan
COSMOS_FLUXES['V0_synthetic']=np.nan


# In[ ]:


kcorrect.load_templates()
kcorrect.load_filters(kcordir+'/data/templates/bessell_ubv.dat')


# In[ ]:


for i in cmm_ind:
    COSMOS_FLUXES.loc[i,['U0_synthetic','B0_synthetic','V0_synthetic']]=kcorrect.reconstruct_maggies(COSMOS_FLUXES.loc[i,'c1':'c6'],redshift=0)[1:]


# Converting to magnitudes and then adding offsets to go from AB magnitudes to Vega magnitudes

# In[ ]:


COSMOS_FLUXES[['U0_synthetic_mag','B0_synthetic_mag','V0_synthetic_mag']]=-2.5*np.log10(COSMOS_FLUXES[['U0_synthetic','B0_synthetic','V0_synthetic']])
COSMOS_FLUXES['U0_synthetic_mag']=COSMOS_FLUXES['U0_synthetic_mag']-0.79
COSMOS_FLUXES['B0_synthetic_mag']=COSMOS_FLUXES['B0_synthetic_mag']+0.09
COSMOS_FLUXES['V0_synthetic_mag']=COSMOS_FLUXES['V0_synthetic_mag']-0.02


# Actually calculating k-correction from Subaru filters to Bessell Filters

# In[ ]:


COSMOS_FLUXES[[f+'_kcorr_B' for f in kcorr_filts]]=-2.5*np.log10(COSMOS_FLUXES[[f+'_synthetic' for f in kcorr_filts]]/np.stack((COSMOS_FLUXES['B0_synthetic'],COSMOS_FLUXES['B0_synthetic'],COSMOS_FLUXES['B0_synthetic'],COSMOS_FLUXES['B0_synthetic'],COSMOS_FLUXES['B0_synthetic'],COSMOS_FLUXES['B0_synthetic']),axis=-1))


# Calculating B-V color

# In[ ]:


COSMOS_FLUXES['rest_frame_B-V']=COSMOS_FLUXES['B0_synthetic_mag']-COSMOS_FLUXES['V0_synthetic_mag']


# Calculating Absolute Magnitude using k-correction (Different cells below cover different redshift ranges)

# In[ ]:


COSMOS_FLUXES['Abs_B_Mag']=np.nan


# In[32]:


COSMOS_FLUXES.loc[COSMOS_FLUXES.Z_BEST<0.1,'Abs_B_Mag']=COSMOS_FLUXES.loc[COSMOS_FLUXES.Z_BEST<0.1,'mag_subaru_B']-0.05122-cosmo.distmod(COSMOS_FLUXES.loc[COSMOS_FLUXES.Z_BEST<0.1,'Z_BEST']).value-COSMOS_FLUXES.loc[COSMOS_FLUXES.Z_BEST<0.1,'subaru_B_kcorr_B']+0.09


# In[33]:


COSMOS_FLUXES.loc[(COSMOS_FLUXES.Z_BEST>0.1)&(COSMOS_FLUXES.Z_BEST<0.35),'Abs_B_Mag']=COSMOS_FLUXES.loc[(COSMOS_FLUXES.Z_BEST>0.1)&(COSMOS_FLUXES.Z_BEST<0.35),'mag_subaru_V']-0.069802-cosmo.distmod(COSMOS_FLUXES.loc[(COSMOS_FLUXES.Z_BEST>0.1)&(COSMOS_FLUXES.Z_BEST<0.35),'Z_BEST']).value-COSMOS_FLUXES.loc[(COSMOS_FLUXES.Z_BEST>0.1)&(COSMOS_FLUXES.Z_BEST<0.35),'subaru_V_kcorr_B']+0.09


# In[34]:


COSMOS_FLUXES.loc[(COSMOS_FLUXES.Z_BEST>0.35)&(COSMOS_FLUXES.Z_BEST<0.55),'Abs_B_Mag']=COSMOS_FLUXES.loc[(COSMOS_FLUXES.Z_BEST>0.35)&(COSMOS_FLUXES.Z_BEST<0.55),'mag_subaru_r']-0.01267-cosmo.distmod(COSMOS_FLUXES.loc[(COSMOS_FLUXES.Z_BEST>0.35)&(COSMOS_FLUXES.Z_BEST<0.55),'Z_BEST']).value-COSMOS_FLUXES.loc[(COSMOS_FLUXES.Z_BEST>0.35)&(COSMOS_FLUXES.Z_BEST<0.55),'subaru_r_kcorr_B']+0.09


# In[35]:


COSMOS_FLUXES.loc[(COSMOS_FLUXES.Z_BEST>0.55)&(COSMOS_FLUXES.Z_BEST<0.75),'Abs_B_Mag']=COSMOS_FLUXES.loc[(COSMOS_FLUXES.Z_BEST>0.55)&(COSMOS_FLUXES.Z_BEST<0.75),'mag_subaru_i']-0.004512-cosmo.distmod(COSMOS_FLUXES.loc[(COSMOS_FLUXES.Z_BEST>0.55)&(COSMOS_FLUXES.Z_BEST<0.75),'Z_BEST']).value-COSMOS_FLUXES.loc[(COSMOS_FLUXES.Z_BEST>0.55)&(COSMOS_FLUXES.Z_BEST<0.75),'subaru_i_kcorr_B']+0.09


# In[36]:


COSMOS_FLUXES.loc[COSMOS_FLUXES.Z_BEST>0.75,'Abs_B_Mag']=COSMOS_FLUXES.loc[COSMOS_FLUXES.Z_BEST>0.75,'mag_subaru_z']-0.00177-cosmo.distmod(COSMOS_FLUXES.loc[COSMOS_FLUXES.Z_BEST>0.75,'Z_BEST']).value-COSMOS_FLUXES.loc[COSMOS_FLUXES.Z_BEST>0.75,'subaru_z_kcorr_B']+0.09


# Calculate absolute magnitude from the "Synthetic" apparent magnitude

# In[37]:


COSMOS_FLUXES['Abs_B_Mag_synthetic']=COSMOS_FLUXES['B0_synthetic_mag']-cosmo.distmod(COSMOS_FLUXES.Z_BEST.values).value


# Diagnostic plotting

# In[48]:


plt.plot(COSMOS_FLUXES.loc[COSMOS_FLUXES.Z_BEST<1.0,'Abs_B_Mag'],COSMOS_FLUXES.loc[COSMOS_FLUXES.Z_BEST<1.0,'Abs_B_Mag']-COSMOS_FLUXES.loc[COSMOS_FLUXES.Z_BEST<1.0,'Abs_B_Mag_synthetic'],'.')
plt.plot(COSMOS_FLUXES.loc[(COSMOS_FLUXES.Z_BEST<1.0)&(COSMOS_FLUXES.Z_USE<2)&(COSMOS_FLUXES.SG_MASTER==0),'Abs_B_Mag'],COSMOS_FLUXES.loc[(COSMOS_FLUXES.Z_BEST<1.0)&(COSMOS_FLUXES.Z_USE<2)&(COSMOS_FLUXES.SG_MASTER==0),'Abs_B_Mag']-COSMOS_FLUXES.loc[(COSMOS_FLUXES.Z_BEST<1.0)&(COSMOS_FLUXES.Z_USE<2)&(COSMOS_FLUXES.SG_MASTER==0),'Abs_B_Mag_synthetic'],'.')
plt.xlabel('Absolute Magnitude')
plt.ylabel('Abs Mag-Synthetic Mag')
print(len(COSMOS_FLUXES.loc[(COSMOS_FLUXES.Z_BEST>0.01)&(COSMOS_FLUXES.Z_BEST<1.0)&(COSMOS_FLUXES.Z_USE<2)&(COSMOS_FLUXES.SG_MASTER==0),'Abs_B_Mag']))
plt.hlines(0.5,-30,200)
plt.hlines(-0.5,-30,200)
plt.savefig('Absmag-synthmagVSabsmag.ps')


# In[ ]:


plt.plot(COSMOS_FLUXES.loc[COSMOS_FLUXES.Z_BEST<1.0,'Abs_B_Mag'],COSMOS_FLUXES.loc[COSMOS_FLUXES.Z_BEST<1.0,'Abs_B_Mag']-COSMOS_FLUXES.loc[COSMOS_FLUXES.Z_BEST<1.0,'Abs_B_Mag_synthetic'],'.')
plt.plot(COSMOS_FLUXES.loc[(COSMOS_FLUXES.Z_BEST<1.0)&(COSMOS_FLUXES.Z_USE<2)&(COSMOS_FLUXES.SG_MASTER==0),'Abs_B_Mag'],COSMOS_FLUXES.loc[(COSMOS_FLUXES.Z_BEST<1.0)&(COSMOS_FLUXES.Z_USE<2)&(COSMOS_FLUXES.SG_MASTER==0),'Abs_B_Mag']-COSMOS_FLUXES.loc[(COSMOS_FLUXES.Z_BEST<1.0)&(COSMOS_FLUXES.Z_USE<2)&(COSMOS_FLUXES.SG_MASTER==0),'Abs_B_Mag_synthetic'],'.')
plt.xlabel('Absolute Magnitude')
plt.ylabel('Abs Mag-Synthetic Mag')
print(len(COSMOS_FLUXES.loc[(COSMOS_FLUXES.Z_BEST>0.01)&(COSMOS_FLUXES.Z_BEST<1.0)&(COSMOS_FLUXES.Z_USE<2)&(COSMOS_FLUXES.SG_MASTER==0),'Abs_B_Mag']))
plt.hlines(0.5,-30,200)
plt.hlines(-0.5,-30,200)
plt.xlim(-30,-15)
plt.savefig('Absmag-synthmagVSabsmag_zoom.ps')


# In[ ]:


for f in kcorr_filt_list:
    plt.plot(COSMOS_FLUXES.loc[(COSMOS_FLUXES.Z_BEST<1.0)&(COSMOS_FLUXES.Z_USE<3)&(COSMOS_FLUXES.SG_MASTER==0),f],COSMOS_FLUXES.loc[(COSMOS_FLUXES.Z_BEST<1.0)&(COSMOS_FLUXES.Z_USE<3)&(COSMOS_FLUXES.SG_MASTER==0),f]-COSMOS_FLUXES.loc[(COSMOS_FLUXES.Z_BEST<1.0)&(COSMOS_FLUXES.Z_USE<3)&(COSMOS_FLUXES.SG_MASTER==0),f+'_synthetic'],'.')
    plt.xlabel('Apparent Magnitude')
    plt.ylabel('Apparent Magnitude-Synthetic Apparent Magniture')
    plt.title(f)
    plt.savefig(f+'synth_vs_meas.ps')


# In[41]:


plt.plot(COSMOS_FLUXES.loc[(COSMOS_FLUXES.Z_BEST<1.0)&(COSMOS_FLUXES.Z_USE<3)&(COSMOS_FLUXES.SG_MASTER==0),'Z_BEST'],COSMOS_FLUXES.loc[(COSMOS_FLUXES.Z_BEST<1.0)&(COSMOS_FLUXES.Z_USE<3)&(COSMOS_FLUXES.SG_MASTER==0),'Abs_B_Mag'],',')


# Calculate surface brightness

# Calculate effective radius and surface brightness determine whether each source is an LCBG

# In[59]:


for r_col in ['r_half_cassata','r_half_tasca','r_half_zurich']:
    COSMOS_FLUXES[r_col+'_B']=COSMOS_FLUXES[r_col]*(814/(445*(1+COSMOS_FLUXES.Z_BEST.values)))**0.108
    COSMOS_FLUXES[r_col.split('_')[-1]+'_Surface_Brightness_B']=COSMOS_FLUXES.Abs_B_Mag+2.5*np.log10((2*np.pi*np.power(cosmo.angular_diameter_distance(COSMOS_FLUXES.Z_BEST.values).value*np.tan(COSMOS_FLUXES[r_col+'_B'].values*4.84814e-6),2)))+2.5*np.log10((360*60*60/(2*np.pi*0.01))**2)
    COSMOS_FLUXES[r_col.split('_')[-1]+'_is_LCBG']=0
    COSMOS_FLUXES.loc[(COSMOS_FLUXES['Abs_B_Mag'].values<-18.5)&(COSMOS_FLUXES[r_col.split('_')[-1]+'Surface_Brightness_B'].values<21)&(COSMOS_FLUXES['rest_frame_B-V'].values<0.6),r_col.split('_')[-1]+'is_LCBG']=1
    #COSMOS_FLUXES['R_eff_arcsec_F814W']=COSMOS_FLUXES.loc[:,'Rh']*0.03
#COSMOS_FLUXES['R_eff_arcsec_B']=COSMOS_FLUXES.loc[:,'Rh']*0.03*(814/(445*(1+COSMOS_FLUXES.Z_BEST.values)))**0.108
#COSMOS_FLUXES['is_LCBG']=0
#COSMOS_FLUXES.loc[(COSMOS_FLUXES.Abs_B_Mag.values<-18.5)&(COSMOS_FLUXES.Surface_Brightness_B.values<21)&(COSMOS_FLUXES['rest_frame_B-V'].values<0.6),'is_LCBG']=1


# In[61]:


COSMOS_FLUXES.to_csv('/Users/lucashunt/ASTRODATA/LCBG_LUMINOSITY_FUNCTION/COSMOS_CATALOGS/Photometry/COSMOS_CONVERTED_CATALOG_OTHER_RADII.csv')

