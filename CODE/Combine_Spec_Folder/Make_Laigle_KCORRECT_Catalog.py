
# coding: utf-8

# Taking the LAIGLE spectroscopic catalog to make a full catalog with selected LCBGs

# | Date | Person | Change |
# | :- | :- | :--------: |
# 04/24/2019  |  L. Hunt  |  <ul><li>Initial Version</li><li>Import CSV</li><li>Use kcorrect to correct get absolut magnitudes</li></ul>
# 04/26/2019 | L. Hunt | <ul><li>Runs faster on setesh</li><li>Once get kcorrections will be faster to work with</li><li>KCORRECT should work with correct filter list (Added new filters to kcorrect dir)</li></ul>

# Import number and plotting modules

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Import astronomy modules

# In[2]:


import astropy as ap
from astropy import units as u
import kcorrect
import kcorrect.utils as ut
from astropy.cosmology import FlatLambdaCDM


# Import basic modules

# In[3]:


from itertools import combinations
import os
import datetime


# In[4]:


def make_kcorr_filt_template(dataframe):
    '''This task will make a kcorrect filter template from a dataframe that optimizes the number of objects with detections in a subset of filters. In this case the dataframe should contain cfht, subaru, and irac wideband filters. '''
    kcordir=os.environ["KCORRECT_DIR"]
    lambdar_to_kcorr={'u_MAG_AUTO':'capak_cfht_megaprime_sagem_u.par',
                      'B_MAG_AUTO':'capak_subaru_suprimecam_B.par',
                      'V_MAG_AUTO':'capak_subaru_suprimecam_V.par',
                      'r_MAG_AUTO':'capak_subaru_suprimecam_r.par',
                      'ip_MAG_AUTO':'capak_subaru_suprimecam_i.par',
                      'zpp_MAG_AUTO':'subaru_suprimecam_zpp.par',
                      'Y_MAG_AUTO':'vircam_Y.par',
                      'J_MAG_AUTO':'vircam_J.par',
                      'H_MAG_AUTO':'vircam_H.par',
                      'Ks_MAG_AUTO':'vircam_K.par'}
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
        if len(dataframe[(dataframe[x[0]]<40)&
                         (dataframe[x[1]]<40)&
                         (dataframe[x[2]]<40)&
                         (dataframe[x[3]]<40)&
                         (dataframe[x[4]]<40)]) > numb1:
            ilist5=ilist4
            ilist4=ilist3
            ilist3=ilist2
            ilist2=ilist1
            ilist1=dataframe[(dataframe[x[0]]<40)&
                             (dataframe[x[1]]<40)&
                             (dataframe[x[2]]<40)&
                             (dataframe[x[3]]<40)&
                             (dataframe[x[4]]<40)].index.tolist()
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
            file.write('data/filters/cosmos_filters/'+lambdar_to_kcorr[filt]+'\n')
    return flist1,kcor_template


# In[5]:


cosmo=FlatLambdaCDM(H0=70,Om0=0.3)


# In[6]:


kcordir=os.environ["KCORRECT_DIR"]
catbasedir=os.environ["COSMOS_DIR"].split('Original')[0]+'/Final_Catalogs'


# In[7]:


Spec_Cat=pd.read_csv(catbasedir+'/final_spec_catalog.csv')


# In[8]:


magnitude_columns=[filt for filt in Spec_Cat.columns.values if '_mag_' in filt.lower()]
magnitude_error_columns=[filt for filt in Spec_Cat.columns.values if 'err' in filt.lower()]


# In[9]:


kcorr_filt_list,kcor_template_filter=make_kcorr_filt_template(Spec_Cat[magnitude_columns])


# In[10]:


print(kcor_template_filter)


# In[11]:


maggies_filt_list=[filt.split('_')[0]+
                   '_maggies' for filt in kcorr_filt_list]

invervar_filt_list=[filt.split('_')[0]+
                    '_invervar' for filt in kcorr_filt_list]

synthetic_maggies_filt_list=[filt.split('_')[0]+
                             '_synthetic_maggies' for filt in 
                             kcorr_filt_list]

rf_synthetic_maggies_filt_list=[filt.split('_')[0]+
                             '0_synthetic_maggies' for filt in 
                             kcorr_filt_list]


# In[12]:


numbers=np.arange(1,7,1)
for string in ['c'+str(number) for number in numbers]:
    Spec_Cat[string]=np.nan


# In[13]:


print('Converting to Maggies')
for column in magnitude_columns:
    Spec_Cat[column.split('_')[0]
             +'_maggies']=ut.mag2maggies(Spec_Cat[column])
    Spec_Cat[column.split('_')[0]
             +'_invervar'
            ]=ut.invariance(Spec_Cat[column.split('_')[0]
                                     +'_maggies'],
                            Spec_Cat[column.split('_')[0]
                                     +'_MAGERR_AUTO']
                           )


# In[14]:


for i in range(0,len(synthetic_maggies_filt_list)):
    Spec_Cat[synthetic_maggies_filt_list[i]]=np.nan
    Spec_Cat[rf_synthetic_maggies_filt_list[i]]=np.nan


# In[15]:


kcorrect.load_templates()
kcorrect.load_filters(kcor_template_filter)


# In[16]:


indexes=Spec_Cat.index.values


# In[17]:


time=datetime.datetime.now()
for i in indexes:
    Spec_Cat.loc[i,'c1':'c6']=kcorrect.fit_nonneg(np.array(Spec_Cat.loc[i,'final_z'],
                                                           dtype=float),
                                                  np.array(Spec_Cat.loc[i,maggies_filt_list],
                                                           dtype=float),
                                                  np.array(Spec_Cat.loc[i,invervar_filt_list],
                                                           dtype=float))
    Spec_Cat.loc[i,synthetic_maggies_filt_list]=kcorrect.reconstruct_maggies(Spec_Cat.loc[i,'c1':'c6'])[1:]
    Spec_Cat.loc[i,rf_synthetic_maggies_filt_list]=kcorrect.reconstruct_maggies(Spec_Cat.loc[i,'c1':'c6'],redshift=0)[1:]
    if i%10000==0:
        print('k-correction for {} sources done'.format(i))
        print(datetime.datetime.now()-time)
        time=datetime.datetime.now()


# In[18]:


Spec_Cat['UJ0_synthetic_maggies']=np.nan
Spec_Cat['BJ0_synthetic_maggies']=np.nan
Spec_Cat['VJ0_synthetic_maggies']=np.nan


# In[19]:


kcorrect.load_templates()
kcorrect.load_filters(kcordir+'/data/templates/bessell_ubv.dat')


# In[20]:


for i in indexes:
    Spec_Cat.loc[i,
                 ['UJ0_synthetic_maggies',
                 'BJ0_synthetic_maggies',
                  'Vj0_synthetic_maggies']
                ]=kcorrect.reconstruct_maggies(Spec_Cat.loc[i,
                                                            'c1':'c6']
                                                                     ,redshift=0)[1:]
    if i%10000==0:
        print('k-correction for {} sources done'.format(i))
        print(datetime.datetime.now()-time)
        time=datetime.datetime.now()


# In[21]:


Spec_Cat[['UJ0_synthetic_AB_mag',
          'BJ0_synthetic_AB_mag',
          'VJ0_synthetic_AB_mag']]=-2.5*np.log10(
    Spec_Cat[['UJ0_synthetic_maggies',
              'BJ0_synthetic_maggies',
              'VJ0_synthetic_maggies']])
Spec_Cat['UJ0_synthetic_vega_mag']=Spec_Cat['UJ0_synthetic_AB_mag']-0.79
Spec_Cat['BJ0_synthetic_vega_mag']=Spec_Cat['BJ0_synthetic_AB_mag']+0.09
Spec_Cat['VJ0_synthetic_vega_mag']=Spec_Cat['VJ0_synthetic_AB_mag']-0.02


# In[33]:


[f.split('_')[0]+'_kcorr_BJ0' 
          for f in 
          synthetic_maggies_filt_list]


# In[30]:


Spec_Cat[[f.split('_')[0]+'_kcorr_BJ0' 
          for f in 
          synthetic_maggies_filt_list]]=-2.5*np.log10(
    Spec_Cat[synthetic_maggies_filt_list]/
    np.stack((Spec_Cat['BJ0_synthetic_maggies'],
              Spec_Cat['BJ0_synthetic_maggies'],
              Spec_Cat['BJ0_synthetic_maggies'],
              Spec_Cat['BJ0_synthetic_maggies'],
              Spec_Cat['BJ0_synthetic_maggies']),axies=-1))


# In[ ]:


Spec_Cat.to_csv(catbasedir+'/Laigle_Cat_With_kcorrections.csv',index=False)

