import pandas as pd
import numpy as np
import astropy as ap
from astropy import units as u
from astropy.coordinates import SkyCoord
import kcorrect
import kcorrect.utils as ut
from astropy.cosmology import FlatLambdaCDM

#*********************************************
#*** Defining Cosmology
#*********************************************

cosmo=FlatLambdaCDM(H0=70,Om0=0.3)

#*********************************************
#*** Reading in catalogs (LAMBDAR and TASCA)
#*********************************************

print('Reading in Catalogs')
COSMOS_PHOT_LAMBDAR=pd.read_csv('/home/lrhunt/CATALOGS/PHOT/G10CosmosLAMBDARCatv05.csv')
TASCA_COSMOS_MORPH=pd.read_csv('/home/lrhunt/CATALOGS/PHOT/TASCA_MORPH.tsv',delim_whitespace=True,header=53,dtype=float,error_bad_lines=False)

#*********************************************
#*********************************************
#*** Matching Catalogs
#*********************************************
#*********************************************

print('Matching Catalogs')

#*********************************************
#*** Generating astropy skycoord objects to match catalogs
#*********************************************

TASCA_COORD=SkyCoord(ra=TASCA_COSMOS_MORPH['RAJ2000'].values*u.degree,dec=TASCA_COSMOS_MORPH['DEJ2000'].values*u.degree)
G10_COORD=SkyCoord(ra=COSMOS_PHOT_LAMBDAR['RA'].values*u.degree,dec=COSMOS_PHOT_LAMBDAR['DEC'].values*u.degree)

#*********************************************
#*** Matching Catalogs
#*********************************************

idx,d2d,d3d=G10_COORD.match_to_catalog_sky(TASCA_COORD)
COSMOS_PHOT_LAMBDAR['Rh']=TASCA_COSMOS_MORPH['Rh'][idx].values
COSMOS_PHOT_LAMBDAR['separation']=d2d.arcsecond
COSMOS_FLUXES=COSMOS_PHOT_LAMBDAR[COSMOS_PHOT_LAMBDAR.separation<1]

#*********************************************
#*********************************************
#*** Calculating apparent magnitudes
#*********************************************
#*********************************************

print('Calculating apparent magnitudes')

#*********************************************
#*** Determining what data is in each column
#*********************************************

column_headers=COSMOS_FLUXES.columns.values
telescope_flux_indeces=[i for i, s in enumerate(column_headers) if (('subaru' in s) or ('uvista' in s) or ('galex' in s) or ('cfht' in s) or ('irac' in s) or ('mips' in s) or ('pacs' in s) or ('spire' in s)) and ('err' not in s)]

#*********************************************
#*** calculating magnitudes
#*********************************************


for i in telescope_flux_indeces:
    COSMOS_FLUXES['mag_'+column_headers[i]]=-2.5*np.log10(COSMOS_FLUXES[column_headers[i]]/3631)
    COSMOS_FLUXES['mag_'+column_headers[i+1]]=2.5/np.log(10)*COSMOS_FLUXES[column_headers[i+1]]/COSMOS_FLUXES[column_headers[i]]

COSMOS_FLUXES.loc[COSMOS_FLUXES.isnull().mag_galex_fuv,'mag_galex_fuv_err']=np.nan
COSMOS_FLUXES.loc[COSMOS_FLUXES.isnull().mag_galex_nuv,'mag_galex_nuv_err']=np.nan

#*********************************************
#*********************************************
#*** Getting k-correction
#*********************************************
#*********************************************

#*********************************************
#*** converting to maggies (see kcorrect.org for explanation)
#*********************************************

print('Converting to maggies')

COSMOS_FLUXES['umaggies']=ut.mag2maggies(COSMOS_FLUXES.mag_cfht_u)
COSMOS_FLUXES['bmaggies']=ut.mag2maggies(COSMOS_FLUXES.mag_subaru_B)
COSMOS_FLUXES['vmaggies']=ut.mag2maggies(COSMOS_FLUXES.mag_subaru_V)
COSMOS_FLUXES['rmaggies']=ut.mag2maggies(COSMOS_FLUXES.mag_subaru_r)
COSMOS_FLUXES['imaggies']=ut.mag2maggies(COSMOS_FLUXES.mag_subaru_i)
COSMOS_FLUXES['zmaggies']=ut.mag2maggies(COSMOS_FLUXES.mag_subaru_z)

COSMOS_FLUXES['uinvervar']=ut.invariance(COSMOS_FLUXES.umaggies,COSMOS_FLUXES.mag_cfht_u_err)
COSMOS_FLUXES['binvervar']=ut.invariance(COSMOS_FLUXES.bmaggies,COSMOS_FLUXES.mag_subaru_B_err)
COSMOS_FLUXES['vinvervar']=ut.invariance(COSMOS_FLUXES.vmaggies,COSMOS_FLUXES.mag_subaru_V_err)
COSMOS_FLUXES['rinvervar']=ut.invariance(COSMOS_FLUXES.rmaggies,COSMOS_FLUXES.mag_subaru_r_err)
COSMOS_FLUXES['iinvervar']=ut.invariance(COSMOS_FLUXES.imaggies,COSMOS_FLUXES.mag_subaru_i_err)
COSMOS_FLUXES['zinvervar']=ut.invariance(COSMOS_FLUXES.zmaggies,COSMOS_FLUXES.mag_subaru_z_err)

allmaggies=np.stack((COSMOS_FLUXES.umaggies.values,COSMOS_FLUXES.bmaggies.values,COSMOS_FLUXES.vmaggies.values,COSMOS_FLUXES.rmaggies.values,COSMOS_FLUXES.imaggies.values,COSMOS_FLUXES.zmaggies.values),axis=-1)
allinvervar=np.stack((COSMOS_FLUXES.uinvervar.values,COSMOS_FLUXES.binvervar.values,COSMOS_FLUXES.vinvervar.values,COSMOS_FLUXES.rinvervar.values,COSMOS_FLUXES.iinvervar.values,COSMOS_FLUXES.zinvervar.values),axis=-1)

#*********************************************
#*** Generating array to store values for calculating k-correction
#*********************************************


carr=np.ndarray((len(COSMOS_FLUXES.bmaggies.values),6))
rmarr=np.ndarray((len(COSMOS_FLUXES.bmaggies.values),7))
rmarr0=np.ndarray((len(COSMOS_FLUXES.bmaggies.values),7))
rmarr0B=np.ndarray((len(COSMOS_FLUXES.bmaggies.values),7))
rmarr0V=np.ndarray((len(COSMOS_FLUXES.bmaggies.values),7))
rmarr0U=np.ndarray((len(COSMOS_FLUXES.bmaggies.values),7))

#*********************************************
#*********************************************
#*** Computing k-correction
#*********************************************
#*********************************************


print('Computing k-corrections and estimated magnitudes')

#*********************************************
#*** Loading filter list (total k-correction)
#*********************************************


kcorrect.load_templates()
kcorrect.load_filters('/home/lrhunt/programs/kcorrect/data/templates/Lum_Func_Filters_US.dat')

for i in range(0,len(carr)):
	carr[i]=kcorrect.fit_nonneg(COSMOS_FLUXES.Z_BEST.values[i],allmaggies[i],allinvervar[i])
for i in range(0,len(carr)):
	rmarr[i]=kcorrect.reconstruct_maggies(carr[i])
	rmarr0[i]=kcorrect.reconstruct_maggies(carr[i],redshift=0)

#*********************************************
#*** Loading filter list (apparent B mag for each object)
#*********************************************

kcorrect.load_templates()
kcorrect.load_filters('/home/lrhunt/programs/kcorrect/data/templates/BESSEL_B2.dat')

for i in range(0,len(carr)):
	rmarr0B[i]=kcorrect.reconstruct_maggies(carr[i],redshift=0)

#*********************************************
#*** Loading filter list (apparent V mag for each object)
#*********************************************

kcorrect.load_templates()
kcorrect.load_filters('/home/lrhunt/programs/kcorrect/data/templates/BESSEL_V2.dat')

for i in range(0,len(carr)):
	rmarr0V[i]=kcorrect.reconstruct_maggies(carr[i],redshift=0)

#*********************************************
#*** Loading filter list (apparent u mag for each object)
#*********************************************

kcorrect.load_templates()
kcorrect.load_filters('/home/lrhunt/programs/kcorrect/data/templates/BESSEL_U2.dat')

for i in range(0,len(carr)):
	rmarr0U[i]=kcorrect.reconstruct_maggies(carr[i],redshift=0)

#*********************************************
#*** Convert from corrected maggies back to k-correction
#*********************************************

kcorr=-2.5*np.log10(rmarr/rmarr0)
kcorrM=-2.5*np.log10(rmarr/rmarr0B)
corrB=-2.5*np.log10(rmarr0B)+0.09
corrV=-2.5*np.log10(rmarr0V)-0.02
corrU=-2.5*np.log10(rmarr0U)-0.79

#*********************************************
#*********************************************
#*** Computing absolute magnitude, color and surface brightness 
#*********************************************
#*********************************************

M=np.zeros_like(COSMOS_FLUXES.Z_BEST.values)
bv=corrB[:,3]-corrV[:,4]
#M=corrB[:,3]-cosmo.distmod(COSMOS_FLUXES.Z_BEST.values).value
for i in range(0,len(COSMOS_FLUXES.Z_BEST.values)):
	if COSMOS_FLUXES.Z_BEST.values[i]<=0.1:
		M[i]=COSMOS_FLUXES.mag_subaru_B.values[i]-0.05122-cosmo.distmod(COSMOS_FLUXES.Z_BEST.values[i]).value-kcorrM[i][2]
	if COSMOS_FLUXES.Z_BEST.values[i]<=0.35 and COSMOS_FLUXES.Z_BEST.values[i]>0.1:
		M[i]=COSMOS_FLUXES.mag_subaru_V.values[i]+0.069802-cosmo.distmod(COSMOS_FLUXES.Z_BEST.values[i]).value-kcorrM[i][3]
	if COSMOS_FLUXES.Z_BEST.values[i]<=0.55 and COSMOS_FLUXES.Z_BEST.values[i]>0.35:
		M[i]=COSMOS_FLUXES.mag_subaru_r.values[i]-0.01267-cosmo.distmod(COSMOS_FLUXES.Z_BEST.values[i]).value-kcorrM[i][4]
	if COSMOS_FLUXES.Z_BEST.values[i]<=0.75 and COSMOS_FLUXES.Z_BEST.values[i]>0.55:
		M[i]=COSMOS_FLUXES.mag_subaru_i.values[i]-0.004512-cosmo.distmod(COSMOS_FLUXES.Z_BEST.values[i]).value-kcorrM[i][5]
	if COSMOS_FLUXES.Z_BEST.values[i]>0.75:
		M[i]=COSMOS_FLUXES.mag_subaru_z.values[i]-0.00177-cosmo.distmod(COSMOS_FLUXES.Z_BEST.values[i]).value-kcorrM[i][6]

M=M+0.09

SBe=M+2.5*np.log10((2*np.pi*np.power(cosmo.angular_diameter_distance(COSMOS_FLUXES.Z_BEST.values).value*np.tan(COSMOS_FLUXES.Rh.values*0.03*4.84814e-6)*(814/(445*(1+COSMOS_FLUXES.Z_BEST.values)))**0.108*1e3,2)))+2.5*np.log10((360*60*60/(2*np.pi*0.01))**2)


#*********************************************
#*********************************************
#*** Adding columns to final catalog
#*********************************************
#*********************************************


COSMOS_FLUXES.loc[:,'corrected_B']=corrB[:,2]
COSMOS_FLUXES.loc[:,'corrected_V']=corrV[:,2]
COSMOS_FLUXES.loc[:,'corrected_u']=corrU[:,2]
COSMOS_FLUXES.loc[:,'k_subB-B']=kcorr[:,2]
COSMOS_FLUXES.loc[:,'k_subV-B']=kcorr[:,3]
COSMOS_FLUXES.loc[:,'k_subr-B']=kcorr[:,4]
COSMOS_FLUXES.loc[:,'k_subi-B']=kcorr[:,5]
COSMOS_FLUXES.loc[:,'k_subz-B']=kcorr[:,6]
COSMOS_FLUXES.loc[:,'rest_frame_B-V']=bv
COSMOS_FLUXES.loc[:,'Abs_B_Mag']=M
COSMOS_FLUXES.loc[:,'Surface_Brightness_B']=SBe
COSMOS_FLUXES['R_eff_arcsec_F814W']=COSMOS_FLUXES.loc[:,'Rh']*0.03
COSMOS_FLUXES['R_eff_arcsec_B']=COSMOS_FLUXES.loc[:,'Rh']*0.03*(814/(445*(1+COSMOS_FLUXES.Z_BEST.values)))**0.108
LCBG=np.zeros_like(COSMOS_FLUXES.Z_BEST.values)
wherelcbg=np.where((COSMOS_FLUXES.Abs_B_Mag.values<-18.5)&(COSMOS_FLUXES.Surface_Brightness_B.values<21)&(COSMOS_FLUXES['rest_frame_B-V'].values<0.6))[0]
LCBG[wherelcbg]=1
COSMOS_FLUXES['is_LCBG']=LCBG


COSMOS_FLUXES.to_csv('COSMOS_LAMBDAR_CATALOG.csv')




