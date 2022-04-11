'''
date: 03/03/2022
author: pedroticiani@usp.br
'''

import os
import subprocess, sys
from os import listdir 
from os.path import isfile, join, exists, isdir
import numpy as np
import pandas as _pd
from scipy.signal import detrend
import matplotlib.pyplot as plt
import matplotlib.pylab as pl 
from PIL import Image
from PIL import PngImagePlugin
from astropy.stats import median_absolute_deviation as MAD
from astropy.convolution import convolve, Box1DKernel
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates.sky_coordinate import SkyCoord
from astroquery.simbad import Simbad
import pyhdust.spectools as spt
import xmltodict as _xmltodict
import requests as _requests
import wget as _wget
from datetime import datetime
from lmfit.models import GaussianModel, LinearModel
import logging
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("requests").setLevel(logging.WARNING)

___email___ = 'pedroticiani@usp.br'

def automatic_BeSS(RA, DEC, path, size='0.2', date_lower='1000-01-01', date_upper="3000-01-01", band_lower='6.4e-7', band_upper='6.8e-7'):
	"""This is a script for downloading BeSS spectra, directly from the database website, 
	using VO Table and pandas dataframes
	
	Parameters
	----------
	RA : str
		Right ascension [° J200] as string
	DEC : str
		Declination [° J2000] as string
	path: str
		specific directory to be downloaded
	size: str
		Radius of the cone search in degree as string
	date_lower: str
		Initial date in format yyyy-mm-dd as string
	date_upper: str
		Final date in format yyyy-mm-dd as string
	band lower: str
		Initial wavelength [meters] in scientific notation as string
	band_upper: str
		Final wavelength [meters] in scientific notation as string
	
	Returns
	-------
	None, the routine downloads file in the script directory.
	
	Example
	-------
	#Halpha for 25 Cyg from 2019-10-01 to 2020-03-27
	>>> RA = "299.979"
	>>> DEC = "37.04"
	>>> date_lower = "2019-10-01"
	>>> date_upper = "2020-03-27"
	>>> automatic_BeSS(RA, DEC, size='0.1', date_lower, date_upper, band_lower='6.4e-7', band_upper='6.7e-7')
	#Data downloaded in the script directory
	
	-------
	#Download all Ha data of a star
	>>> automatic_BeSS(RA="299.979", DEC="37.04")
	
	IMPORTANT NOTE: When using this function, the downloaded files go to the script 
	directory. This is something still undergoing work.
	"""
	
	user_url = 'http://basebe.obspm.fr/cgi-bin/ssapBE.pl?POS={0},{1}&SIZE={2}&BAND={3}/{4}&TIME={5}/{6}'.format(RA, DEC, size, band_lower, band_upper, date_lower, date_upper)
	
	r = _requests.get(url = user_url)

	# xml parsed => dict
	global_dict = _xmltodict.parse(r.text)
	
	# Interesting data selection
	try:
		entries_list = global_dict['VOTABLE']['RESOURCE']['TABLE']['DATA']['TABLEDATA']['TR']
	
	# Dataframe init (eq. Table)
		df01 = _pd.DataFrame()	
	# Browse through the entries and record it in the dataframe df01
		for item in entries_list:  
		# Create a row for the dataframe
			p01 = {'Fits URL': item['TD'][0],
			'Target name': item['TD'][45],
			"Target class": item['TD'][46],
			"vo_format": item['TD'][1]}
	# add row in progress in the dataframe
			df01 = df01.append(p01, ignore_index=True)
	# Dataframe init
		df02 = _pd.DataFrame()
		# Iteration on each row
		for item in entries_list:
			vo_url_fits = item['TD'][0]	
		# Create a row for the dataframe
		# with VO Table value + Header infos
			p02 = {'Fits URL': item['TD'][0],
			   'Target name': item['TD'][45],
			   "Target class": item['TD'][46]}
			   #"Resolution" : header_fits_ip['SPEC_RES'],
			   #"Creation Date" : header_fits_ip['DATE']}
	
		# add row in progress in the dataframe
			df02 = df02.append(p02, ignore_index=True)
			try:
				_wget.download(vo_url_fits, out=path)
			except:
				print('um FITS não conseguiu ser baixado!')
	except:
		print('Star could not get resolved from this method.')

def EW_calculation(param, vl, fx, vw=500):
	"""Calculates equivalent width using sum integral.
	Useful for line profiles. Useful for AA or km/s
	
	param: unit of x-axis of 1D spectrum. 
		u = pixel, wavelength, velocity
	fx: normalized flux of a region of 1D spectrum
		u = no units, as it is relative flux
	vw: estimative of the beginning of continuum
		u = units of velocity (usually km/s)
	"""
	vels = vl
	idx = np.where(np.abs(vels) <= vw)
	outvels = param[idx]
	normflux = fx[idx]
	EW = 0.
	if len(outvels) < 3:
		# normflux = _np.ones(len(outvels))
		EW = EW
	for i in range(len(outvels) - 1):
		dl = outvels[i + 1] - outvels[i]
		# print(dl)
		EW += (1. - (normflux[i + 1] + normflux[i]) / 2.) * dl
	
	return EW

def find_mainIDs(TIC):
	"""This function queries the star TIC into Simbad, delivering other
	known names of the object. This can lead to a big amount of IDs
	"""
	
	result = Simbad.query_objectids(TIC)
	secondID = result[1][0]
	
	return secondID

def starcoord(star_ID):
	"""This function prints the important info on the star position in RA, DEC
	star_ID must be a string.
	"""
	
	star_coord = SkyCoord.from_name(star_ID)
	return star_coord

def writeH5(arq, lista, name):
	'''arq = name of .h5 file.
		type: str
	lista = list you want to store
		type: list
	name = ID of the column storing the list
		type: str
	'''
	
	arq_name = f'{arq}.h5'
	# write
	store = _pd.HDFStore(arq_name)
	lista_para_salvar = _pd.Series(lista)
	store[f'{name}'] = lista_para_salvar
	store.close()
	
	return

def readH5(arq, name):
	'''arq = name of .h5 file.
		type: str
	name = ID of the column storing the list
		type: str
	'''
	
	arq_name = f'{arq}.h5'
	#read
	store = _pd.HDFStore(arq_name, mode='r')
	lista = store[f'{name}']
	store.close()
	
	return
	
def addMetadata(arq, dictionary):
	'''arq = name of the image (image in PNG)
		type: str
	dictionary = dict containing the data
		type: dict
	'''
		
	f = arq
	metad = dictionary
	
	im = Image.open(f)
	meta = PngImagePlugin.PngInfo()
	
	for x in metad:
		meta.add_text(x, metad[x])
	im.save(f, "png", pnginfo=meta)
	
	return
	
def fit_gauss(x,y,ngauss=1,neg=False,g1_cen=None,g2_cen=None,g3_cen=None,neg_cen=None,
			 g1_sig=None,g2_sig=None,g3_sig=None,neg_sig=None):
	'''FUNCTION EXTRACTED FROM STAR_MELT. 
	Parameters
	----------
	x : array or list
		wave.
	y : array or list
		flux.
	ngauss : int, optional
		number of positie Gaussians to fit. The default is 1.
	neg : bool, optional
		Whether to include a negative Gaussian. The default is False.
	g1_cen : list, optional
		list of min and max. The default is None.
	g2_cen : list, optional
		list of min and max. The default is None.
	g3_cen : list, optional
		list of min and max. The default is None.
	neg_cen : list, optional
		list of min and max. The default is None.
	g1_sig : list, optional
		list of min and max. The default is None.
	g2_sig : list, optional
		list of min and max. The default is None.
	g3_sig : list, optional
		list of min and max. The default is None.
	neg_sig : list, optional
		list of min and max. The default is None.
	Returns
	-------
	out : lmfit model
		lmfit model results.
	'''
	
	gauss1 = GaussianModel(prefix='g1_')
	gauss2 = GaussianModel(prefix='g2_')
	gauss3 = GaussianModel(prefix='g3_')
	gauss4 = GaussianModel(prefix='g4_')
	line1=LinearModel(prefix='line_')
	
	pars_g1 = gauss1.guess(y, x=x)
	pars_line = line1.guess(y, x=x)
	pars_g2 = gauss2.guess(y, x=x)
	pars_g3 = gauss3.guess(y, x=x)
	pars_g4 = gauss4.guess(y, x=x ,negative=True)
	
	if ngauss==1:
		mod = gauss1 + line1
		pars=pars_g1 + pars_line
		pars['g1_amplitude'].set(min=0)
		#pars['g1_sigma'].set(max=100)

	elif ngauss==2:
		mod = gauss1 + gauss2 + line1
		pars=pars_g1 + pars_g2 + pars_line
		pars['g1_amplitude'].set(min=0)
		pars['g2_amplitude'].set(min=0)
	
	elif ngauss==3:
		mod = gauss1 + gauss2 + gauss3 + line1
		pars=pars_g1 + pars_g2 + pars_g3 +pars_line
		pars['g1_amplitude'].set(min=0)
		pars['g2_amplitude'].set(min=0)
		pars['g3_amplitude'].set(min=0)
	
	if neg==True:
		mod += gauss4
		pars += pars_g4
		pars['g4_amplitude'].set(max=0)
	
	if g1_cen != None:
		pars['g1_center'].set(value=(g1_cen[0]+g1_cen[1])/2, min=g1_cen[0], max=g1_cen[1])
	if g2_cen != None and ngauss==2:
		pars['g2_center'].set(value=(g2_cen[0]+g2_cen[1])/2, min=g2_cen[0], max=g2_cen[1])
	if g3_cen != None and ngauss==3:
		pars['g3_center'].set(value=(g3_cen[0]+g3_cen[1])/2, min=g3_cen[0], max=g3_cen[1])
	if neg_cen != None and neg==True:
		pars['g4_center'].set(value=(neg_cen[0]+neg_cen[1])/2, min=neg_cen[0], max=neg_cen[1])

	if g1_sig != None:
		pars['g1_sigma'].set(value=(g1_sig[0]+g1_sig[1])/2, min=g1_sig[0], max=g1_sig[1])
	if g2_sig != None and ngauss==2:
		pars['g2_sigma'].set(value=(g2_sig[0]+g2_sig[1])/2, min=g2_sig[0], max=g2_sig[1])
	if g3_sig != None and ngauss==3:
		pars['g3_sigma'].set(value=(g3_sig[0]+g3_sig[1])/2, min=g3_sig[0], max=g3_sig[1])
	if neg_sig != None and neg==True:
		pars['g4_sigma'].set(value=(neg_sig[0]+neg_sig[1])/2, min=neg_sig[0], max=neg_sig[1])
	
	out = mod.fit(y, pars, x=x, weights = 1/np.std(y))    #use weights to obtain red. chi sq
		
	return out
 
def Sliding_Outlier_Removal(x, y, window_size, sigma=3.0, iterate=1):
	'''TODO docstring
	'''
	# remove NANs from the data
	x = x[~np.isnan(y)]
	y = y[~np.isnan(y)]

	#make sure that the arrays are in order according to the x-axis 
	y = y[np.argsort(x)]
	x = x[np.argsort(x)]

	# tells you the difference between the last and first x-value
	x_span = x.max() - x.min()  
	i = 0
	x_final = x
	y_final = y
	while i < iterate:
		i+=1
		x = x_final
		y = y_final
	
	# empty arrays that I will append not-clipped data points to
	x_good_ = np.array([])
	y_good_ = np.array([])
	
	# Creates an array with all_entries = True. index where you want to remove outliers are set to False
	tf_ar = np.full((len(x),), True, dtype=bool)
	ar_of_index_of_bad_pts = np.array([]) #not used anymore
	
	#this is how many days (or rather, whatever units x is in) to slide the window center when finding the outliers
	slide_by = window_size / 5.0 
	
	#calculates the total number of windows that will be evaluated
	Nbins = int((int(x.max()+1) - int(x.min()))/slide_by)
	
	for j in range(Nbins+1):
		#find the minimum time in this bin, and the maximum time in this bin
		x_bin_min = x.min()+j*(slide_by)-0.5*window_size
		x_bin_max = x.min()+j*(slide_by)+0.5*window_size
		
		# gives you just the data points in the window
		x_in_window = x[(x>x_bin_min) & (x<x_bin_max)]
		y_in_window = y[(x>x_bin_min) & (x<x_bin_max)]
		
		# if there are less than 5 points in the window, do not try to remove outliers.
		if len(y_in_window) > 5:            
			# Removes a linear trend from the y-data that is in the window.
			y_detrended = detrend(y_in_window, type='linear')
			y_in_window = y_detrended
			#print(np.median(m_in_window_))
			y_med = np.median(y_in_window)          
			# finds the Median Absolute Deviation of the y-pts in the window
			y_MAD = MAD(y_in_window)         
			#This mask returns the not-clipped data points. 
			# Maybe it is better to only keep track of the data points that should be clipped...
			mask_a = (y_in_window < y_med+y_MAD*sigma) & (y_in_window > y_med-y_MAD*sigma)
			#print(str(np.sum(mask_a)) + '   :   ' + str(len(m_in_window)))
			y_good = y_in_window[mask_a]
			x_good = x_in_window[mask_a]
			
			y_bad = y_in_window[~mask_a]
			x_bad = x_in_window[~mask_a]          
			#keep track of the index --IN THE ORIGINAL FULL DATA ARRAY-- of pts to be clipped out
			try:
				clipped_index = np.where([x == z for z in x_bad])[1]
				tf_ar[clipped_index] = False
				ar_of_index_of_bad_pts = np.concatenate([ar_of_index_of_bad_pts, clipped_index])
			except IndexError:
				#print('no data between {0} - {1}'.format(x_in_window.min(), x_in_window.max()))
				pass
	ar_of_index_of_bad_pts = np.unique(ar_of_index_of_bad_pts)
	#print('step {0}: remove {1} points'.format(i, len(ar_of_index_of_bad_pts)))
	x_final = x[tf_ar]
	y_final = y[tf_ar]
	
	return(x_final, y_final)
		
def plot_spectra(all_spec_fnames, Main_dir, fig_out_dir_01, fig_out_dir_02, star_ID, Emission_only=False, CB=False, scale='linear', Hb=False):
	'''routine for plotting spectra as usual
	   TODO: docstring complete
	'''

	Usefull_all_list = []
	Usefull_all_dict = {}
	HJD_all = []
	if Hb:
		Usefull_lines = [6562.81, 4861.363] 
	else:
		Usefull_lines = [6562.81]
		
	Ha_HJD_list = [] # stores the HJDs of all the Halpha observations
	Ha_HJD_foryear_list = []
	Hb_HJD_list = []
	Hb_HJD_foryear_list = []
	
	for jj, fname in enumerate(all_spec_fnames):
		try:
			hdulist = fits.open(Main_dir + fname)
		except:
			print('Um FITS provavelmente está corrompido;', fname)
		obs_instrument = hdulist[0].header['BSS_INST']
		
		# In the next few lines we remove the spectrum from Ultraviolet instruments for technical
		# reasons (IUE)	
		if obs_instrument not in ['IUE_SWP_HS', 'IUE_LWR_HS']:
			ref_pixel = hdulist[0].header['CRVAL1']
			coord_increment = hdulist[0].header['CDELT1']
			obs_date = hdulist[0].header['DATE-OBS']               
			obs_HJD = float(hdulist[0].header['MID-HJD']) - 2450000
			obs_HJD_2 = float(hdulist[0].header['MID-HJD'])
				
		# Reading FITS as VOTables
		scidata = hdulist[1].data
		scidata_wave = scidata['WAVE']        
		scidata_flux = scidata['FLUX']
			
		# here we can do wavelength calib using ZP + increments, defined at header
		wls = np.zeros(np.shape(scidata_wave),)
		wls[0] = ref_pixel
		for i,value in enumerate(wls[:-1]):
			wls[i+1] = wls[i] + coord_increment
			
		for iiii,xxxx in enumerate(Usefull_lines):
			if ( (xxxx < wls.max()) & (xxxx > wls.min())):
				try:
					Usefull_all_dict[str(xxxx)].append([wls, scidata_flux, obs_HJD])
					HJD_all.append(obs_HJD)
					if str(xxxx) == '6562.81':
						Ha_HJD_list.append(obs_HJD)
					elif str(xxxx) == '4861.363':
						Hb_HJD_list.append(obs_HJD)
				except KeyError:
					Usefull_all_dict[str(xxxx)] = [ [wls, scidata_flux, obs_HJD] ]
					HJD_all.append(obs_HJD)
					if str(xxxx) == '6562.81':
						Ha_HJD_list.append(obs_HJD)
						Ha_HJD_foryear_list.append(obs_HJD_2)
					elif str(xxxx) == '4861.363':
						Hb_HJD_list.append(obs_HJD)		
						Hb_HJD_foryear_list.append(obs_HJD_2)	
								
	for line in (Usefull_all_dict):			
		ax_LP = plt.subplot2grid((3,1), (0,0),rowspan=2)
		ax_EW = plt.subplot2grid((3,1), (2,0))
			
		Ha_EW_list = []		
		Hb_EW_list = []
		EW_list = []
		HJD_list = []	
		
		disk_state = []
		strong_emission_state = []
			
		for i, x in enumerate(Usefull_all_dict[line]):
			wl, flx = x[0], x[1]  # wl here is initially in Angstroms
			HJD = x[2]
				
			# normalization of the spectrum, and transformation of x-axis to velocity 
			# 1500 is a good value for line width up until the continuum level; some stars
			# may have a bigger width, such as Achernar (which we tend to use 2500+)
			vl, fx = spt.lineProf(wl, convolve(flx, Box1DKernel(2)), lbc=float(line), hwidth=1500)
			
			# sigma clip!
			# arguments are (wavelength, flux, window-size, sigma-clip, number_of_iterations)
			vl, fx = Sliding_Outlier_Removal(vl, fx, 50, 5, 7)
			
			# EW calculation in Angstroms. If km/s is wished, change wl to vl below.
			EW = EW_calculation(wl, vl, fx, vw=500)
			if line == '6562.81':
				Ha_EW_list.append(EW)
			elif line == '4861.363':
				Hb_EW_list.append(EW)
				
			# defining color of the spectra; if there is only one, black;
			# if there are more than 1, we pl.jet according to the obs date.
			# if CB (Color Blind) flag is True, a colorblind friendly colormap is used (viridis)	
			if len(Usefull_all_dict[line]) == 1:
				color = 'k'
			else:
				if CB:
					color = pl.cm.viridis((HJD - np.array(HJD_all).min())/(np.array(HJD_all).max() - np.array(HJD_all).min()))
				else:
					color = pl.cm.jet((HJD - np.array(HJD_all).min())/(np.array(HJD_all).max() - np.array(HJD_all).min()))
				
			if Emission_only:
				if EW < 0:
					ax_EW.scatter(HJD, EW, color=color, edgecolor='k',zorder=15, s=70, lw=2)
					ax_LP.plot(vl, fx, label='{0:.1f}'.format(EW), color=color)
				else:
					pass
			else:			
				ax_EW.scatter(HJD, EW, color=color, edgecolor='k',zorder=15, s=70, lw=2)			
				Ha_HJD_foryear_list_s = np.array(Ha_HJD_foryear_list)[np.argsort(Ha_HJD_foryear_list)]
				Hb_HJD_foryear_list_s = np.array(Hb_HJD_foryear_list)[np.argsort(Hb_HJD_foryear_list)]

				# plotting in the top panel all spectra 
				ax_LP.plot(vl, fx, label='{0:.1f}'.format(EW), color=color)
				
				if line == '6562.81':
					line_name = 'H-alpha'
					try:
						writeH5(f'{star_ID}_{line_name}', fx, f'flx_{i}')
					except:
						pass
					
		## If EW < 0: there is a disk; If EW < -5, there may be a strong emission			
		## if EW > 0: multiple things can happen. if EW < 5 and its neighbors fill this condition and/or are NEGATIVE, 
		## prob emissionn else: absorption. this can still include weak residual emission...			
		for i in range(len(Ha_EW_list)):	
			if (Ha_EW_list[i] < 0) and (Ha_EW_list[i] > -5):
				disk_active = True
				strong_emission = False
			elif Ha_EW_list[i] < -5:
				disk_active = True
				strong_emission = True
			elif Ha_EW_list[i] > 0:
				if (Ha_EW_list[i] <= 2):
					if i == 0:
						if (Ha_EW_list[i+1] < 1.):
							disk_active = True
							strong_emission = False
					elif i>=1:
						try:
							if (Ha_EW_list[i+1] < 1.) and (Ha_EW_list[i-1] < 1.):
								disk_active = True
								strong_emission = False
						except:
							pass
				elif (Ha_EW_list[i] > 2):
					disk_active = False
					strong_emission = False
			disk_state.append(disk_active)
			strong_emission_state.append(strong_emission)
		
		writeH5(f'{star_ID}_H-alpha', vl, f'vl')
		writeH5(f'{star_ID}_H-alpha', Ha_HJD_list, f'observation_HJD')
		writeH5(f'{star_ID}_H-alpha', disk_state, f'disk_state')
		writeH5(f'{star_ID}_H-alpha', strong_emission_state, f'strong_emission')
	
		if line == '6562.81':
			line_name = "H-alpha"
			ax_LP.text(0.01,0.9, r'H$\alpha$' + '\n{0} spectra'.format(len(Usefull_all_dict[line])), color='k',\
					fontsize=18, transform=ax_LP.transAxes, bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))
				
			Ha_EW_list_s = np.array(Ha_EW_list)[np.argsort(np.array(Ha_HJD_list))]
			Ha_HJD_list_s = np.array(Ha_HJD_list)[np.argsort(Ha_HJD_list)]

			# calculate some statistics based on EW values
			Ha_EW_min = Ha_EW_list_s.min()
			Ha_EW_max = Ha_EW_list_s.max()
			Ha_mean = np.mean(Ha_EW_list_s)
			Ha_std = np.std(Ha_EW_list_s)
			Ha_range = Ha_EW_min - Ha_EW_max
				
			#Add this text to the plot
			EW_text = 'EW_min = {0:.0f}\nEW_max = {1:.0f}\nEW_avg = {2:.0f}\nEW_std = {3:.0f}\nEW_range = {4:.0f}'.format(Ha_EW_min, Ha_EW_max,\
					Ha_mean, Ha_std, Ha_range)
			ax_LP.text(0.01, 0.7, EW_text, color='k', fontsize=13, transform=ax_LP.transAxes,\
					bbox=dict(facecolor='white',alpha=0.9,edgecolor='none'))				

			if len(Usefull_all_dict[line]) > 1:
				Ha_HJD_range = Ha_HJD_list_s.max() - Ha_HJD_list_s.min()
				Ha_EW_range = Ha_EW_list_s.max() - Ha_EW_list_s.min()
					
				ax_EW.set_ylim(Ha_EW_list_s.max() + Ha_EW_range*0.1, Ha_EW_list_s.min() - Ha_EW_range*0.1)
	
				# this creates the top legend at EW panel with the dates in years
				new_time_array = Time(Ha_HJD_list_s + 2450000, format='jd')
				new_t_years = new_time_array.byear
	   
				ax_EW2 = ax_EW.twiny()
				ax_EW2.xaxis.tick_top()
				ax_EW2.xaxis.set_label_position('top')
				ax_EW2.set_xticks(np.arange(int(new_t_years.min()) - 1, int(new_t_years.max() + 1), 2.0))
				ax_EW2.xaxis.get_ticklocs(minor=True)
				ax_EW2.minorticks_on()
	   
				orig_xlimits = ax_EW.get_xlim()
				orig_xlim_time_jd = Time(np.array(orig_xlimits) + 2450000, format='jd')
				orig_xlim_time_yr = orig_xlim_time_jd.byear
				ax_EW2.set_xlim(orig_xlim_time_yr)
				
				ax_EW.plot(Ha_HJD_list_s, Ha_EW_list_s, linestyle='dotted', color='k')
							
			elif len(Usefull_all_dict[line]) == 1:
				ax_EW.set_ylim(Ha_EW_list_s.max() + 0.2*Ha_EW_list_s.max(), Ha_EW_list_s.max() - 0.2*Ha_EW_list_s.max())
		
		elif line == '4861.363':
			line_name = "H-beta"
			ax_LP.text(0.01,0.9, r'H$\beta$' + '\n{0} spectra'.format(len(Usefull_all_dict[line])), color='k',\
					fontsize=18, transform=ax_LP.transAxes, bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))
				
			Hb_EW_list_s = np.array(Hb_EW_list)[np.argsort(np.array(Hb_HJD_list))]
			Hb_HJD_list_s = np.array(Hb_HJD_list)[np.argsort(Hb_HJD_list)]

			# calculate some statistics based on EW values
			Hb_EW_min = Hb_EW_list_s.min()
			Hb_EW_max = Hb_EW_list_s.max()
			Hb_mean = np.mean(Hb_EW_list_s)
			Hb_std = np.std(Hb_EW_list_s)
			Hb_range = Hb_EW_min - Hb_EW_max
				
			#Add this text to the plot
			EW_text = 'EW_min = {0:.0f}\nEW_max = {1:.0f}\nEW_avg = {2:.0f}\nEW_std = {3:.0f}\nEW_range = {4:.0f}'.format(Hb_EW_min, Hb_EW_max,\
					Hb_mean, Hb_std, Hb_range)
			ax_LP.text(0.01, 0.7, EW_text, color='k', fontsize=13, transform=ax_LP.transAxes,\
					bbox=dict(facecolor='white',alpha=0.9,edgecolor='none'))				

			if len(Usefull_all_dict[line]) > 1:
				Hb_HJD_range = Hb_HJD_list_s.max() - Hb_HJD_list_s.min()
				Hb_EW_range = Hb_EW_list_s.max() - Hb_EW_list_s.min()
					
				ax_EW.set_ylim(Hb_EW_list_s.max() + Hb_EW_range*0.1, Hb_EW_list_s.min() - Hb_EW_range*0.1)
	
				# this creates the top legend at EW panel with the dates in years
				new_time_array = Time(Hb_HJD_list_s + 2450000, format='jd')
				new_t_years = new_time_array.byear
	   
				ax_EW2 = ax_EW.twiny()
				ax_EW2.xaxis.tick_top()
				ax_EW2.xaxis.set_label_position('top')
				ax_EW2.set_xticks(np.arange(int(new_t_years.min()) - 1, int(new_t_years.max() + 1), 2.0))
				ax_EW2.xaxis.get_ticklocs(minor=True)
				ax_EW2.minorticks_on()
	   
				orig_xlimits = ax_EW.get_xlim()
				orig_xlim_time_jd = Time(np.array(orig_xlimits) + 2450000, format='jd')
				orig_xlim_time_yr = orig_xlim_time_jd.byear
				ax_EW2.set_xlim(orig_xlim_time_yr)
				
				ax_EW.plot(Hb_HJD_list_s, Hb_EW_list_s, linestyle='dotted', color='k')
							
			elif len(Usefull_all_dict[line]) == 1:
				ax_EW.set_ylim(Hb_EW_list_s.max() + 0.2*Hb_EW_list_s.max(), Hb_EW_list_s.max() - 0.2*Hb_EW_list_s.max())
							
		ax_LP.set_xlabel('Velocity (km/s)')
		ax_LP.set_ylabel('Relative Flux')
		
		if scale == 'linear':
			ax_LP.set_yscale('linear')
			name = 'linear'
		else:
			ax_LP.set_yscale('log')
			name = 'log'
		
		LP_y_limits = ax_LP.get_ylim()
		if LP_y_limits[0] < 0.0:
			ax_LP.set_ylim([0.0, LP_y_limits[1]])
		
		if Emission_only == False:
			ax_EW.axhline(y=0, color='b', linestyle='--')
		ax_EW.set_xlabel('HJD - 2450000')
		ax_EW.set_ylabel(r'EW ($\AA$)')
				
		ax_LP.set_title('{0}'.format(star_ID))
		ax_LP.grid(color='gray', linestyle='--', linewidth=0.35)
			
		plt.gcf().set_size_inches(15,10)
		plt.tight_layout()
		if CB:
			if Emission_only:
				plt.savefig('{0}/{1}_{2}_emission_{3}_CB.png'.format(fig_out_dir_01, star_ID, line_name, name), dpi=400)
			else:
				plt.savefig('{0}/{1}_{2}_{3}_CB.png'.format(fig_out_dir_01, star_ID, line_name, name), dpi=400)
		else:
			if Emission_only:
				plt.savefig('{0}/{1}_{2}_emission_{3}.png'.format(fig_out_dir_01, star_ID, line_name, name), dpi=400)
			else:
				plt.savefig('{0}/{1}_{2}_{3}.png'.format(fig_out_dir_01, star_ID, line_name, name), dpi=400)		
		print('\nPlot saved for star {0}\n'.format(star_ID))
		
		# adding metadata to the image		
		dictionary = {"date": str(datetime.today()), "star_ID": f"{star_ID}", "EW_min": f"{Ha_EW_min}",\
					"EW_max": f"{Ha_EW_max}", "EW_mean": f"{Ha_mean}", "EW_range": f"{Ha_range}"}
		addMetadata('{0}/{1}_{2}_{3}.png'.format(fig_out_dir_01, star_ID, line_name, name), dictionary)
		
	return
	
	
################################### MAIN ###################################################

# Set equal to True if you want to plot every single full spectrum in a single
# plot. Can be useful if you want to inspect different parts of the spectrum 
# at the same time.

Plot_all_spectra = True

## Sets size of axis labels
params = {'legend.fontsize': 10,
		 'axes.labelsize': 14,
		 'axes.titlesize':16,
		 'xtick.labelsize':12,
		 'ytick.labelsize':12,
		 'xtick.direction':'in',
		 'ytick.direction':'in',
		 'axes.linewidth': 1.5,
		 'font.family': 'DeJavu Sans',
		 'font.serif': ['Montserrat'],
		 'font.size':11}
pl.rcParams.update(params)
#plt.style.use(['science', 'ieee', 'no-latex'])

fig_out_dir_01 = 'Figs/' ## Sets the directory for all plots
	
# set a csv table containing the star's TIC
master_tbl = np.genfromtxt('catalog.csv', delimiter=',', dtype=str) 
	
Plot_all_with_spectra = False
Download = False
scale = ['linear', 'log']
	
if Plot_all_with_spectra:
	all_spec_dirs = [f for f in listdir('Spectra/')] 
	star_ID_list = all_spec_dirs # 'star_ID_list' will be the stars common names or IDs you want

else:
	# If you set Plot_all_with_spectra = False, then here you say which name/ID you want
	star_ID_list = ['59_Cyg']

for star_ID in star_ID_list:
	# Main_dir is where all of the BeSS spectra .fits files are
	Main_dir = 'Spectra/{0}/'.format(star_ID)
	fig_out_dir_02 = 'Spectra/{0}/'.format(star_ID)

	# checking coordinates using the TIC 
	star_coord = starcoord(star_ID)
	RA = str(star_coord.ra.degree) 
	DEC = str(star_coord.dec.degree)	
		
	# downloading
	if Download:
		automatic_BeSS(RA, DEC, Main_dir, size='0.2', date_lower='1000-01-01', date_upper="3000-01-01", band_lower='6.4e-7', band_upper='6.8e-7') 

	# some stars cant be resolved using TIC; query for RA and DEC fails.
	#print(f'O download de todos espectros para {star_ID} concluiu!')

	all_spec_fnames = sorted([ f for f in listdir(Main_dir) if isfile(join(Main_dir,f)) ])
	
	for scale in scale:
		# normal plotting
		plot_spectra(all_spec_fnames, Main_dir, fig_out_dir_01, fig_out_dir_01, star_ID, Emission_only=False, scale=scale)
		# plot spectra for emission only profiles, to highlight Ha when disk was active.	
		plot_spectra(all_spec_fnames, Main_dir, fig_out_dir_01, fig_out_dir_01, star_ID, Emission_only=True, scale=scale)
	
		# ColorBlind
		#plot_spectra(all_spec_fnames, Main_dir, fig_out_dir_01, fig_out_dir_01, star_ID, Emission_only=False, CB=True, scale=scale)
		# plot spectra for emission only profiles, to highlight Ha when disk was active.	
		#plot_spectra(all_spec_fnames, Main_dir, fig_out_dir_01, fig_out_dir_01, star_ID, Emission_only=True, CB=True, scale=scale)
	
########################################################################################################################	
	
	#listaState = []
	#for i in disk_state:
	#	if i == True:
	#		listaState.append('True')
	#	else:
	#		listaState.append('False')
		
	#disk_state = listaState
		
	# preciso salvar os estados do disco com base na data! é possível, e bem melhor.
	#np.savetxt(f'state_{star_ID}.txt', np.c_[str(Ha_HJD_list), disk_state, str(strong_emission_state)])

		#im2 = Image.open('{0}/{1}_{2}.png'.format(fig_out_dir_01, star_ID, line_name))
		#print(im2.info)
