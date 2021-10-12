import numpy as np
import math
import cmath
from pylab import *
import sys, os
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.signal
#import lomb
import matplotlib.ticker as plticker
from matplotlib.ticker import ScalarFormatter
import astropy.io.ascii
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import random
import pdb

# python g_massradius.py $dir $targetname
jupmass = 1.89813e27 #kg
juprad = 7.1492e7#Equatorial radius from IAU resolution B3
earthmass = 5.9721986e24
earthrad = 6.3781e6#Equatorial radius from IAU resolution B3
tsun = 5777

# My colours
violet = '#5A068F'
lightpurple = '#EEE4F5'
gris = '#C6BECC'
grisfonce = '#848285'
lavande = '#AF8CBA'
bordeaux = '#8C0726'
bleu = '#5465AB'
vert = '#00752B'

ft = 36
fs = 18
lt = 20
ls = 30
mm = 10

# Nasa exoplanet archive
workdir = "/Users/Zoe/Documents/SOAP_2/Other_Stars/hd212657/MR"
datafile = workdir+"/planets_2021.04.28_13.49.50.csv" ## replace this!
a = astropy.io.ascii.read(datafile)
bb = a['pl_bmassj']
n = bb.shape[0]

# Jessie's table
datafile =workdir+"/20170228_Jessie.dat"


#  Contour lines ZS2013
datafile2 =workdir+"/20170228_zengsasselov.csv"
steps, fe100, fe50, rocky, h2o50, h2o100, coldh2  = \
np.loadtxt(datafile2, unpack=True, usecols=(0,1,11,21,31,41,42), skiprows=2)
l = rocky.shape[0]

datafile3 =workdir+"/earthlike.csv"
modelmass, earthlike= np.loadtxt(datafile3, unpack=True, skiprows=1)



# Cuts, limits in mass for the flux plot
minm = 0.5
maxm = 20.
minr = 0.7
maxr = 4.
mcut1 = 3.
mcut2 = 7.
minre = -0.8
maxre = 0.7

mcut01 = 1.
mcut02 = 4
mcut03 = 7
mcut04 = 10
mcut05 = 13

r1 = 1.0364*mcut01**0.2764
r2 = 1.0364*mcut02**0.2764
r3 = 1.0364*mcut03**0.2764
r4 = 1.0364*mcut04**0.2764
r5 = 1.0364*mcut05**0.2764

# Incident flux in Earth units
# (Tstar/Tsun)**4. * (Rstar/dist7b)**2.
# semi major axis
# ((theta(7) /365.25)**2. * Mstar )**(1./3.)
# sma = ((orb /365.25)**2. * mstar )**(1./3.)
a['st_teff'].fill_value = 0.
a['st_rad'].fill_value = -99.
a['pl_orbsmax'].fill_value = -99.
a['pl_orbper'].fill_value = -99.
a['st_mass'].fill_value = -99.
teff = a['st_teff'].filled()
rstar = a['st_rad'].filled()
sma = a['pl_orbsmax'].filled()
orb = a['pl_orbper'].filled()
mstar = a['st_mass'].filled()
flux = (teff/tsun)**4. * (rstar/((orb /365.25)**2. * mstar )**(1./3.))**2.



# Planet names
a['pl_hostname'].fill_value = 0.
a['pl_letter'].fill_value = 0.
name = a['pl_hostname'].filled()
letter = a['pl_letter'].filled()

# Planet mass and radius (m, r)
a['pl_bmassj'].fill_value = 0.
a['pl_radj'].fill_value = 0.
a['pl_bmassjerr1'].fill_value = 0.
a['pl_bmassjerr2'].fill_value = 0.
a['pl_radjerr1'].fill_value = 0.
a['pl_radjerr2'].fill_value = 0.
m = a['pl_bmassj'].filled()*jupmass/earthmass
r = a['pl_radj'].filled()*juprad/earthrad
meh = a['pl_bmassjerr1'].filled()*jupmass/earthmass
mel = a['pl_bmassjerr2'].filled()*jupmass/earthmass
reh = a['pl_radjerr1'].filled()*juprad/earthrad
rel = a['pl_radjerr2'].filled()*juprad/earthrad




# ----------------------------------------------------------------
# Adding KOI-280b to m, r, meh/l, reh/l, flux, names, letter
# My KOI-280b:
###
#mk = 4.7
#rk = 2.213
#melk = 2.8
#mehk = 3.1
#relk = 0.082
#rehk = 0.082
#fk = 139.
#nk = 'KOI-280'
#lk = 'b'
#
#m = np.append (m, mk)
#r = np.append (r, rk)
#meh = np.append (meh, mehk)
#mel = np.append (mel, melk)
#reh = np.append (reh, rehk)
#rel = np.append (rel, relk)
#flux = np.append (flux, fk)
#name = np.append (name, nk)
#letter = np.append (letter, lk)
###


# ----------------------------------------------------------------
# ----------------------------------------------------------------
# Calculate the transparency
# Do a first pass to compute the normalisation factors.
alpha = []
gauss1 = []
gauss2 = []
gauss3 = []
gauss4 = []
gauss5 = []
rrock = []
rexcess = []
proba_rocky = []
ptot = 1000 # 10000 made no difference


#  Then add KOI-280
# alpha = np.append(alpha, 1./(sqrt(mehk*mehk + melk*melk + rehk*rehk + relk*relk)))
# Calculate normalisation coefs


good = np.where((mel != 0) & (meh !=0) & (rel != 0)  & (reh != 0) & (m < maxm) & (m > minm) & (r < maxr) & (r > minr))

mel = mel[good]
meh = meh[good]
rel = rel[good]
reh = reh[good]
m = m[good]
r = r[good]
name = name[good]
letter = letter[good]

alphas = 1. /(abs(meh)/m + abs(mel)/m + abs(reh)/r + abs(rel)/r)
massprecisions = 0.5*(abs(meh) + abs(mel))/m
radiusprecisions = 0.5*(abs(reh) + abs(rel))/r
precisions = np.sqrt(massprecisions**2 + radiusprecisions**2 * 9.0)

alphamax = np.max(alphas)


rc('axes', linewidth=4)
# ----------------------------------------------------------------
# Make the mass radius plot
do_mrplot = 1
if do_mrplot == 1:
    bestm = []
    bestr = []
    bestf = []
    bestn = []
    bestl = []
    

    figure2 = figure(figsize=(20,12))
    ax1 = plt.subplot(111)
    ax1.set_xscale("log", nonposx='clip')
    ax1.set_xticks([1,2,3,4,5,6,7,8,9,10,20])
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    #ax1.set_yticks([1.0,1.5,2.0,2.5, 3,3.5,4])
    ax1.set_yticks([1,2,3,4])
    ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    xy = mcut1,0.5

    ax1.minorticks_on()
    ax1.tick_params('both', length=20, width=2, which='major', direction='in', right =1, top = 1)
    ax1.tick_params('both', length=10, width=1, which='minor', direction='in', right =1, top = 1)	

    width, height = mcut2-mcut1,5
    #p = patches.Rectangle(xy, width, height, facecolor="#C6BECC", edgecolor='#C6BECC', alpha=0.35)
    #plt.gca().add_patch(p)

    for i in range(0,len(m)-1):
        if  m[i] < maxm and r[i] < maxr:
        # and mel[i] / m[i] < 0.2 and meh[i] / m[i] < 0.2:

            # Check that we have the error bars
            if mel[i] == 0. or meh[i] == 0. \
            or rel[i] == 0. or reh[i] == 0.:
                # do not include in plot!
                beta = 1.
            else:
                # Transparency of the points proportional to size of error bars:
                #beta = 1. /(abs(meh[i])/m[i] + abs(mel[i])/m[i] + abs(reh[i])/r[i] + abs(rel[i])/r[i])**2
                #beta = beta/alphamax
		thisalpha = alphas[i]/alphamax
		thisalpha = 1. / (precisions[i]/np.max(precisions))
		#print precisions[i]
		#if thisalpha
		print str(alphas[i]/alphamax) + name[i] + ' '+ letter[i] + ' mass=' + str(m[i]) +', radius='+str(r[i])
		thiscolorin = 255 - int(np.floor(255*(alphas[i]/alphamax)))
		#print thiscolorin
		thislevel = hex(thiscolorin).split('x')[1]
		if len(thislevel) == 1:
			thislevel = '0'+thislevel
		thiscolor = '#'+thislevel+thislevel+thislevel
		#print thiscolor
		
		thiszorder = 3/precisions[i]
		if precisions[i] < 0.2: thisalpha = 1
		if precisions[i] >= 0.2: 
			thisalpha = thisalpha**2
			thiszorder = 1

                # if beta > 0.65:
                #     plt.text(1.05*m[i], 1.01*r[i], name[i]+letter[i], fontsize=fs)
                # Plot!
                err_mass = [m[i]+mel[i], m[i]+meh[i]]
                err_rad = [r[i]+rel[i], r[i]+reh[i]]
		#pdb.set_trace()
                #plt.errorbar([m[i],m[i]], [r[i],r[i]], xerr=[[mel[i], meh[i]],[mel[i], meh[i]]], yerr=[[rel[i],reh[i]],[rel[i],reh[i]]], \
                plt.errorbar(m[i], r[i], xerr=np.abs(np.reshape([mel[i], meh[i]],[2,1])), yerr=np.abs(np.reshape([rel[i],reh[i]],[2,1])), \
                fmt = 'o', ms = mm, color=thiscolor, ecolor=thiscolor, capsize=6, elinewidth=4, \
                alpha = thisalpha, markeredgecolor = 'none', capthick = 4,zorder = thiszorder)
		#if precisions[i] <= 0.2: plt.text(1.05*m[i], 1.01*r[i], name[i]+letter[i], fontsize=fs)
			

    waspcolor = '#49007a'#'#6800AD'#'#db2323'
    plt.errorbar(13.1, 3.576,xerr=1.5, yerr=.046, fmt = 'o', ms = mm, color=waspcolor, markeredgecolor = 'none',ecolor=waspcolor, capsize=6,\
    capthick = 4, elinewidth=4, zorder = 100)	
    plt.text(11.5, 3.4, 'WASP-47 d', size=22, weight = 'semibold', color = waspcolor, zorder = 3)
    plt.errorbar(6.83, 1.81,xerr=.66, yerr=.027, fmt = 'o', ms = mm, color=waspcolor, markeredgecolor = 'none',ecolor=waspcolor, capsize=6,\
    capthick = 4, elinewidth=4, zorder = 100)
    plt.text(4.8, 1.87, 'WASP-47 e', size=22, weight = 'semibold', color = waspcolor, zorder = 3)
    #neptune: 3.8538, 17.148
    #uranus: 3.9658, 14.5357
    plt.errorbar( 17.148, 3.8538,xerr=0, yerr=0, fmt = 'o', ms = mm*1.5, color=bleu, markeredgecolor = 'none', zorder = 3)	
    plt.text(13, 3.7, 'Neptune', size=22, weight = 'semibold', color = bleu, zorder = 100)
    plt.errorbar(14.5357, 3.9658,xerr=0, yerr=0, fmt = 'o', ms = mm*1.5, color=bleu, markeredgecolor = 'none', zorder = 3)	
    plt.text(11.7, 3.81, 'Uranus', size=22, weight = 'semibold', color = bleu, zorder = 100)

    plt.errorbar(1, 1,xerr=0, yerr=0, fmt = 'o', ms = mm*1.5, color=bleu, markeredgecolor = 'none', zorder = 3)	
    plt.text(1.02, 0.9, 'Earth', size=22, weight = 'semibold', color = bleu, zorder = 3)
    plt.errorbar(.815, 0.9488,xerr=0, yerr=0, fmt = 'o', ms = mm*1.5, color=bleu, markeredgecolor = 'none', zorder = 3)	
    plt.text(0.815 * 1.02, 0.84, 'Venus', size=22, weight = 'semibold', color = bleu, zorder = 3)
    # fe100, fe50, rocky, h2o50, h2o100, coldh2
    kk = 3
    ax1.plot(steps, fe100, ls = '-', color='#FF781F', lw = kk*2, zorder = 2)
    plt.text(1.55, 0.9, '100% Fe', size=22, weight = 'semibold', color = '#FF781F', zorder = 3, rotation = 8)

    #ax1.plot(steps, fe50, '--', color='#FF781F', lw = kk*2, zorder = 2)
    ax1.plot(steps, rocky, color='#006A7A', lw = kk*2, zorder = 2)
    plt.text(1.3, 1.38, r'100% MgSiO$_3$', size=22, weight = 'semibold', color = '#006A7A', zorder = 3, rotation = 13)
    #ax1.plot(steps, h2o50, '--', color='#006A7A', lw = kk*2, zorder = 2)
    ax1.plot(steps, h2o100, color='#042FBD', lw = kk*2, zorder = 2)
    plt.text(1.3, 1.73, r'100% H$_2$O', size=22, weight = 'semibold', color = '#042FBD', zorder = 3, rotation = 17)

    ax1.plot(modelmass, earthlike, color='#5B7312', lw = kk*2, zorder = 2)
    plt.text(1.45, 1.11, 'Earth-like', size=22, weight = 'semibold', color = '#5B7312', zorder = 3, rotation = 10)

    ax1.plot(steps, coldh2, color='#553A8C', lw = kk*2, zorder = 2)
    # scatter(bestm, bestr, c=bestf, cmap=plt.cm.rainbow, edgecolor='none', s=150)
    plt.text(.9, 3.08, r'Cold H$_2$', size=22, weight = 'semibold', color = '#553A8C', zorder = 3, rotation = 33)

    plt.text(6.8, 2.04, '55 Cnc e', size=22, weight = 'semibold', color = '#000000', zorder = 4)
    plt.tight_layout(rect=(.03, .07, .98, .98))	

    # figlegend(bestf,('Flux'),'upper left')
    plt.tick_params(labelsize=ls)
    xlabel('Mass (Earth Masses)', fontsize=ft)
    ylabel('Radius (Earth Radii)', fontsize=ft)
    loc = plticker.MultipleLocator(base=.1)
    plt.gca().axes.yaxis.set_minor_locator(loc)
    xlim([minm,maxm])
    ylim([minr,maxr])
    # tight_layout()
    # subplots_adjust(hspace=.45, wspace=.45)
    figure2.savefig("/Users/Zoe/Documents/SOAP_2/Other_Stars/hd212657/MR/massradius.pdf")
    print('Ready!')



