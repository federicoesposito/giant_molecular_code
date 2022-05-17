'''
TO DO IMMEDIATELY:
- interpolate between hden when logG0 fail (sww-edge problem)

WHEN EXPORTING THIS MODULE:
- remove ProgressBar
- edit pngm, pcat, ptur, pext

NEXT STEPS:
- write gmc_module.py description
- class Grid with its functions

'''

# import libraries
import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns
from astropy.modeling.models import Sersic1D
from scipy.stats import lognorm
from scipy.interpolate import interp1d
from progressbar import ProgressBar
from joblib import Parallel, delayed
import multiprocessing
import random
from functools import reduce


# plots options
sns.set_palette('spring')
sns.set(rc={'figure.dpi':130, 'savefig.dpi':130})
sns.set_style('ticks')
sns.set_style({'xtick.direction': 'in', 'ytick.direction': 'in'})
sns.set_context('paper', rc={'font.size':14, 'axes.titlesize':16, 
    'axes.labelsize':16, 'xtick.labelsize':14, 'ytick.labelsize':14,
    'xtick.minor.size': 0, 'ytick.minor.size': 0})

# useful global variables
pi = np.pi
pc = 3.086e18 # cm
kpc = 3.086e21 # cm
msun = 1.99e33 # grams
mp = 1.6726e-24 # grams
mu = 1.22 # mean molecular weight (2.2 for molecular gas?)
gamma = 5/3 # adiabatic index (7/5 for molecular gas?)
b_turb = 0.3 # turbulence parameter
kb = 1.38e-16 # erg/K
G = 6.67e-8 # cm^3/g/s^2
lsun = 3.9e33 # erg/s
c_cms = 2.99792458e10 # cm/s

# arrays
hden = np.arange(0, 6.75, 0.25)        # cm^-3
logG0 = np.arange(0, 6.2, 0.25)[::-1]  # 6-13.6 eV
logFX = np.arange(-1, 4.2, 0.25)[::-1] # 1-100 keV
hnames = ['h' + ('%03d' % (h*1e2)).replace('.', '') for h in hden]
gnames = ['g' + ('%.2f' % g).replace('.', '') for g in logG0]
xnames = ['x' + ('%.2f' % x).replace('.', '') for x in logFX]
radii = 10**np.arange(-3, 2.2, 0.005) # kpc

# paths and dataframes
unibo_cluster = False
if unibo_cluster:
    pbase = '/scratch/extra/francesca.pozzi4/esposito/'
    pngm = pbase + 'claudia/ngmgrid/'
    ptur = pbase + 'claudia/turbogrid'
    pcat = pbase + 'datasets/'
else:
    pngm = '/media/phd/cloudy/ngmgrid/'
    ptur = '/media/phd/cloudy/turbogrid/'
    pcat = '/media/phd/jupyter/AGN_impact_CO_sample/datasets/'
emili = pd.read_csv(pngm + 'emis_lines.csv', index_col='Name')
df = pd.read_csv(pcat + 'cat_2022c.csv', index_col='Name')
sers = pd.read_csv(pcat + 'sersic_fit.csv', index_col='Name')

# color lists
CBlist1 = ['#ff7f00', '#984ea3', '#a65628', '#dede00', 
           '#f781bf', '#999999', '#984ea3']
CBlist2 = ['#332288', '#44aa99', '#117733', '#999933', '#88ccee',
           '#ddcc77', '#cc6677', '#882255', '#aa4499', '#ddddd']



### GENERAL FUNCTIONS

def expo(x):
    aa = ('%.1e' % x).replace('+', '')
    bb = '%.f' % (float(aa.split('e')[0]) * 10)
    cc = '%.f' % (float(aa.split('e')[1]) - 1)
    return bb + 'e' + cc

def fmt_clump(x):
    if x < 0: return '{:+04d}'.format(x)
    else: return '{:03d}'.format(x)

def Mmol_r(r, Mmol_tot, rCO):
    Mmol_r = Mmol_tot * (1 - (np.exp(-r/rCO) * (r/rCO + 1)))
    return Mmol_r

def r_G0(Ie, Re, n):
    ''' [erg/s/cm2, kpc, n]
    Given the G0 Sersic fit results (Ie, Re, n) for a galaxy,
    it calculates the radial bins for each G0 value.
    The first returned array is the approximated r(G0),
    the other two are the radial bins bounds around G0
    '''
    logG0s = np.append(logG0 + 0.125, logG0[-1] - 0.125)
    sersic_f = Sersic1D(amplitude=Ie, r_eff=Re, n=n)
    G0_r = np.array([sersic_f(x) for x in radii]) / 1.6e-3
    idx = np.array([np.argmin(np.abs(
        np.log10(G0_r + np.nextafter(0,1)) - G0)) for G0 in logG0s])
    idG = np.array([np.argmin(np.abs(
        np.log10(G0_r + np.nextafter(0,1)) - G0)) for G0 in logG0])
    idx_0 = np.where(idx > 0)[0][0] - 1
    r_G0, r_min_G0, r_max_G0 = [], [], []
    for i in range(len(idx)-1):
        r_G0.append(radii[idG[i]])
        r_min_G0.append(radii[idx[i]])
        r_max_G0.append(radii[idx[i+1]])
    return np.array(r_G0), np.array(r_min_G0), np.array(r_max_G0)



### CLUMPS EXTRACTION FUNCTIONS

def jeans(n, T): # return Jeans radius in pc and mass in Msun
    rho = n * mu * mp
    cs = np.sqrt(gamma * kb * T / (mu * mp))
    RJ = 0.5 * cs * np.sqrt(pi / (G * rho))
    MJ = (4/3) * pi * rho * RJ**3 / msun
    return RJ/pc, MJ

def ndf_histo(ndf, save_path=None):
    fig, [(ax1, ax2), (ax3, ax4)] = plt.subplots(2, 2, 
        figsize=(5,5), sharey=True)
    _, bins, patches = ax1.hist(ndf['hden'])
    ax1.hist(ndf['hden'].iloc[-1], bins=bins)
    ax1.set_xlabel(r'Log (n$_H$ / cm$^{-3}$)')
    _, bins, patches = ax2.hist(ndf['logR'])
    ax2.hist(ndf['logR'].iloc[-1], bins=bins)
    ax2.set_xlabel('Log (R / pc)')
    _, bins, patches = ax3.hist(ndf['logM'])
    ax3.hist(ndf['logM'].iloc[-1], bins=bins)
    ax3.set_xlabel(r'Log (M / M$_{\odot}$)')
    _, bins, patches = ax4.hist(ndf['logNH'])
    ax4.hist(ndf['logNH'].iloc[-1], bins=bins)
    ax4.set_xlabel(r'Log (N$_H$ / cm$^{-2}$)')
    Mtot = (10**ndf['logM']).sum()
    fig.suptitle(r'M$_{GMC}$ = %.2E M$_{\odot}$' % Mtot +
        r', N$_{clumps}$ = ' + str(len(ndf)))
    fig.tight_layout()
    if save_path:
        if '.csv' in save_path:
            save_path = save_path.replace('.csv', '')
        plt.savefig(save_path + '_histo.png', 
            bbox_inches='tight', dpi=300)
    else: plt.show()
    plt.close()



### CLOUDY-READING FUNCTIONS

def error_finder(wd, error_substring):
    '''
    Find error_substring within .out Cloudy files within a wd
    It returns a list(set) of those files
    '''
    filelist, error_filelist = [], []
    for subdir, dirs, files in os.walk(wd):
        filelist.append(files)
    filelist = [item for sublist in filelist for item in sublist 
                if '.out' in item and 'slurm' not in item]
    print('Looking for', error_substring, 'in', len(filelist), 'files...')
    for filename in filelist:
        subdir = '_'.join(filename.split('_', 2)[:2]) + '/'
        with open(wd + subdir + filename, 'r') as f:
            lines = f.readlines()
            for l in lines:
                if error_substring in l:
                    error_filelist.append(filename)
    error_filelist = list(set(error_filelist))
    print('→ found it in', len(error_filelist), 'files!')
    return error_filelist

def closest(series, value):
    '''
    Returns abs_close = the closest value in the input Series
    then a list of the indexes of the 2 closest values (inf and sup)
    abs_close will be either series[inf_close] or series[sup_close]
    '''
    abs_close = series.iloc[series.sub(value).abs().idxmin()]
    inf_idx = series[series < value].idxmax()
    sup_idx = series[series > value].idxmin()
    inf_val, sup_val = series[[inf_idx, sup_idx]]
    return abs_close, [inf_idx, sup_idx], [inf_val, sup_val]

def emis(wd, filename):
    subdir = '_'.join(filename.split('_', 2)[:2]) + '/'
    emis = pd.read_csv(wd + subdir + filename + '.emis', sep='\t')
    return emis

def ovrpdr(wd, filename): # DataFrames of OVR and PDR files for a given Cloudy run
    subdir = '_'.join(filename.split('_', 2)[:2]) + '/'
    ovr = pd.read_csv(wd + subdir + filename + '.ovr', sep='\t')
    pdr = pd.read_csv(wd + subdir + filename + '.pdr', sep='\t')
    return ovr, pdr

def lineNH(wd, ff, line_label, logNH):
    subdir = '_'.join(ff.split('_', 2)[:2]) + '/'
    try:
        idx = ovrpdr(wd, ff)[1]['H colden'].sub(10**logNH).abs().idxmin()
        return emis(wd, ff)[line_label].iloc[idx]
    except TypeError:
        return np.nan

def grid_lineNH(wd, line_name, logNH, save=False):
    savename = wd + line_name + '_fluxNH%3d_' % (logNH*10)
    try:
        pdr = pd.read_csv(savename + 'PDR.csv', index_col=0)
        xdr = pd.read_csv(savename + 'XDR.csv', index_col=0)
    except FileNotFoundError:
        print('Files not found: calculating them now...')
        emili = pd.read_csv(wd + 'emis_lines.csv', index_col='Name')
        line_label = emili.loc[line_name, 'line']
        pdr = pd.DataFrame(index=hnames, columns=gnames[::-1])
        xdr = pd.DataFrame(index=hnames, columns=xnames[::-1])
        for h in hnames:
            for g in gnames:
                ff = 'pdr_' + h + '_' + g
                try: pdr.loc[h, g] = lineNH(wd, ff, line_label, logNH)
                except IndexError: pdr.loc[h, g] = np.nan
            for x in xnames:
                ff = 'xdr_' + h + '_' + x
                try: xdr.loc[h, x] = lineNH(wd, ff, line_label, logNH)
                except IndexError: xdr.loc[h, g] = np.nan
        if save:
            pdr.to_csv(savename + 'PDR.csv')
            xdr.to_csv(savename + 'XDR.csv')
    return pdr, xdr

def NHcosled(wd, filename, i, Jmax=30, norm=None): # CO SLED of a file at given index i
    colist = emis(wd, filename).columns.to_list()[1:31]
    Y = emis(wd, filename)[colist[:Jmax]].iloc[i].to_numpy()
    if norm: Y = Y / Y[norm-1]
    return Y

def interSLED(wd, fname, logNH, Jmax=13, norm=0): # interpolated CO SLED for a file given a logNH
    idx = closest(ovrpdr(wd, fname)[1]['H colden'], 10**logNH)[1]
    NHs = [ovrpdr(wd, fname)[1].loc[i, 'H colden'] for i in idx]
    SLEDs = np.array([NHcosled(wd, fname, i, Jmax, norm) for i in idx])
    fits = [interp1d(NHs, SLEDs[:, co], 'linear') for co in range(Jmax)]
    interSLED = np.array([fit(10**logNH) for fit in fits])
    return interSLED

def paramSLED(f1, f2, SLEDs, param, Jmax=13): # interpolate SLEDs from f1,f2 files given a param (hden or flux)
    if f1[4:8] == f2[4:8]: # flux case
        params = [float(f[10:])/1e2 for f in [f1, f2]]
    elif f1[-4:] == f2[-4:]: # density case
        params = [float(f[5:8])/1e2 for f in [f1, f2]]
    fits = [interp1d(params, SLEDs[:, co], 'linear') for co in range(Jmax)]
    paramSLED = np.array([fit(param) for fit in fits])
    return paramSLED

def sww_check(wd, fname, logNH, Jmax=13): # if sww, find existing [f1,f2] and interpolate SLED
    f1, f2 = fname, fname
    sww = pd.read_csv(wd + 'swwlist.csv', header=None)[0].to_list()
    sww = [s.replace('.out', '') for s in sww]
    if fname in sww:
        # find nearby fluxes
        near_check = [True, True]
        while any(near_check):
            if f1 in sww: f1 = f1[:10] + fmt_clump(int(f1[10:])-25)
            if f2 in sww: f2 = f2[:10] + fmt_clump(int(f2[10:])+25)
            near_check = [f in sww for f in [f1, f2]]
            # breaking conditions
            pdr1 = (fname[:3] == 'pdr') & (int(f1[10:]) < int(logG0[-1]*1e2))
            pdr2 = (fname[:3] == 'pdr') & (int(f2[10:]) > int(logG0[0]*1e2))
            xdr1 = (fname[:3] == 'xdr') & (int(f1[10:]) < int(logFX[-1]*1e2))
            xdr2 = (fname[:3] == 'xdr') & (int(f2[10:]) > int(logFX[0]*1e2))
            if pdr1 or pdr2 or xdr1 or xdr2:
                print('No more simulations available!', f1, f2)
                break
        SLEDs = np.array([interSLED(wd, f, logNH, Jmax=Jmax) for f in [f1, f2]])
        iSLED = paramSLED(f1, f2, SLEDs, float(fname[10:])/1e2, Jmax=Jmax)
    else:
        iSLED = interSLED(wd, fname, logNH, Jmax=Jmax)
    return f1, f2, iSLED

def rSLED(wd, fname, logR, Jmax=13, norm=0):
    idx, Rs = closest(emis(wd, fname)['#depth'], 10**logR)[1:]
    SLEDs = np.array([NHcosled(wd, fname, i, Jmax, norm) for i in idx])
    fits = [interp1d(Rs, SLEDs[:, co], 'linear') for co in range(Jmax)]
    rSLED = np.array([fit(10**logR) for fit in fits])
    return rSLED

def sww_new(wd, fname, logR, sww, Jmax=13):
    f1, f2 = fname, fname
    if fname in sww:
        # find nearby fluxes
        near_check = [True, True]
        while any(near_check):
            if f1 in sww: f1 = f1[:10] + fmt_clump(int(f1[10:])-25)
            if f2 in sww: f2 = f2[:10] + fmt_clump(int(f2[10:])+25)
            near_check = [f in sww for f in [f1, f2]]
        # breaking conditions
        pdr1 = (fname[:3] == 'pdr') & (int(f1[10:]) < logG0[-1]*1e2)
        pdr2 = (fname[:3] == 'pdr') & (int(f2[10:]) > logG0[0]*1e2)
        xdr1 = (fname[:3] == 'xdr') & (int(f1[10:]) < logFX[-1]*1e2)
        xdr2 = (fname[:3] == 'xdr') & (int(f2[10:]) > logFX[0]*1e2)
        if pdr1 or pdr2 or xdr1 or xdr2:
            iSLED = np.empty(Jmax)
            iSLED[:] = np.nan
        else:
            SLEDs = np.array([rSLED(wd, f, logR, Jmax) for f in [f1,f2]])
            iSLED = paramSLED(f1, f2, SLEDs, float(fname[10:])/1e2, Jmax=Jmax)
    else:
        iSLED = rSLED(wd, fname, logR, Jmax=Jmax)
    return f1, f2, iSLED

def sww_NH(wd, fname, logNH, sww, Jmax=13):
    f1, f2 = fname, fname
    if fname in sww:
        # find nearby fluxes
        near_check = [True, True]
        while any(near_check):
            if f1 in sww: f1 = f1[:10] + fmt_clump(int(f1[10:])-25)
            if f2 in sww: f2 = f2[:10] + fmt_clump(int(f2[10:])+25)
            near_check = [f in sww for f in [f1, f2]]
        SLEDs = np.array([interSLED(wd, f, logNH, Jmax) for f in [f1,f2]])
        iSLED = paramSLED(f1, f2, SLEDs, float(fname[10:])/1e2, Jmax=Jmax)
    else:
        iSLED = interSLED(wd, fname, logNH, Jmax=Jmax)
    return f1, f2, iSLED




### CLASSES

class GMC:
    def __init__(self, name, M, R=None, rho0=None, Mach=None, T=None):
        self.name = name
        self.M = M
        self.R = R if R is not None else 0
        self.V = (4/3) * pi * (self.R * pc)**3
        self.rho0 = rho0 if rho0 is not None else 0
        self.Mach = Mach if Mach is not None else 0
        self.T = T if T is not None else 0
        self.n0 = rho0 / (mu * mp) if rho0 is not None else 0
        self.ndf_name = 'GMC_' + self.name + '_n%.0f' % self.n0
        self.ndf_name += '_' + expo(self.M) + 'Msun'
        self.ndf_name += '_%.0f' % self.R + 'pc'
        self.ndf_name += '_M' + '%.0f' % self.Mach
        if T is not None: self.ndf_name += '_%.0f' % self.T + 'K'
        if unibo_cluster: pext = pbase + 'gmc/'
        else: pext = '/media/phd/jupyter/claudia/extractions/'
        self.wd = pext + self.ndf_name + '/'
        self.ndf = self.wd + self.ndf_name + '.csv'
    
    def extract_clumps(self, n_min=None, Rclump='jeans', 
        bound='mass', save=False):
        '''
        Creates a DataFrame containing n,R,M,NH of each GMC clump
        The clumps are extracted from a lognormal density distribution
        Rclump is the method for assigning R,M to each n-clump
        bound is the method for limiting the extraction
        '''
        n_min = self.n0 if n_min is None else n_min
        sigma = np.log(1 + b_turb**2 * self.Mach**2)
        n, R, M, NH = [], [], [], []
        Qtot = 0.
        if bound == 'mass': Qgmc = self.M
        elif bound == 'volume': Qgmc = self.V
        while Qtot < Qgmc:
            rng = np.random.default_rng()
            n1 = rng.lognormal(mean=np.log(self.n0), sigma=sigma)
            if n1 > n_min and n1 <= 10**hden[-1]:
                n.append(n1)
                if Rclump == 'jeans':
                    R.append(jeans(n[-1], self.T)[0])
                    M.append(jeans(n[-1], self.T)[1])
                NH.append(n[-1] * R[-1] * pc)
                if bound == 'mass': Qtot += M[-1]
                elif bound == 'volume': Qtot += (4/3)*pi * (R[-1]*pc)**3
        n, R = np.array(n), np.array(R)    # cm-3, pc
        M, NH = np.array(M), np.array(NH)  # Msun, cm-2
        # ICM properties
        nicm, i = 0, 0     # start from this
        while nicm < 10**hden[0]:
            if len(n) > 0:
                n, R, M, NH = n[:-1], R[:-1], M[:-1], NH[:-1]
                Micm = self.M - M.sum()
                Vicm = self.V - ((4/3) * pi * np.sum((R*pc)**3))
                Ricm = (Vicm * 3/(4*pi))**(1/3) / pc
                nicm = Micm*msun / (mu*mp*Vicm)
                i += 1
            else:
                print('Extraction failed!', i)
                break
        # add ICM to the clump ndf
        n, R = np.append(n, nicm), np.append(R, Ricm)
        M, NH = np.append(M, Micm), np.append(NH, nicm*Ricm*pc)
        # export results
        ndf = pd.DataFrame({
            'hden': np.log10(n), 'logR': np.log10(R),   # cm-3, pc
            'logM': np.log10(M), 'logNH': np.log10(NH)  # Msun, cm-2
            })
        if save:
            try:
                os.mkdir(self.wd)
                ndf.to_csv(self.ndf, index=False)
                ndf_histo(ndf, self.ndf)
            except OSError:
                print('Cannot save: remove first directory %s' % self.wd)
        return ndf
    
    def generate_clumps_sled(self, wd, Jmax=13):
        sww = pd.read_csv(wd + 'swwlist.csv', header=None)[0].to_list()
        sww = [s.replace('.out', '') for s in sww]
        ndf = pd.read_csv(self.ndf)
        exbase = self.ndf.replace('.csv', '_')
        COls = ['CO' + str(c) for c in np.arange(1,Jmax+1)]
        pbar = ProgressBar()
        for k in pbar(range(len(ndf))):
            logn = ndf.loc[k, 'hden']
            logR = ndf.loc[k, 'logR'] + np.log10(pc)
            hh = [hden[hden < logn].max(), hden[hden > logn].min()]
            # PDR
            extract = pd.DataFrame(index=gnames[::-1], columns=COls)
            for g in gnames:
                ff = ['pdr_h%03d_' % (h*1e2) + g for h in hh]
                Hsleds = np.array([sww_new(wd, fname, logR, sww, Jmax)[2] for fname in ff])
                thisSLED = paramSLED(ff[0], ff[1], Hsleds, logn, Jmax)
                extract.loc[g] = np.array(thisSLED)
            extract.to_csv(exbase + 'PDR{:05d}.csv'.format(k))
            # XDR
            extract = pd.DataFrame(index=xnames[::-1], columns=COls)
            for x in xnames:
                ff = ['xdr_h%03d_' % (h*1e2) + x for h in hh]
                Hsleds = np.array([sww_new(wd, fname, logR, sww, Jmax)[2] for fname in ff])
                thisSLED = paramSLED(ff[0], ff[1], Hsleds, logn, Jmax)
                extract.loc[x] = np.array(thisSLED)
            extract.to_csv(exbase + 'XDR{:05d}.csv'.format(k))
    
    def generate_parallel_sled(self, wd, Jmax=13):
        sww = pd.read_csv(wd + 'swwlist.csv', header=None)[0].to_list()
        sww = [s.replace('.out', '') for s in sww]
        ndf = pd.read_csv(self.ndf)
        num_cores = multiprocessing.cpu_count()
        Parallel(n_jobs=num_cores)(delayed(self.parallel_clumps)(
                k, wd, Jmax, sww) for k in range(len(ndf)))
    
    def parallel_clumps(self, k, wd, Jmax, sww):
        COls = ['CO' + str(c) for c in np.arange(1,Jmax+1)]
        exbase = self.ndf.replace('.csv', '_')
        ndf = pd.read_csv(self.ndf)
        logn = ndf.loc[k, 'hden']
        logR = ndf.loc[k, 'logR'] + np.log10(pc)
        hh = [hden[hden < logn].max(), hden[hden > logn].min()]
        # PDR
        extract = pd.DataFrame(index=gnames[::-1], columns=COls)
        for g in gnames:
            ff = ['pdr_h%03d_' % (h*1e2) + g for h in hh]
            Hsleds = np.array([sww_new(wd, fname, logR, sww, Jmax)[2] for fname in ff])
            thisSLED = paramSLED(ff[0], ff[1], Hsleds, logn, Jmax)
            extract.loc[g] = np.array(thisSLED)
        extract.to_csv(exbase + 'PDR{:05d}.csv'.format(k))
        # XDR
        extract = pd.DataFrame(index=xnames[::-1], columns=COls)
        for x in xnames:
            ff = ['xdr_h%03d_' % (h*1e2) + x for h in hh]
            Hsleds = np.array([sww_new(wd, fname, logR, sww, Jmax)[2] for fname in ff])
            thisSLED = paramSLED(ff[0], ff[1], Hsleds, logn, Jmax)
            extract.loc[x] = np.array(thisSLED)
        extract.to_csv(exbase + 'XDR{:05d}.csv'.format(k))
    
    def pdr_xdr_clumps(self, units='lsun', only_icm=False):
        '''
        Calculates the CO SLED for every clump of a GMC
        It returns two lists of n DataFrames (n = number of clumps)
        '''
        pdr_df, xdr_df = [], []
        ndf = pd.read_csv(self.ndf)
        if only_icm: clumplist = [len(ndf) - 1]
        else: clumplist = range(len(ndf))
        for n in clumplist:
            suf = '{:05d}'.format(n) + '.csv'
            pdr_name = self.ndf.replace('.csv', '_PDR' + suf)
            xdr_name = self.ndf.replace('.csv', '_XDR' + suf)
            fPDR = pd.read_csv(pdr_name, index_col=0)
            fXDR = pd.read_csv(xdr_name, index_col=0)
            logR_cm = ndf.loc[n, 'logR'] + np.log10(pc)
            if units == 'lsun': lconv = 2*pi*(10**logR_cm)**2 / lsun
            elif units == 'flux': lconv = 1.
            pdr_df.append(fPDR * lconv)
            xdr_df.append(fXDR * lconv)
        return pdr_df, xdr_df
    
    def pdr_xdr(self, units='lsun', icm=True):
        '''
        Import csv with PDR and XDR CO SLEDs for a GMC
        If not found, it creates them by summing all the clumps SLEDs
        If units='lsun' it calculates the luminosity of each line
        If units='flux' it keeps the Cloudy flux units (erg/s/cm2) 
        '''
        pdr_csv = self.wd + self.ndf_name.replace('GMC', 'PDR') + '.csv'
        xdr_csv = self.wd + self.ndf_name.replace('GMC', 'XDR') + '.csv'
        if icm:
            try:
                pdr_gmc = pd.read_csv(pdr_csv, index_col=0)
                xdr_gmc = pd.read_csv(xdr_csv, index_col=0)
            except FileNotFoundError:
                print(self.name, 'Files not found: calculating them now')
                pdr_df, xdr_df = self.pdr_xdr_clumps(units)
                pdr_gmc = reduce(lambda x, y: x.add(y, fill_value=0), pdr_df)
                xdr_gmc = reduce(lambda x, y: x.add(y, fill_value=0), xdr_df)
                pdr_gmc.to_csv(pdr_csv)
                xdr_gmc.to_csv(xdr_csv)
        else:
            pdr_df, xdr_df = self.pdr_xdr_clumps(units, only_icm=True)
            try:
                pdr_gmc = pd.read_csv(pdr_csv, index_col=0) - pdr_df[0]
                xdr_gmc = pd.read_csv(xdr_csv, index_col=0) - xdr_df[0]
            except FileNotFoundError:
                print(self.name, 'Files not found: calculating them now')
                pdr_df, xdr_df = self.pdr_xdr_clumps(units)
                sum(pdr_df).to_csv(pdr_csv)
                sum(xdr_df).to_csv(xdr_csv)
                pdr_gmc = reduce(lambda x, y: x.add(y, fill_value=0), pdr_df[:-1])
                xdr_gmc = reduce(lambda x, y: x.add(y, fill_value=0), xdr_df[:-1])
        return pdr_gmc, xdr_gmc
    
    def pdr_xdr_plot_SLED(self, Jmax=13, norm=0, save=False):
        '''
        Plots (and saves) the CO SLED of a GMC, PDR and XDR cases
        '''
        pdr_gmc, xdr_gmc = self.pdr_xdr()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9,5), sharey=True)
        color = cm.Purples(np.linspace(0.3, 1, len(pdr_gmc.index[::4])))
        for g, c in zip(pdr_gmc.index[::4], color):
            if norm: denom = pdr_gmc.loc[g, pdr_gmc.columns[norm-1]]
            else: denom = 1.
            ax1.plot(np.arange(1,Jmax+1), (pdr_gmc.loc[g]/denom)[:Jmax],
                marker='.', ms=14, mec='k', color=c,
                label=float(g[1:])/1e2)
        ax1.legend(title='log(G0)', loc='lower left')
        color = cm.Greens(np.linspace(0.3, 1, len(xdr_gmc.index[::4])))
        for x, c in zip(xdr_gmc.index[::4], color):
            if norm: denom = xdr_gmc.loc[x, xdr_gmc.columns[norm-1]]
            else: denom = 1.
            ax2.plot(np.arange(1,Jmax+1), (xdr_gmc.loc[x]/denom)[:Jmax],
                marker='.', ms=14, mec='k', color=c, 
                label=float(x[1:])/1e2)
        ax2.legend(title='log(FX)', loc='lower right')
        ax2.yaxis.set_label_position('right')
        ax2.yaxis.tick_right()
        ax2.yaxis.set_tick_params(labelright=True)
        for ax in [ax1, ax2]:
            ax.set_xticks(np.arange(1,Jmax+1,2))
            ax.set_yscale('log')
            ax.set_xlabel('J')
            if norm:
                nlab = r'L$_{{CO({} \rightarrow {})}}$'.format(norm, norm-1)
                ax.set_ylabel(r'L$_{CO(J \rightarrow J-1)}$ / ' + nlab)
            else:
                ax.set_ylabel(r'L$_{CO(J \rightarrow J-1)}$ [L$_{\odot}$]')
        fig.suptitle(self.ndf_name)
        fig.tight_layout();
        if save: 
            plt.savefig(self.ndf.replace('.csv', '_COSLED.png'),
                bbox_inches='tight', dpi=300)
        else: plt.show()
        plt.close()
    
    def galaxy_xdr(self, logLX, rCO, Mmol_tot):
        '''       [_, 1-100 keV, kpc, Msun]
        Returns a Series with the number of XDRs for each FX bin
        for a galaxy with given logLX(1-100 keV), rCO, Mmol_tot
        The GMCs out of every FX bin end up in the 'rmndr' column
        '''
        logRX = (logLX - logFX - np.log10(4*pi))/2 - np.log10(kpc)
        Nxdr = pd.Series(index=xnames + ['rmndr'], dtype=np.int64)
        for logr, F, xx in zip(logRX, logFX, xnames):
            r = np.array([10**(logr-0.125/2), 10**(logr+0.125/2)])
            Ngmc = (Mmol_r(r, Mmol_tot, rCO) / self.M).astype('int')
            Nxdr.loc[xx] = Ngmc[1] - Ngmc[0]
        Nxdr.loc['rmndr'] = int(Mmol_tot / self.M) - Ngmc[1]
        return Nxdr
    
    def galaxy_pdr(self, Ie, Re, n, rCO, Mmol_tot, SFR, G0floor=True):
        '''       [_, erg/s/cm2, kpc, (n), kpc, Msun, Msun/yr, *]
        Returns a Series with the number of PDRs for each G0 bin
        for a galaxy with a fitted Sersic profile (Ie, Re, n)
        and with given rCO, Mmol_tot and SFR.
        The GMCs out of every G0 bin end up in the 'rmndr' column
        If G0floor=True, the SFR acts as a minimum G0 value:
        no GMC can see a G0 lower than this G0_min = SFR
        '''
        rmin, rmax = r_G0(Ie, Re, n)[1:]
        rmin_rmax = np.array([[x,y] for x, y in zip(rmin, rmax)])
        Npdr = pd.Series(index=gnames + ['rmndr'], dtype=np.int64)
        for r, G, gg in zip(rmin_rmax, logG0, gnames):
            Ngmc = (Mmol_r(r, Mmol_tot, rCO) / self.M).astype('int')
            Npdr.loc[gg] = Ngmc[1] - Ngmc[0]
        Npdr.loc['rmndr'] = int(Mmol_tot / self.M) - Ngmc[1]
        # setting a floor for G0 = SFR
        if G0floor == True:
            logSFR = np.log10(SFR)
            if logSFR > logG0[-1] - 0.25:   # threshold
                flidx = np.argmin(np.abs([logSFR - G for G in logG0]))
                floor = logG0[flidx]
                last_line = 'g' + ('%.2f' % floor).replace('.', '')
                GMCfloor = Npdr.loc[last_line:].sum()
                Npdr.loc[last_line:] = 0
                Npdr.loc[last_line] = GMCfloor
        return Npdr
    
    def galaxy_pdr_kick(self, Npdr, SFR):
        Npdr_kicked = Npdr.copy()
        kick = round(Npdr.sum() * SFR / 1e3)
        take = random.choices(
            population = Npdr.index.to_list(), 
            weights = Npdr.values/Npdr.sum(), 
            k = kick)
        ll_idx = [(Npdr[i] < Npdr[i-1]) & (Npdr[i] == 0) 
                  for i in range(1, len(Npdr))]
        if any(ll_idx):
            last_line = Npdr.index[np.where(ll_idx)[0][0]]
        else: last_line = 'rmndr'
        Npdr_tmp = Npdr.copy()
        Npdr_tmp[last_line:] = Npdr.sum()
        give = random.choices(population = Npdr.index.to_list(), 
                       weights = Npdr.sum() - Npdr_tmp.values, 
                       k = kick)
        for t, g in zip(take, give):
            Npdr_kicked[t] -= 1
            Npdr_kicked[g] += 1
        return Npdr_kicked
    
    def icm_sled(self, galaxy, Jmax=13, logNH=None):
        '''
        Calculates the CO SLED for the spread ICM for a Esposito+22 galaxy.
        ---------------> !!!!  If log(n_icm) < -0.125 it ???      !!!!
        If logNH=None it uses rCO both for Cloudy output and luminosity
        Else: it uses logNH for Cloudy output row and rCO for luminosity
        '''
        if 'turb' in self.name: pgrid = '/media/phd/cloudy/turbogrid/'
        else: pgrid = '/media/phd/cloudy/ngmgrid/'
        sww = pd.read_csv(pgrid + 'swwlist.csv', header=None)[0].to_list()
        sww = [s.replace('.out', '') for s in sww]
        
        picm = '/media/phd/jupyter/claudia/icm_sleds/'
        if logNH: picm = picm[:-1] + '_logNH%3d' % (logNH*10) + '/'
        exbase = picm + galaxy + '_icmSLED_' + self.name + '_'
        COls = ['CO' + str(c) for c in np.arange(1,Jmax+1)]
        
        rCO, Mtot = df.loc[galaxy, ['rCO', 'Mmol_tot_XMW']]
        Vtot = 2 * pi * (rCO*kpc)**3 * 0.01/ 0.17  # cylinder with height=zCO
        logMicm = pd.read_csv(self.ndf).iloc[-1]['logM']
        M_icm_gal = (10**logMicm) * Mtot / self.M  # summing the ICM clumps
        n_icm = msun*M_icm_gal / (Vtot * mu * mp)  # spreading the ICM over Vtot
        
        if n_icm >= 10**(-0.125):                  # density threshold
            try:
                pdr_icm = pd.read_csv(exbase + 'PDR.csv', index_col=0)
                xdr_icm = pd.read_csv(exbase + 'XDR.csv', index_col=0)
            except FileNotFoundError:
                print(exbase, ': Files not found → calculating them now')
                logn = np.log10(n_icm)
                if logNH: logR = logNH - logn          # fixed logNH case
                else: logR = np.log10(rCO*kpc)         # Cloudy row output @ rCO case
                if logn < 0: logn = 0.00001            # approximate -0.125 <= log(n) <= 0
                hh = [hden[hden < logn].max(), hden[hden > logn].min()]
                pdr_icm = pd.DataFrame(index=gnames[::-1], columns=COls)
                for g in gnames:
                    ff = ['pdr_h%03d_' % (h*1e2) + g for h in hh]
                    Hsleds = np.array([sww_new(wd, fname, logR, sww, Jmax)[2] for fname in ff])
                    thisSLED = paramSLED(ff[0], ff[1], Hsleds, logn, Jmax)
                    pdr_icm.loc[g] = np.array(thisSLED)
                pdr_icm = pdr_icm * 4*pi*(rCO*kpc)**2 / lsun # L = 4 pi rCO^2
                pdr_icm.to_csv(exbase + 'PDR.csv')
                xdr_icm = pd.DataFrame(index=xnames[::-1], columns=COls)
                for x in xnames:
                    ff = ['xdr_h%03d_' % (h*1e2) + x for h in hh]
                    Hsleds = np.array([sww_NH(wd, fname, logNH, sww, Jmax)[2] for fname in ff])
                    thisSLED = paramSLED(ff[0], ff[1], Hsleds, logn, Jmax)
                    xdr_icm.loc[x] = np.array(thisSLED)
                xdr_icm = xdr_icm * 4*pi*(rCO*kpc)**2 / lsun # luminosity is integrated over rCO
                xdr_icm.to_csv(exbase + 'XDR.csv')
            return pdr_icm, xdr_icm
        else:
            raise Exception





# run with corrected jeans function
gmcA = GMC('A', 6.2e3, 4, 8.2e-22, 10, T=10)
gmcB = GMC('B', 9.9e4, 16, 2.1e-22, 20, T=10)
gmcC = GMC('C', 3.9e6, 100, 3.3e-23, 50, T=10)
gmcV19 = GMC('V19', 1e5, 15, 6.122e-22, 10, T=10)
gmcV20 = GMC('V20', 1e5, 15, 6.122e-22, 10, T=10)
#gmcAlfa = GMC('Alfa', )
#gmcBeta = GMC('Beta', )
#gmcGamma = GMC('Gamma', )

# run with the updated low Cloudy densities
gmcAlo = GMC('Alo', 6.2e3, 4, 8.2e-22, 10, T=10)
gmcBlo = GMC('Blo', 9.9e4, 16, 2.1e-22, 20, T=10)
gmcClo = GMC('Clo', 3.9e6, 100, 3.3e-23, 50, T=10)
gmcV19lo = GMC('V19lo', 1e5, 15, 6.122e-22, 10, T=10)

# run with turbogrid
gmcAturb = GMC('Aturb', 6.2e3, 4, 8.2e-22, 10, T=10)
gmcBturb = GMC('Bturb', 9.9e4, 16, 2.1e-22, 20, T=10)
gmcCturb = GMC('Cturb', 3.9e6, 100, 3.3e-23, 50, T=10)
gmcV19turb = GMC('V19turb', 1e5, 15, 6.122e-22, 10, T=10)


# run with old jeans
gmcAdam = GMC('Adam', M=1e4, R=5, rho0=mu*mp*300, Mach=50, T=10)
gmcLilith = GMC('Lilith', M=1e4, R=5, rho0=mu*mp*300, Mach=50, T=10)
gmcSachiel = GMC('Sachiel', M=2e3, R=2, rho0=mu*mp*300, Mach=20, T=10)

gmclist = [gmcA, gmcB, gmcC, gmcV19]
gmclost = [gmcAlo, gmcBlo, gmcClo, gmcV19lo]
gmcturb = [gmcAturb, gmcBturb, gmcCturb, gmcV19turb]


