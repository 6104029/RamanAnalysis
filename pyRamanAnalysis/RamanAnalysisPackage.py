import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob as glob
import os
import sys
import math
import scipy.optimize as opt
from sklearn.metrics import r2_score
import seaborn as sns

def load_insitu_data(file_name, rows_per_spectrum=391, measurement_duration=62):
    """
    Reads Raman spectroscopy data from text files matching the given pattern.

    Parameters:
        file_pattern (str): The file name pattern to match (e.g., "HS006_400_20min_60x1_1500stat_LP5_*.txt").
        rows_per_spectrum (int): Number of rows per spectrum in the data files.
        measurement_duration (int): Duration of the measurement in seconds.

    Returns:
        tuple: A tuple containing:
            - rmn (list): A list of DataFrames, each representing a spectrum.
            - selected_data (dict): A dictionary with file and spectrum identifiers as keys and spectra as values.
            - measdur (int): The measurement duration in seconds.
    """

    # Initialize an empty list for reading data and an empty dictionary for selected data
    rmn = []
    selected_data = {}

    # Get the list of files matching the pattern and sort them
    files = glob.glob(file_name)
    files.sort(key=str.lower)

    # Loop through the files and read them
    for n in files:
        try:
            # Read the entire file into a DataFrame
            data = pd.read_csv(n, sep='\t', header=None, names=['Depth', 'Wavenumber', 'Intensity'], skiprows=0, index_col=False)
            
            # Determine the number of spectra in the file
            num_spectra = len(data) // rows_per_spectrum
            
            # Split the data into individual spectra
            spectra = [data.iloc[i*rows_per_spectrum:(i+1)*rows_per_spectrum].drop(columns=['Depth']) for i in range(num_spectra)]
            
            # Append all spectra to the rmn list
            rmn.extend(spectra)
            
            # Store each spectrum in the selected_data dictionary with a unique key
            for i, spectrum in enumerate(spectra):
                selected_data[f"{n}_spectrum_{i+1}"] = spectrum
        except pd.errors.ParserError:
            # Handle the exception if there is a parsing error
            print(f"Error parsing file: {n}")

    return rmn, selected_data, measurement_duration

def quickplot():
    """
    This function reads all text files in the current directory,
    extracts the first two columns (wavenumber and intensity),
    and plots the data.
    """    
    post_reaction_files = glob.glob("*.txt")
    # print(post_reaction_files)

    for file in post_reaction_files:
        # Read the data from the file
        data = pd.read_csv(file, sep="\t", header=None)
        
        # Extract the first column (time) and the second column (current)
        wavenumber = data[0]
        intensity = data[1]
        
        # Plot the data
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(wavenumber, intensity, label=os.path.basename(file))
        ax.legend()
        
        # Close the figure to prevent it from being displayed again
    return fig, ax

def load_exsitu_data(path = os.getcwd(), filetype = '*.txt'):
    """
    Load ex situ data (all txt files) from a specified directory and return the data as a DataFrame.
    
    Returns:
        pd.DataFrame: The loaded ex situ data and a list of file names.
    """
    
    # Imports raw data files into list 'rmn' 
    files = glob.glob(os.path.join(path, filetype))
    rmn = []
    selected_data = {}  # Change selected_data from list to dictionary
    files.sort(key=str.lower)
    for n in files:
        try:
            rmn.append(pd.read_csv(n, sep='\t', header=None, names=['Wavenumber', 'Intensity'], skiprows=0, index_col=False))
        except pd.errors.ParserError:
            print(f"Error parsing file: {n}")

    return rmn, files

def plot_exsitu_data(rmn, b1=None, b2=None):
    ##############################################################################  
    """
    Plot the ex situ data from the list of DataFrames.
    
    Parameters:
        rmn (list): List of DataFrames containing the ex situ data.
        b1 (float, optional): Lower bound for wavenumber range. If None, uses min of first DataFrame.
        b2 (float, optional): Upper bound for wavenumber range. If None, uses max of first DataFrame.
    """
    if b1 is None:
        b1 = rmn[0]['Wavenumber'].min()
    if b2 is None:
        b2 = rmn[0]['Wavenumber'].max()
    clr = plt.cm.plasma(np.linspace(0,0.5,len(rmn[:])))
    fig, ax = plt.subplots(figsize=(7,5))

    for i1,rmn_f in enumerate(rmn):
        # fig, ax = plt.subplots(figsize=(7,5))
        x = rmn_f['Wavenumber']
        y = rmn_f['Intensity']
        xc = x[(x > b1) & (x < b2)]
        yc = y[(x > b1) & (x < b2)]
        ycn = yc/yc[(x > 1000) & (x < 1500)].max()
        ax.plot(xc,ycn,color=clr[i1] )	

        ax.set_xlabel('Raman Shift (cm$^-$$^1$)')
        ax.set_ylabel('Intensity (a.u.)')
        ax.set_yticks([])

def define_boundaries():
    ## Sets the boundaries for the fit
    #Peak position, width, and amplitude per band
    #D band/ D1
    B1D1 = np.array([1320, 0 , 0])
    IGD1 = np.array([1330, 150, 0.5])
    B2D1 = np.array([1340, 500,200])

    #G band
    B1G = np.array([1589, 0 , 0])
    IGG = np.array([1590, 50, 0.5])
    B2G = np.array([1591, 100,200])

    #D* band/ D4
    B1D4 = np.array([1120, 0 , 0])
    IGD4 = np.array([1175, 175, 0.5])
    B2D4 = np.array([1210, 320,20])

    #D'' band/ D3   
    B1D3 = np.array([1490, 0 , 0])
    IGD3 = np.array([1525, 150, 0.5])
    B2D3 = np.array([1550, 210,200])

    #D' band/ D2
    B1D2 = np.array([1600, 0 , 0])
    IGD2 = np.array([1620, 50, 0.5])
    B2D2 = np.array([1640, 100,200])

    #Two phonon range
    #2D4 band
    B12D4 = np.array([2400, 0 , 0])
    IG2D4 = np.array([2450, 50, 0.5])
    B22D4 = np.array([2600, 250,200])

    #2D band
    B12D = np.array([2600, 0 , 0])
    IG2D = np.array([2670, 50, 0.5])
    B22D = np.array([2700, 250,200])

    #D+G band
    B1DG = np.array([2900, 0 , 0])
    IGDG = np.array([2950, 50, 0.5])
    B2DG = np.array([3000, 250,200])

    #2D2 band
    B12D2 = np.array([3000, 0 , 0])
    IG2D2 = np.array([3100, 50, 0.5])
    B22D2 = np.array([3200, 250,200])


    #background polynomial
    B1BG = [-10, -10, -10]
    IGBG = [0, 0, 0]
    B2BG = [10, 10, 10]

    #background linear
    B1BGlin = [-10, -10]
    IGBGlin = [0, 0]
    B2BGlin = [10, 10]

    #Sets the boundaries for the first part of the fit
    B1 = []; B1.extend(B1D1); B1.extend(B1G); B1.extend(B1D2); B1.extend(B1D4); B1.extend(B1D3); B1FIRST = B1 
    IG = []; IG.extend(IGD1); IG.extend(IGG); IG.extend(IGD2); IG.extend(IGD4); IG.extend(IGD3); IGFIRST = IG 
    B2 = []; B2.extend(B2D1); B2.extend(B2G); B2.extend(B2D2); B2.extend(B2D4); B2.extend(B2D3); B2FIRST = B2 

    #Sets the boundaries for the second part of the fit
    B1 = []; B1.extend(B12D4); B1.extend(B12D); B1.extend(B1DG); B1.extend(B12D2);  B1SECOND = B1
    IG = []; IG.extend(IG2D4); IG.extend(IG2D); IG.extend(IGDG); IG.extend(IG2D2);  IGSECOND = IG
    B2 = []; B2.extend(B22D4); B2.extend(B22D); B2.extend(B2DG); B2.extend(B22D2);  B2SECOND = B2

    return B1FIRST, IGFIRST, B2FIRST, B1SECOND, IGSECOND, B2SECOND, B1BG, IGBG, B2BG


def fit_exsitu_data(rmn, B1FIRST, IGFIRST, B2FIRST, B1SECOND, IGSECOND, B2SECOND, B1BG, IGBG, B2BG, b1=None, b2=None):
    """
    Fit the ex situ data from the list of DataFrames and obtain the optimized parameters.
    """
    if b1 is None:
        b1 = rmn[0]['Wavenumber'].min()
    if b2 is None:
        b2 = rmn[0]['Wavenumber'].max()
    
    # Defines Lorentzian function
    def L1(x,p,w,a):
        L = a*((1/math.pi)*0.5*w/((x-p)**2+(0.5*w)**2))
        return L

    def G1(x,p,w,a):
        G = a*(1/(w*math.sqrt(2*math.pi)))*np.exp(-0.5*((x-p)/w)**2)
        return G

    def BG(x,m,b,c):
        BG = m*x**2 + b*x + c
        return BG

    def L4G1(x,p1,w1,a1,p2,w2,a2,p3,w3,a3,p4,w4,a4,p5,w5,a5,m,b,c):
        L = L1(x,p1,w1,a1) + L1(x,p2,w2,a2) + L1(x,p3,w3,a3) + L1(x,p4,w4,a4) + G1(x,p5,w5,a5) + BG(x,m,b,c)
        return L

    def L4(x,p1,w1,a1,p2,w2,a2,p3,w3,a3,p4,w4,a4,m,b,c):
        L = L1(x,p1,w1,a1) + L1(x,p2,w2,a2) + L1(x,p3,w3,a3) + L1(x,p4,w4,a4) + BG(x,m,b,c)
        return L
    
    OPTtot = []

    for i1,rmn_f in enumerate(rmn):
        x = rmn_f['Wavenumber']
        y = rmn_f['Intensity']

        xc = x[(x > b1) & (x < b2)]
        yc = y[(x > b1) & (x < b2)]

        #Normalize the data
        ycn = yc/yc[(x > 1000) & (x < 1500)].max()

        #Define part of the spectrum that is background
        xcbg = xc [(xc > 200) & (xc < 290)|(xc > 730) & (xc < 850) | (xc > 1700) & (xc < 2200) | (xc > 3350)]
        ycbg = ycn[(xc > 200) & (xc < 290)|(xc > 730) & (xc < 850) | (xc > 1700) & (xc < 2200) | (xc > 3350)]
        
        #Fit the background
        OPTBG = opt.curve_fit(BG, xcbg, ycbg, p0=IGBG, bounds=(B1BG,B2BG))[0]

        #Sets optimized background as boundaries for the BG fit
        VAR = abs(OPTBG*1.01-OPTBG)
        B1BGpol = OPTBG - VAR
        B2BGpol = OPTBG + VAR
        IGBGpol = OPTBG

        #Boundaries for the first part of the fit
        B1F = B1FIRST + list(B1BGpol)
        IGF = IGFIRST + list(IGBGpol)
        B2F = B2FIRST + list(B2BGpol)

        b1FIRST = 700
        b2FIRST = 2000
        xcFIRST = xc[(xc > b1FIRST) & (xc < b2FIRST)]
        ycFIRST = ycn[(xc > b1FIRST) & (xc < b2FIRST)]

        #Fit the data
        OPT = opt.curve_fit(L4G1,xcFIRST,ycFIRST,p0=IGF, bounds=(B1F,B2F))[0] 
        fit1 = L4G1(xc, *OPT) #Bereken de gefitte y-waarden
        ratioDG = OPT[2]/OPT[5]
        #Boundaries for the second part of the fit
        B1S = B1SECOND + list(B1BGpol)
        IGS = IGSECOND + list(IGBGpol)
        B2S = B2SECOND + list(B2BGpol)

        b1SECOND = 2000
        b2SECOND = 3200
        xcSECOND = xc[(xc > b1SECOND) & (xc < b2SECOND)]
        ycSECOND = ycn[(xc > b1SECOND) & (xc < b2SECOND)]

        #Fit the data
        OPT2 = opt.curve_fit(L4,xcSECOND,ycSECOND,p0=IGS, bounds=(B1S,B2S))[0]
        fit2 = L4(xc, *OPT2) #Bereken de gefitte y-waarden

        OPTspec = list(OPT[:-3]) + list(OPT2)
        OPTtot.append(OPTspec)
    return OPTtot

def fit_exsitu_data_2(rmn, b1=None, b2=None):

    if b1 is None:
        b1 = rmn[0]['Wavenumber'].min()
    if b2 is None:
        b2 = rmn[0]['Wavenumber'].max()

    B1D1 = np.array([1320, 0 , 0])
    IGD1 = np.array([1330, 150, 0.5])
    B2D1 = np.array([1340, 500,200])

    #G band
    B1G = np.array([1570, 0 , 0])
    IGG = np.array([1580, 50, 0.5])
    B2G = np.array([1590, 100,200])

    B1BG = [-10, -10, -10]
    IGBG = [0, 0, 0]
    B2BG = [10, 10, 10]

    def L1(x,p,w,a):
        L = a*((1/math.pi)*0.5*w/((x-p)**2+(0.5*w)**2))
        return L
    def BG(x,m,b,c):
        BG = m*x**2 + b*x + c
        return BG
    def L2(x,p1,w1,a1,p2,w2,a2,m,b,c):
        L = L1(x,p1,w1,a1) + L1(x,p2,w2,a2) + BG(x,m,b,c)
        return L
    OPTtot = []

    for i1,rmn_f in enumerate(rmn):
        x = rmn_f['Wavenumber']
        y = rmn_f['Intensity']

        xc = x[(x > b1) & (x < b2)]
        yc = y[(x > b1) & (x < b2)]

        #Normalize the data
        ycn = yc/yc[(x > 1000) & (x < 1500)].max()

        #Define part of the spectrum that is background
        xcbg = xc [(xc > 200) & (xc < 290)|(xc > 730) & (xc < 850) | (xc > 1700) & (xc < 2200) | (xc > 3350)]
        ycbg = ycn[(xc > 200) & (xc < 290)|(xc > 730) & (xc < 850) | (xc > 1700) & (xc < 2200) | (xc > 3350)]
        
        #Fit the background
        OPTBG = opt.curve_fit(BG, xcbg, ycbg, p0=IGBG, bounds=(B1BG,B2BG))[0]

        #Sets optimized background as boundaries for the BG fit
        VAR = abs(OPTBG*1.01-OPTBG)
        B1BGpol = OPTBG - VAR
        B2BGpol = OPTBG + VAR
        IGBGpol = OPTBG

        #Boundaries for the first part of the fit
        B1F = list(B1D1) + list(B1G) + list(B1BGpol)
        IGF = list(IGD1) + list(IGG) + list(IGBGpol)
        B2F = list(B2D1) + list(B2G) + list(B2BGpol)

        b1FIRST = 700
        b2FIRST = 2000
        xcFIRST = xc[(xc > b1FIRST) & (xc < b2FIRST)]
        ycFIRST = ycn[(xc > b1FIRST) & (xc < b2FIRST)]

        #Fit the data
        OPT = opt.curve_fit(L2,xcFIRST,ycFIRST,p0=IGF, bounds=(B1F,B2F))[0] 
        fit1 = L2(xc, *OPT) #Bereken de gefitte y-waarden
        ratioDG = OPT[2]/OPT[5]

        OPTtot.append(OPT)
    return OPTtot

def plot_fitted_data( rmn, OPTtot, files, b1=None, b2=None):
    """
    Plot the fitted data from the list of DataFrames.
    """	
    if b1 is None:
        b1 = rmn[0]['Wavenumber'].min()
    if b2 is None:
        b2 = rmn[0]['Wavenumber'].max()

    def L1(x,p,w,a):
        L = a*((1/math.pi)*0.5*w/((x-p)**2+(0.5*w)**2))
        return L
    def G1(x,p,w,a):
        G = a*(1/(w*math.sqrt(2*math.pi)))*np.exp(-0.5*((x-p)/w)**2)
        return G
    def BG(x,m,b,c):
        BG = m*x**2 + b*x + c
        return BG
    def L8G1(x,p1,w1,a1,p2,w2,a2,p3,w3,a3,p4,w4,a4,p5,w5,a5,p6,w6,a6,p7,w7,a7,p8,w8,a8,p9,w9,a9,m,b,c):
        L = L1(x,p1,w1,a1) + L1(x,p2,w2,a2) + L1(x,p3,w3,a3) + L1(x,p4,w4,a4) +G1(x,p5,w5,a5) + L1(x,p6,w6,a6) + L1(x,p7,w7,a7) + L1(x,p8,w8,a8) + L1(x,p9,w9,a9) + BG(x,m,b,c)
        return L
    
    for i1, (OPT, rmn_f) in enumerate(zip(OPTtot, rmn)):
        x = rmn_f['Wavenumber']
        y = rmn_f['Intensity']

        xc = x[(x > b1) & (x < b2)]
        yc = y[(x > b1) & (x < b2)]
        ycn = yc / yc[(xc > 1000) & (xc < 1500)].max()

        fit = L8G1(xc, *OPT) 
        ratioDG = OPT[2] / OPT[5]

        fig, ax = plt.subplots(figsize=(7,5))
        ax.plot(xc, ycn, '.')
        ax.plot(xc, fit, c='red', label = 'Fit1, D/G ratio = ' + str(round(ratioDG,2)))
        
        #Plots individual Lorentzians
        # Plot 4 Lorentzians from first part
        for i in range(0, 12, 3):
            L = L1(xc, OPT[i], OPT[i+1], OPT[i+2]) + BG(xc,OPT[-3],OPT[-2],OPT[-1])
            ax.plot(xc, L, c='black', alpha=0.25)

        # Plot Gaussian from first part
        ax.plot(xc, G1(xc, OPT[12], OPT[13], OPT[14]) + BG(xc,OPT[-3],OPT[-2],OPT[-1]), c='black', alpha=0.25)

        # Plot 4 Lorentzians from second part (starts at index 15)
        for i in range(15, 27, 3):
            L = L1(xc, OPT[i], OPT[i+1], OPT[i+2]) + BG(xc,OPT[-3],OPT[-2],OPT[-1])
            ax.plot(xc, L, c='black', alpha=0.25)

        #Plots Background
        ax.plot(xc, BG(xc,OPT[-3],OPT[-2],OPT[-1]), c='green', alpha=0.5, label = 'Background')

        #Plots Difference
        ax.plot(xc, ycn - fit , c='blue', alpha =0.4, label = 'Difference')
        ax.plot(xc, np.zeros(len(xc)), c='black', alpha =0.4)
        ax.legend()

        #Sets limits and labels    
        ax.set_xlabel('Raman Shift (cm$^-$$^1$)'); ax.set_yticks([]); ax.yaxis.set_label_position('left')
        ax.set_ylabel('Intensity (a.u.)')
        ax.set_title(files[i1].split('_')[1])
        ax.set_xlim(b1, b2)
        # plt.savefig('HS006' + files[i1].split('_')[1] +'_'+ str(i1)+ '.svg')
    return fig, ax

def calculate_ratios(OPTtot):
    """
    Calculate the D/G and 2D/G ratios from the fitted parameters.
    
    Parameters:
        OPTtot (list): List of fitted parameters for each spectrum.
    
    Returns:
        list: List of tuples containing D/G and 2D/G ratios for each spectrum.
    """
    ratios = []
    ppos = []
    for i1, OPT in enumerate(OPTtot):

        # Calculate all the ratios
        #Calculate the ratios
        ratioDG = OPT[2]/OPT[5]
        ratioDDP = OPT[2]/OPT[8]
        widthG = OPT[4]
        ratioG2D = OPT[5]/OPT[19]
        ratio2DDG = OPT[19]/OPT[22]
        ratioGDP = OPT[5]/OPT[8]

        #Add the ratios to the list ratios
        ratios.append([ratioDG, ratioDDP, widthG, ratioG2D, ratio2DDG, ratioGDP])

        #adds peak positions to the list ppos
        pos2D4 = OPT[15]
        pos2D = OPT[18]
        posDG = OPT[21]
        pos2D2 = OPT[24]
        posD = OPT[0]
        posG = OPT[3]
        posD2 = OPT[6]
        posD3 = OPT[12]
        posD4 = OPT[9]
        ppos.append([posD, posG, posD2, posD4, posD3,pos2D4, pos2D, posDG, pos2D2])
    pposdf = pd.DataFrame(ppos)
    pposdf.columns = ['D', 'G', 'D2', "D3", "D4", "2D4", "2D", "DG", "2D2"]
    pposdf = pposdf.iloc[:].reset_index(drop=True)

    data_str = '\n'.join(map(str, ratios))
    data_str = data_str.replace('[', '').replace(']', '').replace(',', '')
    rows = data_str.strip().split('\n')
    table = [[float(col) for col in row.split()] for row in rows]
    headers = ['ratioDG', 'ratioDDP','widthG', 'ratioG2D', 'ratio2D2DG', 'ratioGDP']
    ratiosdf = pd.DataFrame(table, columns=headers)

    return ratios, ppos, pposdf, ratiosdf


def calculate_group_averages(ratiosdf, files, spectra_per_sample=10):
    """
    Calculate the group averages of the ratios DataFrame.
    """

    # Group DataFrame by every x rows (averaging over the number of spectra per sample)
    groups = ratiosdf.groupby(ratiosdf.index // spectra_per_sample)

    # Calculate the average of each group
    group_averages = groups.mean()

    # Define the number of spectra per sample
    ratiosdf['Sample'] = ratiosdf.index // spectra_per_sample
    df_melted = ratiosdf.melt(id_vars='Sample', var_name='Column', value_name='Value')
    fig, axes = plt.subplots(1, 6, figsize=(18, 6), sharey=False)
    tick_labels = [files[i].split('_')[1] for i in range(0, len(files), spectra_per_sample)]
    for i, column in enumerate(ratiosdf.columns[:-1]):  # Exclude 'Sample' column
        # Filter data for the current column
        data = df_melted[df_melted['Column'] == column]
        
        # Plot boxplot for each sample
        sns.boxplot(x='Sample', y='Value', data=data, ax=axes[i])

        # Set labels and title for each subplot
        axes[i].set_title(f'{column}')
        axes[i].set_xlabel('Sample')
        axes[i].set_ylabel('Values')
        axes[i].set_xticks(range(len(tick_labels)))
        axes[i].set_xticklabels(tick_labels, rotation=90)


        # Calculate dynamic y-axis limits based on min and max of each column
        ymin, ymax = data['Value'].min(), data['Value'].max()
        axes[i].set_ylim(ymin, ymax)



    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()
    return group_averages

def save_individual_spectra(FN):
    """
    Save individual spectra to CSV files.
    """

    # Get all matching files in the current folder
    files = glob.glob(FN + "*.txt")

    # Make a real output folder based on the prefix
    output_folder = f"Individual_spectra"
    os.makedirs(output_folder, exist_ok=True)
    print(f"Saving individual spectra to: {output_folder}")

    # Initialize
    rmn = []
    selected_data = {}
    rows_per_spectrum = 392
    measdur = 62

    # Sort files and process
    files.sort(key=str.lower)

    for n in files:
        try:
            data = pd.read_csv(n, sep='\t', header=None, names=['Depth', 'Wavenumber', 'Intensity'], index_col=False)
            num_spectra = len(data) // rows_per_spectrum

            for i in range(num_spectra):
                spectrum = data.iloc[i*rows_per_spectrum:(i+1)*rows_per_spectrum].drop(columns=['Depth'])
                rmn.append(spectrum)

                base_name = os.path.splitext(os.path.basename(n))[0]
                key = f"{base_name}_spectrum_{i+1}"
                selected_data[key] = spectrum

                # Save each spectrum
                out_path = os.path.join(output_folder, f"{key}.txt")
                spectrum.to_csv(out_path, sep='\t', index=False, header=False)
                print(f"Saved: {out_path}")

        except pd.errors.ParserError:
            print(f"Error parsing file: {n}")