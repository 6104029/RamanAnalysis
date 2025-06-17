# RamanAnalysis
## Description
This repository contains my RamanAnalysis_package. Among others, the package can be used to import .txt files and fit carbon Raman spectra with Lorentzians and a Gaussian.

## Installation 
- clone the repository:
```
git clone https://github.com/6104029/RamanAnalysis
```
- install the package by **navigating to the cloned repository** in your python environment and executing the following command:

```
pip install -r requirements.txt
pip install -e .
```
- You should now be able to load the package in python by using:

```python
import DRIFTS_package as ir
```

## Installation (simplest version)
- Download the repository. 
- Copy the *RamanAnalysis* folder to the folder where the script you wanna use for analysis is located. Example:

```
My_Scripts
│
└───Raman
│   │   myRaman_analysis.py
│   │   myRaman_notebook.ipynb
│   │
│   └───*RamanAnalysis*
```

## Usage
Check the example notebook for more info how to use this package.

## Support
If you find bugs or have other questions send me a message or open an issue.

## Contributing
If you want to contribute, let me know.

## Authors
Hanya Spoelstra, Utrecht University

## Acknowledgements
Package structure was adapted from pyTGA, built by Sebastian Rejman
