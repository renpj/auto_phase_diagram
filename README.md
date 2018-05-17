# auto_phase_diagram
Automatic analysis and plot surface phase diagram from DFT calculation.

## Pre-request
* python >=2.7, not test for python 3 and python < 2.7
* numpy
* pandas (data analysis)
* xlrt, xlwt (handle excel file)

### Optional
* veusz (for plot images)
* xvfb (simulate X environment)
* xvfbwrapper (python wrapper for xvfb)

*Most of them can install by apt (Debian-based) and pip*
## Usage
1. Put all required data into an excel file, name as e.g. input.xlsx 
2. run "python auto_phase_diagram.py input.xlsx"

## Output
1. G_result.xlsx
2. \*.dat or \*.csv
3. \*.vsz
4. \*.jpg, if you have veusz and xvfb installed
