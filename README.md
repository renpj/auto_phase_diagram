# auto_phase_diagram
Automatic analysis and plot phase diagram from DFT calculation.

## Pre-request
* python >2.7, not support python 3 now
* numpy
* pandas (data analysis)
* xlrt, xlwt (handle excel file)
* veusz (for plot images)
* xvfb (simulate X environment)
* xvfbwrapper (python wrapper for xvfb)

*Most of them can install by apt (Debian-based) and pip*
## Usage
1. Put all required data into an excel file, name as e.g. auto_phase_diagram.xlsx 
2. run python auto_phase_diagram.py auto_phase_diagram.xlsx

## Output
1. G_result.xlsx
2. \*.vsz
3. \*.jpg
