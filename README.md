# auto_phase_diagram
Automatic analysis and plot surface phase diagram from DFT calculation.

## Pre-request
* python
* numpy
* pandas (data analysis)
* xlrt, xlwt (handle excel file)

### Optional
* veusz (for plotting images)
* xvfb (simulate X environment)
* xvfbwrapper (python wrapper for xvfb)

*Most of them can install by apt (Debian-based) and pip*

## Install
1. python `setup.p`y install
2. add `export PATH=$PATH:YOUR_INSTALL_PATH/bin/` into `.bashrc`. 
## Usage
1. Put all required data into an excel file, name as e.g. input.xlsx. You can run `plot_phase_diagram --example` to get the example.xlsx.
2. Run `plot_phase_diagram.py input.xlsx [--probability threshold]`. "--probability" is optional, where "threshold" is float and means the minimum ratio for an exist phase. 
* Note: input.xlsx should has xls or xlsx postfix

## Output
1. G_result.xlsx
2. \*.dat or \*.csv
3. \*.vsz
4. \*.jpg, if you have veusz and xvfb installed
