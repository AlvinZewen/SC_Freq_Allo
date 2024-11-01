Frequency Optimization and Yield Estimation

This repository contains two primary components for solving a frequency optimization problem, as discussed in our related paper on arXiv.

1. Frequency Optimization with GAMS (freq_opt.gams)

The GAMS code titled freq_opt.gams is designed to optimize frequency allocation based on the methods presented in the arXiv paper. 
This script helps find the optimal frequency configuration according to the constraints and objectives detailed in the paper.

Usage
To run the optimization:
Input
Prepare the required and optimization input files as described below.
The optimization requires the input of an undirected graph, which should be provided as an .inc file. Examples are in the 'maps' folder.
There is also an optional file for an initial guess, which can help speed up the optimization in some cases. Users may provide their own initial guesses if they have them, but this is not mandatory.

Run the freq_opt.gams file using GAMS:
***gams freq_opt.gams***

Output
The optimization will output the optimal frequencies (freqs.csv) for the nodes.


2. Yield Estimation with Python (get_yield.py)
The Python script get_yield.py assists in estimating the yield from a given frequency profile. This script calculates the estimated yield based on the output of the frequency optimization process.

***python get_yield.py freqs.csv***

note: to run the get_yield.py code, one has to generate the connectivity of the optimized graph. (extra code will be updated shortly)
