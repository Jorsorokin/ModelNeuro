"""
neuron_test.py

A test for the model neuron classes "Neuron" and "Stim"
"""

# imports...located in /Users/Jordan/Git/ModelNeuro
import os
import profile

os.chdir('/Users/Jordan/Git/ModelNeuro')
from neuron import Neuron
from stim import Stim

# create our neuron
S = Stim(500,100,0)
S.add_current(100,300,10)
N = Neuron(500,100,-70)
N.add_stim(S)
N.simulate()
N.plot_results()
N.print_model()