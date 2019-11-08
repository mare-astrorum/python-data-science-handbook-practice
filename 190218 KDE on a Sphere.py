#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 19:44:58 2019

@author: ai
"""

%matplotlib auto
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np


from sklearn.datasets import fetch_species_distributions

data = fetch_species_distributions()


latlon = np.vstack([data.train['dd lat'],
                    data.train['dd long']]).T
species = np.array([d.decode('ascii').startswith('micro')
                    for d in data.train['species']], dtype='int')

from mpl_toolkits.basemap import Basemap
from sklearn.datasets.species_distributions import construct_grids