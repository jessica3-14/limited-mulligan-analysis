import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from scipy.stats import norm


import random
from datetime import date, timedelta
from requests import get
from json import loads
import json
import time
import math
import os


def gaussian_from_trunc(cutoff, kept_wr, emp_mull_rate):
    
    p = emp_mull_rate
    
    z = norm.ppf(p)                 # Φ^{-1}(p)
    phi = norm.pdf(z)               # φ(z)
    
    lam = phi / (1 - p)             # inverse Mills ratio
    
    sigma = (kept_wr - cutoff) / (lam - z)
    mu = cutoff - sigma * z
    
    return mu, sigma

print(str(gaussian_from_trunc(.389,.50,.1836)))
print(str(gaussian_from_trunc(.4147,.5465,.1895)))