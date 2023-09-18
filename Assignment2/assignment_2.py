
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class bcolors:
    ENDC = '\033[m'
    BOLD = '\033[1m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    
def colorit(text: str = '',color: str = '', verbose: bool = True):
    if verbose:
        print(color + text + bcolors.ENDC)
        