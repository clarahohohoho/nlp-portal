import re
from itertools import chain

def NRIC_Finder(text):
    r = re.compile(r'\b[fgstFGST]\d{7}[a-zA-Z]', flags=re.I | re.X)
    NRIClist = r.findall(text)

    return NRIClist

def Phone_Finder(text):
    r = re.compile(r'\d{8}', flags=re.I | re.X)
    phonelist = r.findall(text)

    return phonelist