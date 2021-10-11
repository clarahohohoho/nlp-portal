import re
from itertools import chain
import spacy

def NRIC_Finder(text):
    r = re.compile(r'\b[fgstFGST]\d{7}[a-zA-Z]', flags=re.I | re.X)
    NRIClist = r.findall(text)

    return NRIClist

def Phone_Finder(text):
    r = re.compile(r'\d{8}', flags=re.I | re.X)
    phonelist = r.findall(text)

    return phonelist

def ner(text):

    ner_result = {'person': [], 'org':[]}


    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    for ent in doc.ents:

        if ent.label_ == 'PERSON' and ent.text not in ner_result['person']:
            ner_result['person'].append(ent.text)
        elif ent.label_ == 'ORG' and ent.text not in ner_result['org']:
            ner_result['org'].append(ent.text)

    return ner_result

def substringSieve(string_list):
    string_list.sort(key=lambda s: len(s), reverse=True)
    out = []
    for s in string_list:
        if not any([s in o for o in out]):
            out.append(s)
    return out