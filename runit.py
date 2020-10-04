import pandas as pd
from flask import Flask, jsonify, request
from urllib.request import urlopen
from html.parser import HTMLParser
import bert_run


def compit(inputs):    
    txt_input = inputs
    toxic1, severe_toxic1, obsence1, threat1, insult1, identity_hate1 = bert_run.rate_toxic(txt_input)

    toxic=str(toxic1[0])
    severe_toxic=str(severe_toxic1[0])
    obsence=str(obsence1[0])
    threat=str(threat1[0])
    insult=str(insult1[0])
    identity_hate=str(identity_hate1[0])
    print(toxic, severe_toxic, obsence, threat, insult, identity_hate)
    
    return("Toxicity: "+ toxic+", "+"Obscenity: "+obsence+", "+"Insult Level: "+ insult+", "+"Identity Hate: "+identity_hate+", "+"Threat: "+ threat)

compit('you suck, stupid')