import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Avg RTT DL (ms)','Avg RTT UL (ms)'
def aggregated(data):
    numeric_aggregated= data.groupby("MSISDN/Number").agg("Avg RTT DL (ms)":"mean",
                                                       "Avg RTT UL (ms)":"mean",
                                                       "Avg Bearer TP DL (kbps)": "mean",
                                                       "Avg Bearer TP UL (kbps)": "mean")
                                                       
    #find the most common handset type per msisdn
    handset= data.groupby("MSISDN/Number")['handset Type'].agg(pd.Series.mode).reset_index()
    merged_data=pd.merge(numeric_aggregated, handset, on="MSISDN/Number")
    return merged_data

    