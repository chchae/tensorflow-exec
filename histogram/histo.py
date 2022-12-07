import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns


#with open( './gbsa-result.csv', newline='' ) as csvfile :
#    csvreader = csv.reader( csvfile, delimiter=',', quotechar='|' )

df = pd.read_csv( './histogram/score.csv' )
print( df )

df_all = pd.read_csv( './histogram/score_all.csv' )
print( df_all )


def plot_dockE( data, data_all, title ):
    # bins = np.arange( np.floor( min(data) ), np.ceil( max(data) ), 0.5 )
    bins = np.arange( np.floor( min(data) ), 0, 0.5 )
    plt.hist( data_all, bins, histtype='step' )
    plt.hist( data, bins, histtype='step' )
    plt.yscale( 'log' )
    plt.title( title )
    plt.show()
    plt.savefig( 'histo.png' )

plt.clf()
plot_dockE( df['docking score'], df_all['r_i_docking_score'], 'docking score' )

