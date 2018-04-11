import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save
import os 

for dirname, dirnames, filenames in os.walk('rewards/'):
    break

for filename in filenames:
    if filename[-3:] != "npy":
        filenames.remove(filename)

    

for filename in filenames:
    rewards = np.load("rewards/"+filename)
    fig = pd.DataFrame(rewards).rolling(50, center=False).mean().plot()
    plt.xlabel("Episodes")
    plt.ylabel("Recompenses")
    if (filename[6]=="1"):
        plt.title("OrnsteinUhlenbeckProcess"+filename[7:-4])
    elif (filename[6]=="2"):
        plt.title("NormalProcess"+filename[7:-4])
    elif (filename[6]=="3"):
        plt.title("UniformProcess"+filename[7:-4])
    fig.get_figure().savefig("plots/"+filename[:-4]+".png")
    tikz_save("tikz/"+filename[:-4]+".tex")

# plt.show()


