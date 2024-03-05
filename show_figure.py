import matplotlib.pyplot as plt
import pandas as pd
def show_figure(list1):
    df = pd.DataFrame(list1,columns=["V","A"])
    plt.figure(figsize=(20,8),dpi=400)
    plt.plot(df["V"])
    plt.show()