import matplotlib.pyplot as plt
import pandas as pd


def show_figure(list1):
    df = pd.DataFrame(list1, columns=["V", "A"])
    plt.figure(figsize=(20, 8), dpi=400)
    plt.plot(df["V"])
    plt.show()


def calculate_formula(probe_list, pheromone_list, beta, alpha):
    """
    :param probe_list: 每一个对应的probe_list
    :param pheromone_list: 每一个对应的荷尔蒙值
    :param beta: 先验的权重
    :param alpha: 后验的权重
    :return:
    """
    numerators = [(probe_value ** beta) * (pheromone ** alpha) for pheromone, probe_value in
                  zip(pheromone_list, eval(probe_list))]
    denominator = sum(numerators)
    results = [num / denominator if denominator != 0 else 0 for num in numerators]
    return results
