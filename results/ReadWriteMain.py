# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 08:04:03 2024

@author: Martin Voigt Vejling
Email: mvv@es.aau.dk
"""

import matplotlib.pyplot as plt
import pickle


if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-darkgrid")

    with open("Main/Pos002/Pos002.pickle", "rb") as file:
        res = pickle.load(file)

    for i, key in enumerate(list(res.keys())):
        with open(f"Main/Pos002/res{i}.pickle", "wb") as file:
            pickle.dump(res[f"{key}"], file, pickle.HIGHEST_PROTOCOL)
