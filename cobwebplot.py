#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intented for Python 3.5

Define and test of cobweb plots.

 +------------------------------------------------------+
 |	Date of creation: 15 Jan. 2019						| 
 |	Last modif: 22 Mar. 2019							|	
 +------------------------------------------------------+
 |	CNRM - Meteo-France, CNRS (UMR 3589) 				|	
 |	GMEI/LISA 											|	
 +------------------------------------------------------+

Copyright (C) 2019  CNRM/GMEI/LISA Thomas Rieutord

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import time
import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt


def cobwebplot(
    X,
    Y,
    posLowest_negHighest=-1,
    n_threads=None,
    variablesNames=None,
    categorical_namesNvalues=None,
    titre=None,
    storeImages=False,
    fmtImages=".svg",
    figureDir="",
):
    """Draw a cobweb plot from the data. Only for categorical inputs.
	
	Cobweb plots are useful to highlight the influence of several inputs 
	onto an output. For few high/low outputs, the input values yielding
	to each output are linked by a thread. Repeated passage of the threads
	underline the "typical" values of some inputs parameters leading to
	such output. One the other hand, when the threads spread randomly along
	all possible value for an input, it shows that this input is not
	influential.
	
	[IN]
		- X (np.array[N,p]): matrix of inputs parameters
		- Y (np.array[N]): vector of outputs
		- posLowest_negHighest (int): Min or max output ? 1 gives the n_threads lowest. -1 gives the n_threads highest.
		- n_threads (int): number of threads to display. Default is 5% extreme.
		- variablesNames (list of str): names of all inputs + name of output (last one)
		- categorical_namesNvalues (dict): names and values of categorical inputs with their possible values, organized as {input_name:input_values}
		- titre (str): title of the plot
		- storeImages (opt, bool): if True, the figures are saved in the figureDir directory. Default is False
		- fmtImages (opt, str): format under which figures are saved when storeImages=True. Default is .svg
		- figureDir (opt, str): directory in which figures are saved when storeImages=True. Default is current directory.
	
	[OUT]
		- (matplotlib.pyplot figure): display the cobweb plot
			It has p+1 vertical bars, regularly spaced.
			In the X-axis are the p inputs parameters, one for each vertical bar. The last vertical bar is for the output
			In the Y-axis are the values of each inputs, normalised to range within the same bounds as the output.
	"""

    N, p = X.shape

    if n_threads is None:
        n_threads = int(N * 0.05)

    if variablesNames is None:
        variablesNames = ["X" + str(j + 1) for j in range(p)]
        variablesNames.append("Y")

    if categorical_namesNvalues is None:
        # By default, all variables are considered quantitative
        categorical_namesNvalues = {}
    if isinstance(categorical_namesNvalues, list):
        # Possibility to provide only the list of index of categorical variables (default naming)
        dex_catg = categorical_namesNvalues.copy()
        categorical_namesNvalues = {
            variablesNames[j]: np.unique(X[:, j]) for j in dex_catg
        }

    lowest_highest = {1: "lowest", -1: "highest"}
    if titre is None:
        titre = "Cobweb plot for " + " ".join(
            [str(n_threads), lowest_highest[posLowest_negHighest], variablesNames[-1]]
        )

    # Positions of vertical bars (regularly spaced)
    xPos = 2 * np.arange(p)

    # Positions of categorical variables on the Y-axis
    ymin = min(Y)
    ymax = max(Y)
    yPosText = {
        key: np.linspace(ymin, ymax, len(categorical_namesNvalues[key]) + 2)[1:-1]
        for key in categorical_namesNvalues.keys()
    }
    yPos = np.zeros((N, p))
    for j in range(p):
        if variablesNames[j] in categorical_namesNvalues.keys():
            yPos[:, j] = yPosText[variablesNames[j]][X[:, j].astype(int)]
        else:
            yPos[:, j] = (ymax - ymin) * (X[:, j] - min(X[:, j])) / (
                max(X[:, j]) - min(X[:, j])
            ) + ymin

    # Sorting the output value (if posLowest_negHighest=1 : ascending, if posLowest_negHighest=-1 decreasing)
    ordrered = np.argsort(posLowest_negHighest * Y)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    fig = plt.figure(figsize=(16, 10))

    # p input parameters
    plt.subplot2grid((1, p + 1), (0, 0), colspan=p)

    plt.title(titre)

    # Red (horizontalish) threads
    for i in range(n_threads):
        plt.plot(xPos, yPos[ordrered[i], :], "r-", alpha=0.1, linewidth=2)

    # Black vertical bars
    for j in range(p):
        plt.plot([xPos[j], xPos[j]], [ymin, ymax], "k-")
        if variablesNames[j] in categorical_namesNvalues.keys():
            for k in range(len(categorical_namesNvalues[variablesNames[j]])):
                plt.text(
                    xPos[j],
                    yPosText[variablesNames[j]][k],
                    str(categorical_namesNvalues[variablesNames[j]][k]),
                )
    plt.xticks(xPos, variablesNames, fontsize=20)
    ax = plt.gca()
    ax.yaxis.set_ticklabels([])

    # One output score
    plt.subplot2grid((1, p + 1), (0, p))
    plt.title("Output")
    plt.plot([0, 0], [min(Y), max(Y)], "k-", linewidth=2)
    plt.plot(np.zeros(n_threads), Y[ordrered[0:n_threads]], "ro")
    plt.xticks([0], [variablesNames[-1]], fontsize=20)
    ax = plt.gca()
    ax.yaxis.set_ticks_position("right")

    plt.show(block=False)
    if storeImages:
        plt.savefig(
            figureDir
            + "_".join(
                [
                    "cobweb_example-mixed",
                    lowest_highest[posLowest_negHighest],
                    variablesNames[-1],
                ]
            )
            + fmtImages
        )
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    return fig


def cobwebplot_quantitative(
    X,
    Y,
    posLowest_negHighest=-1,
    n_threads=None,
    variablesNames=None,
    titre=None,
    storeImages=False,
    fmtImages=".svg",
    figureDir="",
):
    """Draw a cobweb plot from the data. Only for quantitative inputs.
	
	Cobweb plots are useful to highlight the influence of several inputs 
	onto an output. For few high/low outputs, the input values yielding
	to each output are linked by a thread. Repeated passage of the threads
	underline the "typical" values of some inputs parameters leading to
	such output. One the other hand, when the threads spread randomly along
	all possible value for an input, it shows that this input is not
	influential.
	
	[IN]
		- X (np.array[N,p]): matrix of inputs parameters
		- Y (np.array[N]): vector of outputs
		- posLowest_negHighest (int): Min or max output ? 1 gives the n_threads lowest. -1 gives the n_threads highest.
		- n_threads (int): number of threads to display. Default is 5% extreme.
		- variablesNames (list of str): names of all inputs + name of output (last one)
		- titre (str): title of the plot
		- storeImages (opt, bool): if True, the figures are saved in the figureDir directory. Default is False
		- fmtImages (opt, str): format under which figures are saved when storeImages=True. Default is .svg
		- figureDir (opt, str): directory in which figures are saved when storeImages=True. Default is current directory.
	
	[OUT]
		- (matplotlib.pyplot figure): display the cobweb plot
			It has p+1 vertical bars, regularly spaced.
			In the X-axis are the p inputs parameters, one for each vertical bar. The last vertical bar is for the output
			In the Y-axis are the values of each inputs, normalised to range within the same bounds as the output.
	"""

    N, p = X.shape

    if n_threads is None:
        n_threads = int(N * 0.05)

    if variablesNames is None:
        variablesNames = ["X" + str(j + 1) for j in range(p)]
        variablesNames.append("Y")

    lowest_highest = {1: "lowest", -1: "highest"}
    if titre is None:
        titre = "Cobweb plot for " + " ".join(
            [str(n_threads), lowest_highest[posLowest_negHighest], variablesNames[-1]]
        )

    # Positions of vertical bars (regularly spaced)
    xPos = 2 * np.arange(p)

    # Positions of quantitative variables on the Y-axis
    ymin = min(Y)
    ymax = max(Y)
    yPos = np.zeros((N, p))
    for j in range(p):
        yPos[:, j] = (ymax - ymin) * (X[:, j] - min(X[:, j])) / (
            max(X[:, j]) - min(X[:, j])
        ) + ymin

    # Sorting the output value (if posLowest_negHighest=1 : ascending, if posLowest_negHighest=-1 decreasing)
    ordrered = np.argsort(posLowest_negHighest * Y)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    fig = plt.figure(figsize=(16, 10))

    # p input parameters
    plt.subplot2grid((1, p + 1), (0, 0), colspan=p)

    plt.title(titre)

    # Red (horizontalish) threads
    for i in range(n_threads):
        plt.plot(xPos, yPos[ordrered[i], :], "r-", alpha=0.1, linewidth=2)

    # Black vertical bars
    for j in range(p):
        plt.plot([xPos[j], xPos[j]], [ymin, ymax], "k-")
    plt.xticks(xPos, variablesNames, fontsize=20)
    ax = plt.gca()
    ax.yaxis.set_ticklabels([])

    # One output score
    plt.subplot2grid((1, p + 1), (0, p))
    plt.title("Output")
    plt.plot([0, 0], [min(Y), max(Y)], "k-", linewidth=2)
    plt.plot(np.zeros(n_threads), Y[ordrered[0:n_threads]], "ro")
    plt.xticks([0], [variablesNames[-1]], fontsize=20)
    ax = plt.gca()
    ax.yaxis.set_ticks_position("right")

    plt.show(block=False)
    if storeImages:
        plt.savefig(
            figureDir
            + "_".join(
                [
                    "cobweb_example-quantitative",
                    lowest_highest[posLowest_negHighest],
                    variablesNames[-1],
                ]
            )
            + fmtImages
        )
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    return fig


def cobwebplot_categorical(
    X,
    Y,
    posLowest_negHighest=-1,
    n_threads=None,
    variablesNames=None,
    inputs_namesNvalues=None,
    titre=None,
    storeImages=False,
    fmtImages=".svg",
    figureDir="",
):
    """Draw a cobweb plot from the data. Only for categorical inputs.
	
	Cobweb plots are useful to highlight the influence of several inputs 
	onto an output. For few high/low outputs, the input values yielding
	to each output are linked by a thread. Repeated passage of the threads
	underline the "typical" values of some inputs parameters leading to
	such output. One the other hand, when the threads spread randomly along
	all possible value for an input, it shows that this input is not
	influential.
	
	[IN]
		- X (np.array[N,p]): matrix of inputs parameters
		- Y (np.array[N]): vector of outputs
		- posLowest_negHighest (int): Min or max output ? 1 gives the n_threads lowest. -1 gives the n_threads highest.
		- n_threads (int): number of threads to display. Default is 5% extreme.
		- variablesNames (list of str): names of all inputs + name of output (last one)
		- inputs_namesNvalues (dict): names and values of all inputs organized as {input_name:input_values}
		- titre (str): title of the plot
		- storeImages (opt, bool): if True, the figures are saved in the figureDir directory. Default is False
		- fmtImages (opt, str): format under which figures are saved when storeImages=True. Default is .svg
		- figureDir (opt, str): directory in which figures are saved when storeImages=True. Default is current directory.
	
	[OUT]
		- (matplotlib.pyplot figure): display the cobweb plot
			It has p+1 vertical bars, regularly spaced.
			In the X-axis are the p inputs parameters, one for each vertical bar. The last vertical bar is for the output
			In the Y-axis are the values of each inputs, normalised to range within the same bounds as the output.
	"""

    N, p = X.shape

    if n_threads is None:
        n_threads = int(N * 0.05)

    if variablesNames is None:
        variablesNames = ["X" + str(j + 1) for j in range(p)]
        variablesNames.append("Y")

    if inputs_namesNvalues is None:
        inputs_namesNvalues = {variablesNames[j]: np.unique(X[:, j]) for j in range(p)}

    lowest_highest = {1: "lowest", -1: "highest"}
    if titre is None:
        titre = "Cobweb plot for " + " ".join(
            [str(n_threads), lowest_highest[posLowest_negHighest], variablesNames[-1]]
        )

    # Positions of vertical bars (regularly spaced)
    xPos = 2 * np.arange(p)

    # Positions of categorical variables on the Y-axis
    ymin = min(Y)
    ymax = max(Y)
    yPosText = {
        key: np.linspace(ymin, ymax, len(inputs_namesNvalues[key]) + 2)[1:-1]
        for key in inputs_namesNvalues.keys()
    }
    yPos = np.zeros((N, p))
    for j in range(p):
        yPos[:, j] = yPosText[variablesNames[j]][X[:, j]]

    # Sorting the output value (if posLowest_negHighest=1 : ascending, if posLowest_negHighest=-1 decreasing)
    ordrered = np.argsort(posLowest_negHighest * Y)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    fig = plt.figure(figsize=(16, 10))

    # p input parameters
    plt.subplot2grid((1, p + 1), (0, 0), colspan=p)

    plt.title(titre)

    # Red (horizontalish) threads
    for i in range(n_threads):
        plt.plot(xPos, yPos[ordrered[i], :], "r-", alpha=0.1, linewidth=2)

    # Black vertical bars
    for j in range(p):
        plt.plot([xPos[j], xPos[j]], [ymin, ymax], "k-")
        for k in range(len(inputs_namesNvalues[variablesNames[j]])):
            plt.text(
                xPos[j],
                yPosText[variablesNames[j]][k],
                str(inputs_namesNvalues[variablesNames[j]][k]),
            )
    plt.xticks(xPos, variablesNames, fontsize=20)
    ax = plt.gca()
    ax.yaxis.set_ticklabels([])

    # One output score
    plt.subplot2grid((1, p + 1), (0, p))
    plt.title("Output")
    plt.plot([0, 0], [min(Y), max(Y)], "k-", linewidth=2)
    plt.plot(np.zeros(n_threads), Y[ordrered[0:n_threads]], "ro")
    plt.xticks([0], [variablesNames[-1]], fontsize=20)
    ax = plt.gca()
    ax.yaxis.set_ticks_position("right")

    plt.show(block=False)
    if storeImages:
        plt.savefig(
            figureDir
            + "_".join(
                [
                    "cobweb_example-categorical",
                    lowest_highest[posLowest_negHighest],
                    variablesNames[-1],
                ]
            )
            + fmtImages
        )
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    return fig


if __name__ == "__main__":

    # Test for quantitative variables
    # =================================
    print("\n=== Quantitative variables ===")

    # Load test data
    # ----------------

    qttv = pd.read_csv("DATA_quantitative_inputs.txt", sep=" ")
    variablesNames = list(qttv.columns)
    X = qttv.iloc[:, :-1].values
    Y = qttv.iloc[:, -1].values

    N, p = X.shape
    print("Number of obs:", N, "Number of parameters:", p)
    print("variablesNames=", variablesNames)

    # Make cobweb plots
    # -------------------
    # Output vector is Y, input matrix is X.

    # Default plot
    cobwebplot_quantitative(X, Y)

    # More personalised plot
    cobwebplot_quantitative(
        X,
        Y,
        n_threads=80,
        variablesNames=variablesNames,
        storeImages=True,
        fmtImages=".png",
    )

    # Test for categorical variables
    # ================================
    print("\n=== Categorical variables ===")

    # Load test data
    # ----------------

    catg = pd.read_csv("DATA_categorical_inputs.txt", sep=" ")
    variablesNames = list(catg.columns)
    X = catg.iloc[:, :-1].values
    Y = catg.iloc[:, -1].values

    settingsValues = {}
    with open("DATA_categories.txt", "r") as f:
        for l in f.readlines():
            inp, rest = l.strip().split(":")
            settingsValues[inp] = rest.split(" ")

    N, p = X.shape
    print("Number of obs:", N, "Number of parameters:", p)
    print("variablesNames=", variablesNames)

    # Make cobweb plots
    # -------------------
    # Output vector is Y, input matrix is X.

    # Default plot
    cobwebplot_categorical(X, Y)

    # More personalised plot
    cobwebplot_categorical(
        X,
        Y,
        n_threads=100,
        variablesNames=variablesNames,
        inputs_namesNvalues=settingsValues,
        storeImages=True,
        fmtImages=".png",
    )

    # Test for mixed variables types
    # ================================
    print("\n=== Mixed variables types ===")

    # Load test data
    # ----------------

    mix = pd.read_csv("DATA_mixedtypes_inputs.txt", sep=" ")
    variablesNames = list(mix.columns)
    X = mix.iloc[:, :-1].values
    Y = mix.iloc[:, -1].values

    settingsValues = {}
    with open("DATA_categories.txt", "r") as f:
        for l in f.readlines():
            inp, rest = l.strip().split(":")
            settingsValues[inp] = rest.split(" ")

    N, p = X.shape
    print("Number of obs:", N, "Number of parameters:", p)
    print("variablesNames=", variablesNames)

    # Make cobweb plots
    # -------------------

    # Default plot
    cobwebplot(X, Y)

    # More personalised plot
    cobwebplot(
        X,
        Y,
        variablesNames=variablesNames,
        categorical_namesNvalues=settingsValues,
        storeImages=True,
        fmtImages=".png",
    )

    input("\n Press Enter to exit (close down all figures)\n")
