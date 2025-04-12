#!/usr/bin/env python3
"""
IZV cast1 projektu
Autor: Ondřej Bahounek (xbahou00)

Detailni zadani projektu je v samostatnem projektu e-learningu.
Nezapomente na to, ze python soubory maji dane formatovani.

Muzete pouzit libovolnou vestavenou knihovnu a knihovny predstavene na prednasce
"""

from bs4 import BeautifulSoup
import requests
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from typing import List, Callable, Dict, Any
import html
import re

def integrate(f: Callable[[NDArray], NDArray], a: float, b: float, steps=1000) -> float:
    x_values = np.linspace(a, b, steps)

    # X_i - X_{i-1}
    dy = np.diff(x_values)

    # ( X_{i-1} + X_i ) / 2
    function_arguments = (x_values[1:] + x_values[:-1]) / 2

    # f( ( X_{i-1} + X_i ) / 2 )
    function_values = f(function_arguments)

    # X_i - X_{i-1} * f( ( X_{i-1} + X_i ) / 2 )
    square_area = dy * function_values
    return square_area.sum()


def generate_graph(a: List[float], show_figure: bool = False, save_path: str | None = None):
    STEPS = 1000
    x_values = np.linspace(-3, 3, STEPS)

    # compute function value
    f = np.power(np.array(a).reshape(3, 1), 2) * np.power(x_values, 3) * np.sin(x_values)

    plt.figure(figsize=(6, 4))

    # plot graphs
    plt.plot(f[0], color="blue")
    plt.plot(f[1], color="orange")
    plt.plot(f[2], color="limegreen")

    # fill under the graph
    plt.fill_between(np.arange(0, STEPS, 1), f[0], color="blue", alpha=0.5)
    plt.fill_between(np.arange(0, STEPS, 1), f[1], color="sandybrown", alpha=0.5)
    plt.fill_between(np.arange(0, STEPS, 1), f[2], color="lightgreen", alpha=0.5)

    # compute integral values
    integrals = np.trapz(f, x=x_values).round(2)
    plt.text(STEPS, f[0][STEPS - 1], r'$\int f_{1.0}f(x)dx=' + str(integrals[0]) + '$', fontsize="small")
    plt.text(STEPS, f[1][STEPS - 1], r'$\int f_{1.5}f(x)dx=' + str(integrals[1]) + '$', fontsize="small")
    plt.text(STEPS, f[2][STEPS - 1], r'$\int f_{2.0}f(x)dx=' + str(integrals[2]) + '$', fontsize="small")

    # set graph boundaries
    plt.xlim(0, 1333)
    plt.ylim(0, 40)

    # name the axes
    plt.xlabel("x")
    plt.ylabel(r'$f_a(X)$')

    plt.legend([r'$y_{1.0}(x)$', r'$y_{1.5}(x)$', r'$y_{2.0}(x)$'], loc="upper center", bbox_to_anchor=(0.5, 1.13),
               ncols=3)

    # change the ticks to proper format
    x_ticks = np.linspace(0, STEPS, 7)
    x_tick_labels = np.arange(-3, 4, 1)
    plt.xticks(x_ticks, x_tick_labels)

    if show_figure:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()


def generate_sinus(show_figure: bool = False, save_path: str | None = None):
    STEPS = 10_000
    zero_to_hundred = np.linspace(0, 100, STEPS)

    # compute values
    f1_values = 0.5 * np.cos((np.pi * zero_to_hundred) / 50)
    f2_values = 0.25 * (np.sin(np.pi * zero_to_hundred) + np.sin(1.5 * np.pi * zero_to_hundred))
    f3_values = f1_values + f2_values

    # third graph is plot as two graphs
    mask = f3_values >= f1_values
    f3_green = np.where(mask, f3_values, np.nan)
    f3_red = np.where(~mask, f3_values, np.nan)

    # generate ticks
    x_ticks = np.linspace(0, STEPS, 6)
    x_labels = np.arange(0, 101, 20)
    y_ticks = np.linspace(-0.8, 0.8, 5)

    fig, ax = plt.subplots(3, 1, figsize=(12, 10))
    plt.setp(ax, xlim=(0, STEPS), ylim=(-0.8, 0.8), xticks=x_ticks, xticklabels=x_labels, yticks=y_ticks, xlabel="t")
    ax[0].plot(f1_values)
    ax[1].plot(f2_values)
    ax[2].plot(f3_green, color="green")
    ax[2].plot(f3_red, color="red")

    # set label for each individually
    plt.sca(ax[0])
    plt.ylabel(r'$f_1(t)$')
    plt.sca(ax[1])
    plt.ylabel(r'$f_2(t)$')
    plt.sca(ax[2])
    plt.ylabel(r'$f_1(t) + f_1(t)$')

    if show_figure:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()
def download_data() -> List[Dict[str, Any]]:
    url = "https://ehw.fit.vutbr.cz/izv/st_zemepis_cz.html"
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')

    ret_list = []

    elements_with_class = soup.find_all(class_='nezvyraznit')
    for element in elements_with_class:
        td_tags = element.find_all("td")

        position = td_tags[0].find('strong').text
        lat = float(td_tags[2].text.replace('°', '').replace(',', '.'))
        long = float(td_tags[4].text.replace('°', '').replace(',', '.'))
        height_text = html.unescape(td_tags[6].text)
        height = float(re.sub(r'&nbsp;|\s', '', height_text).replace(',', '.'))

        ret_list.append({'position': position, 'lat': lat, 'long': long, 'height': height})
    return ret_list
