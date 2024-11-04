import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl

# Graphic style
fontsize = 12
font = {"family":"Dejavu Sans", "weight":"normal", "size":fontsize}
mpl.rc("font", **font)

__all__ = ["plot_lightcurve", "plot_lightcurves"]

def plot_lightcurves (obj, delay = 0, title = None, show_plot = True):
    """
    Function to plot light-curves with possible delay, considered it as a float.
    """

    _dir = dir(obj)

    if "LC1" in _dir:
        dict = obj.LC1.dict
    else:
        dict = obj.dict

    fig, ax = plt.subplots(figsize = (9, 6))
    ax.errorbar(dict["data"]["Dates"], dict["data"]["y"], yerr = dict["data"]["yerr"])
    ax.set_xlabel("Year")
    ax.set_ylabel(dict["name"])
    ax.grid()

    if "LC2" in _dir:
        dict = obj.LC2.dict
        color = "red"
        ax2 = ax.twinx()
        ax2.set_ylabel(dict["name"], color=color)
        ax2.errorbar(dict["data"]["Dates"] + datetime.timedelta(days = delay), dict["data"]["y"], yerr = dict["data"]["yerr"], color=color)
        ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    if title is not None:
        ax.set_title(title)

    if show_plot:
        plt.show()

    return fig, ax
