import datetime
import numpy as np
import matplotlib.pyplot as plt

import os, sys
from utils.utils_convert import convert_datetime_to_MJD, convert_MJD_to_dates
from utils.utils_plot import plot_lightcurves
from utils.utils_math import compute_delay_matrix, compute_zDCF

from iminuit.cost import LeastSquares
from iminuit import Minuit

__all__ = ["Single", "Couple"]

class Single (object):
    """
    Class to handle more easily the single light-curve for the zDCF analysis.

    Arguments:
      - data_dict, dict,
        Dictionary with only two keys.
         - The first key is the "name" of the physical quantity.
         - The second one is the a "data" dictionary, consisting of the "Dates" (datetime.datetime), "MJD", "y" and "yerr" (floats) arrays

    Methods:
      - plot_LC,
        Plot the LC onto a matplotlib plot.
      - write_LC_file_for_zdcf_analysis,
        Arguments:
         - directory, str
           Path of the directory where to save stuff
         - suffix, str
           Possible suffix to identify some files. Default is None
      [TO BE COMPLETED...]

    """

    def __init__ (
            self,
            data_dict = None
        ):
        if data_dict is None:
            #print ("Please set the name and data using the set_LC method of the Single class.")
            self.dict = data_dict
        else:
            self.name = data_dict["name"]
            self.dict = data_dict
        return

    def set_LC (self, name, data):
        """
        Function to set the dict attribute by giving a
         - name, str
           This string will be placed as self.dict["name"]
         - data, dict
           This dictionary has to be given composed of 4 keys:
            - "Dates": an array of datetime.datetime objects
            - "MJD"  : an array of MJD floats
            - "y"    : an array of floats
            - "yerr" : an array of floats
           For the conversion between datetime.datetime objects and MJDs,
           the utils_read.convert_datetime_to_MJD function can be used.
           If the dictionary does not have the MJD key, it will be created.
        """
        if "Dates" in data.keys() and "MJD" not in data.keys():
            dates = data["Dates"]
            data["MJD"] = convert_datetime_to_MJD(dates)
        if "MJD" in data.keys() and "Dates" not in data.keys():
            mjds = data["MJD"]
            data["Dates"] = convert_MJD_to_dates (mjds)

        self.dict = {
            "name": name,
            "data": data,
        }
        return


    def plot_LC (self, show_plot = True):
        fig, ax = plot_lightcurves(self, show_plot = show_plot)
        return fig, ax


    def write_LC_file_for_zdcf_analysis (self, directory = "./", suffix = None):
        if suffix is None:
            suffix = self.dict["name"]
        else:
            suffix = self.dict["name"] + "_" + suffix
        outname = os.path.join(directory, "LC_%s.txt" % suffix)
        arr = np.transpose(np.array([self.dict["data"]["MJD"], self.dict["data"]["y"], self.dict["data"]["yerr"]]))
        np.savetxt(outname, arr, delimiter = ",")
        print ("Successfully saved MJD data to the file %s" % outname)
        return


class Couple (Single):
    """
    Class to handle more easily the couple of light-curves for the zDCF analysis.
    It has two attributes, which are LC1 and LC2, which are Single objects. Thus, you can extract things from them using the dict attribute.
    """

    def __init__ (self, data_dict_1 = None, data_dict_2 = None):
        self.LC1 = Single(data_dict_1)
        self.LC2 = Single(data_dict_2)
        self.nMCs = 50
        self.verbose = True
        self.zDCF = None
        self.correlation_params_dict = {
            "zDCF": {
                "equal-width binning":{
                    "max_delay": 1800,
                    "bin_step": 60,
                },
                "equal-population binning": {
                    "n": 50,
                },
             },
        }
        return

    def info(self):
        self.take_time_arrays (verbose = True)
        self.take_y_arrays (verbose = True)
        return

    def plot_LCs (self, show_plot = True):
        fig, ax = plot_lightcurves(self, show_plot = show_plot)
        return fig, ax

    def take_time_arrays (self, verbose = False):
        t_A = np.array(self.LC1.dict["data"]["MJD"])
        t_B = np.array(self.LC2.dict["data"]["MJD"])
        N_A, N_B = len(t_A), len(t_B)
        if verbose is True:
            print ("As an example, here are the first 10 MJD times of the two light-curves:")
            print (self.LC1.dict["name"], ":", t_A[:10], "MJD")
            print (self.LC2.dict["name"], ":", t_B[:10], "MJD")
            print (f"The {self.LC1.dict['name']} light-curves has {N_A} points from {t_A[0]:.3f} MJD to {t_A[-1]:.3f} MJD")
            print (f"The {self.LC2.dict['name']} light-curves has {N_B} points from {t_B[0]:.3f} MJD to {t_B[-1]:.3f} MJD")
            return
        return t_A, t_B, N_A, N_B

    def take_y_arrays (self, verbose = False):
        a, a_err = self.LC1.dict["data"]["y"], self.LC1.dict["data"]["yerr"]
        b, b_err = self.LC2.dict["data"]["y"], self.LC2.dict["data"]["yerr"]
        if verbose is True:
            print (f"The {self.LC1.dict['name']} light-curve has average {np.average(a):.2e} and standard deviation {np.std(a):.2e} over the entire light-curve.")
            print (f"The {self.LC2.dict['name']} light-curves has average {np.average(b):.2e} and standard deviation {np.std(b):.2e} over the entire light-curve.")
            return
        return a, a_err, b, b_err


    def perform_zDCF_analysis (
        self,
        nMCs = None,
        equal_population_binning = True,
        n = None,
        bins = None,
        max_delay = None,
        do_checks = True,
        show_plot = True,
        save_to_object_dict = True,
    ):
        """
        Function to compute the zDCF of the two light-curves constituting the Couple object.
        Arguments:
         - nMCs, float
         - equal_population_binning, bool
         - n, int
         - bins, array-like
         - max_delay, float
         - do_checks, bool
         - show_plot, bool
         - save_to_object_dict, bool
           Boolean flag i to sa
        Returns:
         - delay_vec, z_vec, delta_uncertainty_vec, z_uncertainty_vec
        """
        t_A, t_B, N_A, N_B = self.take_time_arrays()
        a, a_err, b, b_err = self.take_y_arrays()
        tau_ij = compute_delay_matrix(t_A, t_B)

        if equal_population_binning is True:
            if n is None:
                n = self.correlation_params_dict["zDCF"]["equal-population binning"]["n"]
            print (f"Computing zDCF with {n} events per time-lag bin...")
            ordered_delays = np.sort(tau_ij, axis = None)
            n_delay_bins = len(ordered_delays) // n
            print (f"The zDCF plot will have {n_delay_bins} bins")
            order_A = np.argsort(tau_ij, axis = None) // N_B
            order_B = np.argsort(tau_ij, axis = None) % N_B
            aa, bb, aa_err, bb_err = a[order_A], b[order_B], a_err[order_A], b_err[order_B]
            bins_center_vec, unc_bins_center_vec = np.zeros(n_delay_bins), np.zeros(n_delay_bins)
            r_vec, z_vec, unc_r_vec, unc_z_vec = np.zeros(n_delay_bins), np.zeros(n_delay_bins), np.zeros(n_delay_bins), np.zeros(n_delay_bins)
            for i_bin in range(n_delay_bins - 1):
                # Time
                tmin, tmax = ordered_delays[n * i_bin], ordered_delays[n * (i_bin + 1)]
                bin_center, unc_bin_center = (tmin + tmax) * 0.5, (tmax - tmin) * 0.5
                bins_center_vec[i_bin], unc_bins_center_vec[i_bin] = bin_center, unc_bin_center
                # Points
                a_bin, a_err_bin = aa[n * i_bin: n * (i_bin + 1)], aa_err[n * i_bin: n * (i_bin + 1)]
                b_bin, b_err_bin = bb[n * i_bin: n * (i_bin + 1)], bb_err[n * i_bin: n * (i_bin + 1)]
                n_ = len(a_bin)
                if n_ != n:
                    print (f"WARNING: there are {n_} elements in the {i_bin} bin, instead of {n}!")
                # Correlation coefficients
                z, r = compute_zDCF(a_bin, b_bin)
                """
                Computation of the uncertainty using nMCs simulations (i.e. extracting the flux values
                from a multi-normal distribution having expectation values and standard_deviations
                equal to those of the real data
                """
                if nMCs is None:
                    nMCs = self.nMCs
                zMC, rMC = np.zeros(nMCs), np.zeros(nMCs)
                for imc in range(nMCs):
                    new_a_bin = np.random.normal(a_bin, a_err_bin)
                    new_b_bin = np.random.normal(b_bin, b_err_bin)
                    zMC[imc], rMC[imc] = compute_zDCF(new_a_bin, new_b_bin)
                z_average, s_z, r_average, s_r = np.mean(zMC), np.std(zMC), np.mean(rMC), np.std(rMC)
                r_vec[i_bin], unc_r_vec[i_bin], z_vec[i_bin], unc_z_vec[i_bin] = r, s_r, z, s_z

        else:
            if bins is None:
                xmax = self.correlation_params_dict["zDCF"]["equal-width binning"]["max_delay"]
                bin_step = self.correlation_params_dict["zDCF"]["equal-width binning"]["bin_step"]
                bins = np.arange(-xmax - bin_step/2, xmax + bin_step/2, bin_step)
            bins_center_vec, unc_bins_center_vec = np.zeros(len(bins)-1), np.zeros(len(bins)-1)
            z_vec, unc_z_vec, r_vec, unc_r_vec = np.zeros(len(bins)-1), np.zeros(len(bins)-1), np.zeros(len(bins)-1), np.zeros(len(bins)-1)
            if do_checks:
                values, bins = np.histogram(tau_ij.reshape(N_A*N_B), bins = bins);
                if show_plot:
                    plt.stairs(values, bins, fill = True)
                    plt.xlabel("Delay")
                    plt.ylabel("Occurrences")
                    plt.grid()
                    plt.show()
                if np.count_nonzero(values < 10) > 0:
                    print ("WARNING: THERE ARE BINS WITH LESS THAN 10 ENTRIES. \n --> CHECK THE CHOSEN BINNING.")
            for i_bin in range(len(bins)-1):
                bin_center = (bins[i_bin] + bins[i_bin + 1]) * 0.5
                unc_bins_center = ( - bins[i_bin] + bins[i_bin + 1]) * 0.5
                bins_center_vec[i_bin], unc_bins_center_vec[i_bin] = bin_center, unc_bins_center
                mask = (tau_ij >= bins[i_bin]) * (tau_ij < bins[i_bin + 1])
                n = np.count_nonzero(mask == True)
                a_bin, a_err_bin = a[np.where(mask == True)[0]], a_err[np.where(mask == True)[0]]
                b_bin, b_err_bin = b[np.where(mask == True)[1]], b_err[np.where(mask == True)[1]]
                # Correlation coefficients
                z, r = compute_zDCF(a_bin, b_bin)
                """
                Computation of the uncertainty using nMCs simulations (i.e. extracting the flux values
                from a multi-normal distribution having expectation values and standard_deviations
                equal to those of the real data.
                """
                if nMCs is None:
                    nMCs = self.nMCs
                zMC, rMC = np.zeros(nMCs), np.zeros(nMCs)
                for imc in range(nMCs):
                    new_a_bin = np.random.normal(a_bin, a_err_bin)
                    new_b_bin = np.random.normal(b_bin, b_err_bin)
                    zMC[imc], rMC[imc] = compute_zDCF(new_a_bin, new_b_bin)
                z_average, s_z, r_average, s_r = np.mean(zMC), np.std(zMC), np.mean(rMC), np.std(rMC)
                r_vec[i_bin], unc_r_vec[i_bin], z_vec[i_bin], unc_z_vec[i_bin] = r, s_r, z, s_z

        if max_delay is not None:
            mask = np.abs(bins_center_vec) < max_delay
            bins_center_vec, unc_bins_center_vec = bins_center_vec[mask], unc_bins_center_vec[mask]
            z_vec, unc_z_vec = z_vec[mask], unc_z_vec[mask]

        self.zDCF = {
            "delay": bins_center_vec,
            "delay_err": unc_bins_center_vec,
            "zDCF": z_vec,
            "zDCF_err": unc_z_vec,
            "r": r_vec,
            "r_err": unc_r_vec,
        }
        if show_plot is True:
            self.zDCF_results_info()
            self.do_zdcf_plot()
        return bins_center_vec, z_vec, r_vec, unc_bins_center_vec, unc_z_vec, unc_r_vec


    def zDCF_results_info (self):
        if self.zDCF is None:
            print ("First you need to run the perform_zDCF_analysis method!")
            return
        delay, z_vec, unc_z_vec = self.zDCF["delay"], self.zDCF["zDCF"], self.zDCF["zDCF_err"]
        print ()
        print ("The point of maximum anticorrelation, i.e. the point with lowest zDCF, is found at ")
        print (f" - delay = {delay[z_vec.argmin()]:.2f} days,")
        print (f" - corresponding zDCF = {z_vec.min():.2f} +/- {unc_z_vec[z_vec.argmin()]:.2f}.")
        print ()
        print ("The point of maximum correlation, i.e. the point with highest zDCF, is found at ")
        print (f" - delay = {delay[z_vec.argmax()]:.2f} days,")
        print (f" - corresponding zDCF = {z_vec.max():.2f} +/- {unc_z_vec[z_vec.argmax()]:.2f}")
        return


    def do_zdcf_plot (self, r = False, fig = None, ax = None, show_plot = True):
        if self.zDCF is None:
            print ("First you need to run the perform_zDCF_analysis method!")
            return

        if r is False:
            quantity = "zDCF"
        else:
            quantity = "r"

        if fig is None:
            fig, ax = plt.subplots(figsize = (8, 6))
            ax.grid(which = "both", lw = 0.5)
            ax.set_xlabel("Delay [days]")
            ax.set_ylabel(quantity)

        ax.errorbar(
            x    = self.zDCF["delay"],
            y    = self.zDCF[quantity],
            xerr = self.zDCF["delay_err"],
            yerr = self.zDCF[quantity + "_err"],
            ls = "None",
        )

        if show_plot is True:
            plt.show()
        return fig, ax

    def do_maximum_anticorrelation_plot (self, fig = None, ax = None, show_plot = True):
        if self.zDCF is None:
            print ("First you need to run the perform_zDCF_analysis method!")
            return
        delay = self.zDCF["delay"][self.zDCF["zDCF"].argmin()]
        if ax is None:
            fig, ax = plot_lightcurves(self, delay = delay, show_plot = show_plot)
        if show_plot is True:
            plt.show()
        return fig, ax

    def do_maximum_correlation_plot (self, fig = None, ax = None, show_plot = True):
        if self.zDCF is None:
            print ("First you need to run the perform_zDCF_analysis method!")
            return
        delay = self.zDCF["delay"][self.zDCF["zDCF"].argmax()]
        if ax is None:
            fig, ax = plot_lightcurves(self, delay = delay, show_plot = show_plot)
        if show_plot is True:
            plt.show()
        return fig, ax

    def write_zDCF_result_to_file (self, filename = None, output_dir = "."):
        """
        Function to write the zDCF results into a txt file, formatted as "Delay [days]\tDelay uncertainty [days]\tzDCF \tzDCF uncertainty".

        Arguments:
         - filename, str, default = None
         - output_dit, str, default = "."

        """
        if self.zDCF is None:
            print ("First you need to run the perform_zDCF_analysis method!")
            return
        x, y, xerr, yerr = self.zDCF["delay"], self.zDCF["zDCF"], self.zDCF["delay_err"], self.zDCF["zDCF_err"]

        if filename is None:
            filename = f"zDCF_analysis_{self.LC1.dict['name']}_{self.LC2.dict['name']}.txt"
        filename = os.path.join(
            output_dir,
            filename,
        )
        np.savetxt(
            filename,
            np.array([x, xerr, y, yerr]).T,
            header = "Delay [days]\tDelay uncertainty [days]\tzDCF \tzDCF uncertainty",
            delimiter = "\t",
            fmt = "%.6f",
        )
        print (f"Successfully saved file {filename}!")
        return

    def take_ZDCF_bin_with_minimum_or_maximum_correlation (self, min_or_max = "max"):
        """
        Function to retrieve the delay bin (and corresponding zDCF + uncertainty) in the zDCF computation 
        corresponding to the min_or_max (could be "min" or "max") correlation point.
        Arguments:
         - min_or_max, str
          Could be "min" or "max"
        Returns:
         - d, z, unc_z
          Floats, they're the delay at which the zDCF is at the min_or_max.
          z and unc_z represent the corresponding zDCF value and uncertainty.
        """
        delay, z_vec, unc_z_vec = (
            self.zDCF["delay"], self.zDCF["zDCF"], self.zDCF["zDCF_err"]
        )
        if min_or_max == "min":
            d, z, unc_z = delay[z_vec.argmin()], z_vec.min(), unc_z_vec[z_vec.argmin()]
        else:
            d, z, unc_z = delay[z_vec.argmax()], z_vec.max(), unc_z_vec[z_vec.argmax()]
        return d, z, unc_z

    def analysis (
        self,
        max_or_min = "max",
        fit_region_max_percent_of_peak = 0.35,
        fit_region_max_delay_from_peak = 1000,
        second_order = False,
        zDCF_uncertainty_enhancement = 1,
        prefix = None,
        outdir = "./",
        plot_dir = "./",
    ):
        """
        Function to estimate the delay (+ uncertainty) at which the zDCF correlation parameter is minimum or maximum,
        together with the value (+ uncertainty) at the minimum or maximum.
        The estimate is done through the fit of the minimum or maximum with a third-order (or second-order) polinomial function.
        Fit function (x is the delay): f(x) = p0 + p1 * (x - x0) + p2 * (x - x0)**2 + p3 * (x - x0)**3
        Fit parameters:
         1) x0: the delay at which the minimum or maximum occurs
         2) p0 = f(x0): the value of the zDCF corresponding to x0
         3) p1 = 0: fixed to 0 because the fit showed there is always no need for this term.
         4) p2: parameter describing the degree of aperture of the parabola of the function around the minimum/maximum
         5) p3: parameter describing the asymmetry of the function used to describe the minimum/maximum
        The fit is done through MINUIT.
        The uncertainties are computed by fully propagating the error using the covariance matrix returned from the fit.

        Arguments:
         - self, object of the Couple class
          Couple object for which the perform_zDCF_analysis was already run.
         - max_or_min, str
          Can be "min" or "max".
          If "min" ("max"), the fit region is selected to be around the bin having the lowest (highest) zDCF value.
          Please take care that the lowest/highest zDCF value is not at the extremes of the delay array to which the zDCF is computed.
          In case it is, please rerun the perform_zDCF_analysis tuning the max_delay argument.
         - fit_region_max_percent_of_peak, float
          It is necessary to select the fit region. With respect to the zDCF value of the peak, the first requirement is that
          the fit regions extends to the surrounding region with zDCF > zDCF_peak * (1 - fit_region_max_percent_of_peak)
         - fit_region_max_delay_from_peak, float
          It is necessary to select the fit region. With respect to the delay D value at the peak, the second requirement is that
          the fit region extends up to [D - fit_region_max_delay_from_peak, D + fit_region_max_delay_from_peak]
         - second_order, bool
          If the peak of the zDCF plot is symmetric with respect to the center, it is possible to set p3 to 0 and perform the fit.
         - zDCF_uncertainty_enhancement, float,
          Sometimes the zDCF values have small statistical uncertainties. They can be enhanced by a factor 
          zDCF_uncertainty_enhancement to account for further systematics effects.
        """
        if self.zDCF is None:
            raise Exception ("No zDCF was computed for the given Couple object. Please run it with the perform_zDCF_analysis method.")
            return

        fig, ax = plt.subplots(figsize = (6, 5))
        ax.errorbar(
            x = self.zDCF["delay"],
            y = self.zDCF["zDCF"],
            xerr = self.zDCF["delay_err"],
            yerr = self.zDCF["zDCF_err"] * zDCF_uncertainty_enhancement,
            ls = "None",
            marker = "o",
            ms = 4,
        )

        d, z, unc_z = self.take_ZDCF_bin_with_minimum_or_maximum_correlation(max_or_min)
        if max_or_min == "max":
            mask1 = (self.zDCF["zDCF"] > z * (1 - fit_region_max_percent_of_peak))
            ax.set_ylim(0, z + 0.2)
            y_displ = 0.1
        else:
            mask1 = (self.zDCF["zDCF"] < z * (1 - fit_region_max_percent_of_peak))
            ax.set_ylim(z - 0.2, 0)
            y_displ = 0.5

        mask2 = np.abs(self.zDCF["delay"] - d) < fit_region_max_delay_from_peak 
        mask_fit = mask1 * mask2
        x1, x2 = self.zDCF["delay"][mask_fit][0], self.zDCF["delay"][mask_fit][-1]
        xx = np.linspace(x1, x2, 100)
        lsq = LeastSquares(
            self.zDCF["delay"][mask_fit],
            self.zDCF["zDCF"][mask_fit],
            self.zDCF["zDCF_err"][mask_fit] * zDCF_uncertainty_enhancement,
            third
        )
        lsq.loss = "soft_l1"
        m = Minuit(lsq, x0 = d, p0 = z, p1 = 0., p2 = -0.5, p3 = 0.)
        m.fixed["p1"] = True
        fit_type = "third"
        if second_order:
            m.fixed["p3"] = True
            fit_type = "second"
        #m.limits = [(1, 1.e4), (5.79e-8, 5.82e-8), (0.1e-10, 1e-8)]
        m.migrad()
        m.minos()
        yy = third(xx, *m.values)
        # Calcolo incertezza:
        x0, p0, p1, p2, p3 = m.values
        A = np.array(
            [
                -1. * (p1 + 2*p2*(xx - x0) + 3*p3*(xx - x0)**2),
                np.ones(len(xx)),
                (xx - x0),
                (xx - x0)**2,
                (xx - x0)**3
            ],
            dtype = object
        ).T
        covar_res = (A.dot(m.covariance).dot(A.T))
        uncertainty = np.array([covar_res[i][i]**0.5 for i in range(len(xx))])
        # Plot
        ax.errorbar(xx, yy, label = "Fit")
        ax.fill_between(xx, yy - uncertainty, yy + uncertainty, color = "peachpuff")
        ax.grid()
        ax.set_xlim(x1 - 300, x2 + 300)
        ax.set_xlabel("Delay")
        ax.set_ylabel("zDCF")
        # Print results on plot
        string = "$\chi^2$ / DOF = %.1f / %d = %.1f\n" % (m.fval, m.ndof, m.fval/m.ndof)
        string += "D = %d +/- %d days\n" % (m.values["x0"], m.errors["x0"])
        string += "zDCF (D) = %.2f +/- %.2f" % (m.values["p0"], m.errors["p0"])
        bbox={"facecolor":"white", "edgecolor":"white"}
        ax.text(x1+(x2-x1)*0.2, np.sign(z)*y_displ, string, size = 14, bbox = bbox);
        # Save plot
        ###
        name = "Fit_"
        name += disk_vs_p.LC2.dict["name"] + "_vs_" + disk_vs_p.LC1.dict["name"]
        name += "_" + max_or_min + "_" + fit_type
        if prefix is not None:
            name += "_" + prefix
        file_name = os.path.join(outdir, name + ".txt")
        np.savetxt(
            file_name,
            np.array([m.values, m.errors, m.fixed], dtype = object).T, 
            header = "Values\tUncertainty\tFixed (the five parameters are x0, p0, p1, p2, p3)",
            fmt = "%.2e",
        )
        print ("Successfully saved", file_name)
        figure_name = os.path.join(plot_dir, name + ".png")
        fig.savefig(figure_name)
        print ("Successfully saved", figure_name)
        print (m)
        return m
