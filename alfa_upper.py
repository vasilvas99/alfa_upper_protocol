import sys

import numpy as np
from pathlib import Path
from typing import Tuple
import PySimpleGUI as sg
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.optimize import curve_fit


@dataclass
class Configuration:
    source_path: Path
    n_initial: float
    tau_initial: float
    t0_initial: float
    fix_t0: bool
    delimiter: str


def johnson_mehl_avrami_kolmogorov(t, tau, n, t0):
    return 1 - np.exp(-((2 * (t - t0) / tau) ** n))


def fit_data(t, alfa, n_ini, tau_ini, t0_ini, fix_t0=False):
    # Find the maximum value of alfa
    alfamax = np.max(alfa)
    if fix_t0:
        print(f"Fixing t0 to {t0_ini}")
        p0 = [tau_ini, n_ini]
        fitting_func = lambda t, tau, n: johnson_mehl_avrami_kolmogorov(
            t, tau, n, t0_ini
        )
    else:
        p0 = [tau_ini, n_ini, t0_ini]
        fitting_func = johnson_mehl_avrami_kolmogorov

    # Define the range of alfa values to fit
    alfa_range = np.arange(0.2, alfamax + 0.001, 0.001)
    fitting_parameters = []
    for alpha_upper in alfa_range:
        # Filter data within the specified range
        mask = alfa <= alpha_upper
        t_fit = t[mask]
        alfa_fit = alfa[mask]
        # Perform curve fitting
        popt, _ = curve_fit(fitting_func, t_fit, alfa_fit, p0=p0)

        # Append the fitting parameters to the list
        fitting_parameters.append(popt)

    return fitting_parameters


def load_from_file(
    data_file_path: Path, delimiter="\t"
) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(data_file_path, delimiter=delimiter)
    t_data = np.copy(data[:, 0])
    alfa_data = np.copy(data[:, 1])
    return t_data, alfa_data


def parse_gui_input_values(values):
    try:
        path = values["-path-"]
        n_ini = values["-n-ini-"]
        tau_ini = values["-tau-ini-"]
        t0_ini = values["-t0-ini-"]
        fix_t0 = values["-fix-t0-"]
    except KeyError as ex:
        raise ValueError(f"Empty input field!")

    try:
        path = Path(path).resolve(True)
    except Exception as ex:
        raise ValueError(f"Could not resolve provided path!")

    if not path.is_file():
        raise ValueError(f"Could not resolve provided path!")

    try:
        n_ini = float(n_ini)
    except Exception as ex:
        raise ValueError(f"Could not parse n initial as a floating point number!")

    try:
        tau_ini = float(tau_ini)
    except Exception as ex:
        raise ValueError(f"Could not parse tau initial as a floating point number!")

    try:
        t0_ini = float(t0_ini)
    except Exception as ex:
        raise ValueError(f"Could not parse tau initial as a floating point number!")

    return Configuration(
        source_path=path,
        n_initial=n_ini,
        tau_initial=tau_ini,
        t0_initial=t0_ini,
        fix_t0=fix_t0,
        delimiter="\t",
    )


def run(args):
    # Load directly from provided path
    data_path = Path(args.source_path).resolve(True)
    t, alfa = load_from_file(data_path)

    # Fit the data and extract the fitting parameters
    parameters = fit_data(
        t,
        alfa,
        n_ini=args.n_initial,
        tau_ini=args.tau_initial,
        t0_ini=args.t0_initial,
        fix_t0=args.fix_t0,
    )

    # Extract tau and n values
    tau_values = [popt[0] for popt in parameters]
    n_values = [popt[1] for popt in parameters]

    # Generate alfaup values for the x-axis
    alfaup_values = np.arange(0.3, 0.3 + 0.001 * len(parameters), 0.001)

    # Plotting
    fig, ax1 = plt.subplots()

    # Plot n values
    color_n = "tab:red"
    ax1.set_xlabel("alfaup")
    ax1.set_ylabel("n", color=color_n)
    ax1.scatter(alfaup_values, n_values, color=color_n)
    ax1.tick_params(axis="y", labelcolor=color_n)

    # Create a second y-axis for tau values
    ax2 = ax1.twinx()

    # Plot tau values
    color_tau = "tab:blue"
    ax2.set_ylabel("tau", color=color_tau)
    ax2.scatter(alfaup_values, tau_values, color=color_tau)
    ax2.tick_params(axis="y", labelcolor=color_tau)

    # Set title and show the plot
    plt.title(f"Fitting Parameters as a Function of alfaup for:\n{data_path.name}")
    plt.show()


def main():
    sg.theme("Dark Grey 13")
    labels = sg.Column(
        [[sg.Text("N initial")], [sg.Text("tau initial")], [sg.Text("t0 initial")]]
    )
    inputs = sg.Column(
        [
            [sg.InputText(key="-n-ini-")],
            [sg.InputText(key="-tau-ini-")],
            [sg.InputText(key="-t0-ini-")],
        ]
    )
    layout = [
        [sg.Text("Data File Path (Tab-seperated list  for (t,alfa))")],
        [sg.Input(key="-path-"), sg.FileBrowse(key="-path-")],
        [labels, inputs],
        [sg.Checkbox("Check to fix t0 to initial value.", key="-fix-t0-")],
        [sg.OK(), sg.Button("Exit")],
    ]

    window = sg.Window("Alfa upper protocol for JMAK", layout)

    while True:
        event, values = window.read()
        if event == "OK":
            try:
                config = parse_gui_input_values(values)
                run(config)
            except ValueError as err:
                sg.popup_error(f"Failed to parse:\n{str(err)}")

        if event == "Exit" or event == sg.WIN_CLOSED:
            break
    window.close()


if __name__ == "__main__":
    main()
