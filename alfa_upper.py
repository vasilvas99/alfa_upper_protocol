import sys

from cli2gui import Cli2Gui
import numpy as np
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from argparse import ArgumentParser, BooleanOptionalAction


def johnson_mehl_avrami_kolmogorov(t, tau, n, t0):
    return 1 - np.exp(-((2 * (t - t0) / tau) ** n))


def fit_data(t, alfa, n_ini, tau_ini, t0_ini, fix_t0=False):
    # Find the maximum value of alfa
    alfamax = np.max(alfa)
    print(f"{n_ini=}, {tau_ini=}, {t0_ini}, {fix_t0=}")
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


def cli():
    parser = ArgumentParser("Calculate the JMAK n vs alphaup dependece")
    parser.add_argument(
        "source_path",
        help="Path to source data file",
    )
    parser.add_argument(
        "n_initial", type=float, help="Initial guess for n. Used for all fits"
    )
    parser.add_argument(
        "tau_initial", type=float, help="Initial guess for n. Used for all fits."
    )
    parser.add_argument(
        "t0_initial", type=float, help="Initial guess for t0. Used for all fits."
    )
    parser.add_argument(
        "--fix-t0",
        type=bool,
        action=BooleanOptionalAction,
        help="Whether or not to fix t0 to the initial guess",
    )
    parser = parser.parse_args()
    return parser


def run(args):
    # # Generate example dataset
    # t = np.linspace(0, 5, 100)  # 100 evenly spaced values from 0 to 5
    # alfa = np.tanh(2 * t) ** 2

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
    args = cli()
    run(args)


decorator_function = Cli2Gui(
    run_function=run,
    auto_enable=True,
)

gui = decorator_function(main)


if __name__ == "__main__":
    gui()