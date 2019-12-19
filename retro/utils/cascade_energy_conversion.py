#!/usr/bin/env python

"""
Functions for computing the hadronic factor given either the energy of an EM or hadronic cascade.

Modified only slightly from
sandbox/jpandre/multinest_icetray/private/multinest/HadronicFactorCorrection.cxx
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import numba


__all__ = ["NJIT_KW", "hf_vs_hadr_energy", "hf_vs_em_energy", "hadr2em", "em2hadr", "test"]


NJIT_KW = dict(nopython=True, cache=True, nogil=True, error_model='numpy', fastmath=False)

@numba.jit(**NJIT_KW)
def hf_vs_hadr_energy(hadr_energy):
    """Hadronic factor as a function of a hadronic cascade's energy.

    Parameters
    ----------
    hadr_energy

    Returns
    -------
    hadr_factor

    Notes
    -----
    C++ function originally named "HadronicFactorHadEM"

    """
    E0 = 0.18791678  # pylint: disable=invalid-name
    m = 0.16267529
    f0 = 0.30974123
    e = 2.71828183

    en = max(e, hadr_energy)
    return 1 - pow(en / E0, -m) * (1 - f0)


@numba.jit(**NJIT_KW)
def hf_vs_em_energy(em_energy):
    """Hadronic factor as function of the energy of an EM cascade.

    Parameters
    ----------
    em_energy

    Returns
    -------
    hadr_factor

    Notes
    -----
    C++ function was originally named "HadronicFactorEMHad"

    """
    precision = 1.0e-6

    # The hadr_factor is always between 0.5 and 1.0
    hadr_energy_max = em_energy / 0.5
    hadr_energy_min = em_energy / 1.0
    estimated_energy_em_max = hadr_energy_max * hf_vs_hadr_energy(hadr_energy_max)
    estimated_energy_em_min = hadr_energy_min * hf_vs_hadr_energy(hadr_energy_min)
    if em_energy < estimated_energy_em_min or em_energy > estimated_energy_em_max:
        print("Problem with boundary definition for hadronic factor calculation. Returning NaN.")
        return np.nan

    while (hadr_energy_max - hadr_energy_min) / hadr_energy_min > precision:
        hadr_energy_cur = (hadr_energy_max + hadr_energy_min) / 2.0
        estimated_energy_em_cur = hadr_energy_cur * hf_vs_hadr_energy(hadr_energy_cur)
        if estimated_energy_em_cur < em_energy:
            hadr_energy_min = hadr_energy_cur
            estimated_energy_em_min = estimated_energy_em_cur

        elif estimated_energy_em_cur > em_energy:
            hadr_energy_max = hadr_energy_cur
            estimated_energy_em_max = estimated_energy_em_cur

        else:  # estimated_energy_em_cur == em_energy
            return hf_vs_hadr_energy(hadr_energy_cur)

    hadr_factor = hf_vs_hadr_energy((hadr_energy_max + hadr_energy_min) / 2.0)

    return hadr_factor


@numba.vectorize
def hadr2em(hadr_energies):
    """Given a hadronic cascade's energy, find the energy of an EM cascade with
    roughly equivalent Cherenkov light output.

    Parameters
    ----------
    hadr_energies

    Returns
    -------
    em_energies

    """
    return hadr_energies * hf_vs_hadr_energy(hadr_energies)


@numba.vectorize
def em2hadr(em_energies):
    """Given an EM cascade's energy, find the energy of a hadronic cascade with
    roughly equivalent Cherenkov light output.

    Parameters
    ----------
    em_energies

    Returns
    -------
    hadr_energies

    """
    return em_energies / hf_vs_em_energy(em_energies)


def test():
    """Unit tests and debug plots for `hf_vs_em_energy` and
    `hf_vs_hadr_energy`"""
    import matplotlib

    matplotlib.use("agg")
    import matplotlib.pyplot as plt

    energies = np.logspace(-1, 4, 1000)

    # Interpret `energies` as those of EM showers
    hadr_factors_vs_em = np.array([hf_vs_em_energy(e) for e in energies])
    # round-trip test
    equiv_hadr_en = em2hadr(energies)
    equiv_em_en = hadr2em(equiv_hadr_en)
    assert np.allclose(equiv_em_en, energies)

    # Interpret `energies` as those of hadronic showers
    hadr_factors_vs_hadr = np.array([hf_vs_hadr_energy(e) for e in energies])
    # round-trip test
    equiv_em_en = hadr2em(energies)
    equiv_hadr_en = em2hadr(equiv_em_en)
    assert np.allclose(equiv_hadr_en, energies)

    fig, ax0 = plt.subplots(1, 1, figsize=(8, 5))
    # ax0, ax1 = axes
    ylabel = (
        r"$\bar F$"
        + " : Average relative luminosity of\n(hadr cascade) / (equally energetic EM cascade)"
    )

    color = "C0"
    ax0.plot(energies, hadr_factors_vs_em, color=color)
    ax0.set_xlabel("Energy of electromagnetic cascade (GeV)", color=color)
    ax0.tick_params(axis="x", labelcolor=color)

    ax1 = ax0.twiny()
    color = "C1"
    ax1.plot(energies, hadr_factors_vs_hadr, color=color, linestyle="--")
    ax1.set_xlabel("Energy of hadronic cascade (GeV)", color=color)
    ax1.tick_params(axis="x", labelcolor=color)

    for ax in [ax0, ax1]:
        ax.set_xscale("log")
        ax.set_xlim(energies[0], energies[-1])
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both")

    fig.tight_layout()
    fig.savefig("hadr_factor_vs_em_and_hadr_energy.pdf")


if __name__ == "__main__":
    test()
