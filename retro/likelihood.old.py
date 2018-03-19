# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Define likelihood functions used in Retro with the various kinds of tables we
have generated.
"""

from __future__ import absolute_import, division, print_function

__all__ = ['get_neg_llh']

__author__ = 'P. Eller, J.L. Lanfranchi'
__license__ = '''Copyright 2017 Philipp Eller and Justin L. Lanfranchi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.'''

from itertools import izip
from os.path import abspath, dirname
import sys

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(abspath(__file__)))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.utils.stats import poisson_llh


#@profile
def get_neg_llh(
        pinfo_gen, event, dom_tables, noise_charge=0, tdi_table=None,
        detailed_info_list=None
    ):
    """Get log likelihood.

    Parameters
    ----------
    pinfo_gen

    event : retro.Event namedtuple or convertible thereto

    dom_tables

    tdi_table

    detailed_info_list : None or appendable sequence
        If a list is provided, it is appended with a dict containing detailed
        info from the calculation useful, e.g., for debugging. If ``None`` is
        passed, no detailed info is made available.

    Returns
    -------
    llh : float
        Negative of the log likelihood

    """
    neg_llh = 0
    noise_counts = 0

    if tdi_table is not None:
        total_expected_q = tdi_table.get_photon_expectation(pinfo_gen=pinfo_gen)
    else:
        total_expected_q = 0

    expected_q_accounted_for = 0

    # Loop over pulses (aka hits) to get likelihood of those hits coming from
    # the hypo
    for string, depth_idx, pulse_time, pulse_charge in izip(*event.pulses):
        expected_charge = dom_tables.get_photon_expectation(
            pinfo_gen=pinfo_gen,
            hit_time=pulse_time,
            string=string,
            depth_idx=depth_idx
        )

        expected_charge_excluding_noise = expected_charge

        if expected_charge < noise_charge:
            noise_counts += 1
            # "Add" in noise (i.e.: expected charge must be at least as
            # large as noise level)
            expected_charge = noise_charge

        # Poisson log likelihood (take negative to interface w/ minimizers)
        pulse_neg_llh = -poisson_llh(expected=expected_charge,
                                     observed=pulse_charge)

        neg_llh += pulse_neg_llh
        expected_q_accounted_for += expected_charge

    # Penalize the likelihood (_add_ to neg_llh) by expected charge that
    # would be seen by DOMs other than those hit (by the physics event itself,
    # i.e. non-noise hits). This is the unaccounted-for excess predicted by the
    # hypothesis.
    unaccounted_excess_expected_q = total_expected_q - expected_q_accounted_for
    if tdi_table is not None:
        if unaccounted_excess_expected_q > 0:
            print('neg_llh before correction    :', neg_llh)
            print('unaccounted_excess_expected_q:',
                  unaccounted_excess_expected_q)
            neg_llh += unaccounted_excess_expected_q
            print('neg_llh after correction     :', neg_llh)
            print('')
        else:
            print('WARNING!!!! DOM tables account for %e expected charge, which'
                  ' exceeds total expected from TDI tables of %e'
                  % (expected_q_accounted_for, total_expected_q))
            #raise ValueError()

    # Record details if user passed a list for storing them
    if detailed_info_list is not None:
        detailed_info = dict(
            noise_counts=noise_counts,
            total_expected_q=total_expected_q,
            expected_q_accounted_for=expected_q_accounted_for,
        )
        detailed_info_list.append(detailed_info)

    #print('time to compute likelihood: %.5f ms' % ((time.time() - t0) * 1000))
    return neg_llh
