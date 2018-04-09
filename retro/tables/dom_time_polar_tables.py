# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Class for single-DOM 3D (t, r, costheta) tables (i.e., no directionality).
"""

from __future__ import absolute_import, division, print_function

__all__ = '''
    RETRO_DOM_TABLE_FNAME_PROTO
    RETRO_DOM_TABLE_FNAME_RE
    load_t_r_theta_table
    DOMTimePolarTables
'''.split()

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

from os.path import abspath, dirname, isdir, join
import re
import sys

import numpy as np

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.const import DC_DOM_QUANT_EFF, IC_DOM_QUANT_EFF
from retro.retro_types import RetroPhotonInfo, TimeSphCoord
from retro.tables.pexp_t_r_theta import pexp_t_r_theta
from retro.utils.misc import expand, force_little_endian


RETRO_DOM_TABLE_FNAME_PROTO = [
    (
        'retro_nevts1000'
        '_{string:s}'
        '_DOM{depth_idx:d}'
        '_r_cz_t_angles.fits'
    ),
    (
        'retro_dom_table'
        '_set_{hash_val:s}'
        '_string_{string}'
        '_depth_{depth_idx:d}'
        '_seed_{seed}'
        '.fits'
    )
]
"""String templates for single-DOM "final-level" retro tables"""

RETRO_DOM_TABLE_FNAME_RE = [
    re.compile(
        r'''
        retro_nevts1000
        _(?P<string>[0-9a-z]+)
        _DOM(?P<depth_idx>[0-9]+)
        _r_cz_t_angles\.fits.*
        ''', re.IGNORECASE | re.VERBOSE
    ),
    re.compile(
        r'''
        retro_dom_table
        _set_(?P<hash_val>[0-9a-f]+)
        _string_(?P<string>[0-9a-z]+)
        _depth_(?P<depth_idx>[0-9]+)
        _seed_(?P<seed>[0-9]+)
        \.fits.*
        ''', re.IGNORECASE | re.VERBOSE
    )
]
"""Regex for single-DOM retro tables"""


def load_t_r_theta_table(fpath, depth_idx, scale=1, exponent=1,
                         photon_info=None):
    """Extract info from a file containing a (t, r, theta)-binned Retro table.

    Parameters
    ----------
    fpath : string
        Path to FITS file corresponding to the passed ``depth_idx``.

    depth_idx : int
        Depth index (e.g. from 0 to 59)

    scale : float
        Scaling factor to apply to the photon survival probability from the
        table, e.g. for quantum efficiency. This is applied _before_
        `exponent`. See `Notes` for more info.

    exponent : float >= 0, optional
        Modify probabilties in the table by ``prob = 1 - (1 - prob)**exponent``
        to allow for up- and down-scaling the efficiency of the DOMs. This is
        applied to each DOM's table _after_ `scale`. See `Notes` for more
        info.

    photon_info : None or RetroPhotonInfo namedtuple of dicts
        If None, creates a new RetroPhotonInfo namedtuple with empty dicts to
        fill. If one is provided, the existing component dictionaries are
        updated.

    Returns
    -------
    photon_info : RetroPhotonInfo namedtuple of dicts
        Tuple fields are 'survival_prob', 'theta', 'phi', and 'length'. Each
        dict is keyed by `depth_idx` and values are the arrays loaded
        from the FITS file.

    bin_edges : TimeSphCoord namedtuple
        Each element of the tuple is an array of bin edges.

    Notes
    -----
    The parameters `scale` and `exponent` modify a table's probability `P` by::

        P = 1 - (1 - P*scale)**exponent

    This allows for `scale` (which must be from 0 to 1) to be used for e.g.
    quantum efficiency--which always reduces the detection probability--and
    `exponent` (which must be 0 or greater) to be used as a systematic that
    modifies the post-`scale` probabilities up and down while keeping them
    valid (i.e., between 0 and 1). Larger values of `scale` (i.e., closer to 1)
    indicate a more efficient DOM. Likewise, values of `exponent` greater than
    one scale up the DOM efficiency, while values of `exponent` between 0 and 1
    scale the efficiency down.

    """
    # pylint: disable=no-member
    import pyfits

    assert 0 <= scale <= 1
    assert exponent >= 0

    if photon_info is None:
        empty_dicts = []
        for _ in RetroPhotonInfo._fields:
            empty_dicts.append({})
        photon_info = RetroPhotonInfo(*empty_dicts)

    with pyfits.open(expand(fpath)) as table:
        data = force_little_endian(table[0].data)

        if scale == exponent == 1:
            photon_info.survival_prob[depth_idx] = data
        else:
            photon_info.survival_prob[depth_idx] = (
                1 - (1 - data * scale)**exponent
            )

        photon_info.theta[depth_idx] = force_little_endian(table[1].data)

        photon_info.deltaphi[depth_idx] = force_little_endian(table[2].data)

        photon_info.length[depth_idx] = force_little_endian(table[3].data)

        # Note that we invert (reverse and multiply by -1) time edges; also,
        # no phi edges are defined in these tables.
        data = force_little_endian(table[4].data)
        t = - data[::-1]

        r = force_little_endian(table[5].data)

        # Previously used the following to get "agreement" w/ raw photon sim
        #r_volumes = np.square(0.5 * (r[1:] + r[:-1]))
        #r_volumes = (0.5 * (r[1:] + r[:-1]))**2 * (r[1:] - r[:-1])
        r_volumes = 0.25 * (r[1:]**3 - r[:-1]**3)

        photon_info.survival_prob[depth_idx] /= r_volumes[np.newaxis, :, np.newaxis]

        photon_info.time_indep_survival_prob[depth_idx] = np.sum(
            photon_info.survival_prob[depth_idx], axis=0
        )

        theta = force_little_endian(table[6].data)

        bin_edges = TimeSphCoord(
            t=t, r=r, theta=theta, phi=np.array([], dtype=t.dtype)
        )

    return photon_info, bin_edges


class DOMTimePolarTables(object):
    """Load and use information from individual-dom (t,r,theta)-binned Retro
    tables.

    Parameters
    ----------
    tables_dir : string

    hash_val : None or string
        Hash string identifying the source Retro tables to use.

    geom : shape (n_strings, n_depths, 3) numpy ndarray, dtype float64

    use_directionality : bool
        Whether to use photon directionality information from the hypothesis
        and table to modify the expected surviving photon counts.

    ic_exponent, dc_exponent : float >= 0, optional
        Modify probabilties in the table by ``prob = 1 - (1 - prob)**exponent``
        to allow for up- and down-scaling the efficiency of the DOMs.
        `ic_exponent` is applied to IceCube (non-DeepCore) DOMs and
        `dc_exponent` is applied to DeepCore DOMs. Note that this is applied to
        each DOM's table after the appropriate quantum efficiency scale factor
        has already been applied (quantum efficiency is applied as a simple
        multiplier; see :attr:`IC_DOM_QUANT_EFF` and
        :attr:`DC_DOM_QUANT_EFF`).

    naming_version : int or None
        Version of naming for single-DOM+directionality tables (original is 0).
        Passing None uses the latest version. Note that any derived tables use
        the latest naming version regardless of what is passed here.

    """
    def __init__(self, tables_dir, hash_val, geom, use_directionality,
                 ic_exponent=1, dc_exponent=1, naming_version=None):
        # Translation and validation of args
        tables_dir = expand(tables_dir)
        assert isdir(tables_dir)
        assert len(geom.shape) == 3
        assert isinstance(use_directionality, bool)
        assert ic_exponent >= 0
        assert dc_exponent >= 0
        if naming_version is None:
            naming_version = len(RETRO_DOM_TABLE_FNAME_PROTO) - 1
        self.naming_version = naming_version
        self.dom_table_fname_proto = RETRO_DOM_TABLE_FNAME_PROTO[naming_version]

        self.tables_dir = tables_dir
        self.hash_val = hash_val
        self.geom = geom
        self.use_directionality = use_directionality
        self.ic_exponent = ic_exponent
        self.dc_exponent = dc_exponent
        self.tables = {'ic': {}, 'dc': {}}
        self.bin_edges = {'ic': {}, 'dc': {}}

    def load_table(self, string, dom, force_reload=False):
        """Load a table from disk into memory.

        Parameters
        ----------
        string : int in [1, 86]
            Indexed from 1, currently 1-86. Strings 1-78 are "regular" IceCube
            strings, while strings 79-86 are DeepCore strings. (It should be
            noted, though, that strings 79 and 80 are considered in-fill
            strings, with "a mix of high quantum-efficiency and standard DOMs";
            which are which is _not_ taken into consideration in the software
            yet.)

        dom : int in [1, 60]
            Indexed from 0, currently 0-59

        force_reload : bool

        """
        if string < 79:
            subdet = 'ic'
            dom_quant_eff = IC_DOM_QUANT_EFF
            exponent = self.ic_exponent
        else:
            subdet = 'dc'
            dom_quant_eff = DC_DOM_QUANT_EFF
            exponent = self.dc_exponent

        if not force_reload and dom in self.tables[subdet]:
            return

        depth_idx = dom - 1
        if self.naming_version == 0:
            fpath = join(
                self.tables_dir,
                self.dom_table_fname_proto.format(
                    string=subdet.upper(), depth_idx=depth_idx
                )
            )
        elif self.naming_version == 1:
            raise NotImplementedError()
            #fpath = join(
            #    self.tables_dir,
            #    self.dom_table_fname_proto.format(
            #        hash_val=self.hash_val,
            #        string=subdet,
            #        depth_idx=depth_idx,
            #        seed=seed, # TODO
            #    )
            #)
        else:
            raise NotImplementedError()

        photon_info, bin_edges = load_t_r_theta_table(
            fpath=fpath,
            depth_idx=depth_idx,
            scale=dom_quant_eff,
            exponent=exponent
        )

        #length = photon_info.length[depth_idx]
        #deltaphi = photon_info.deltaphi[depth_idx]
        self.tables[subdet][depth_idx] = RetroPhotonInfo(
            survival_prob=photon_info.survival_prob[depth_idx],
            time_indep_survival_prob=photon_info.time_indep_survival_prob[depth_idx],
            theta=photon_info.theta[depth_idx],
            deltaphi=photon_info.deltaphi[depth_idx],
            length=(photon_info.length[depth_idx]
                    * np.cos(photon_info.deltaphi[depth_idx]))
        )

        self.bin_edges[subdet][depth_idx] = bin_edges

    def load_tables(self):
        """Load all tables"""
        # TODO: parallelize the loading of each table to reduce CPU overhead
        # time (though most time I expect to be disk-read times, this could
        # still help speed the process up)
        for string in range(1, 86+1):
            for dom in range(1, 60+1):
                self.load_table(string=string, dom=dom, force_reload=False)

    def get_photon_expectation(self, sources, hit_time, string, dom,
                               use_directionality=None):
        """Get the expectation for photon survival.

        Parameters
        ----------
        sources : shape (N,) numpy ndarray, dtype SRC_T

        use_directionality : None or bool
            Whether to use photon directionality informatino in hypo / table to
            modify expected surviving photon counts. If specified, overrides
            argument passed at class instantiation time. Otherwise, that value
            for `use_directionality` is used.

        Returns
        -------
        total_photon_count, expected_photon_count : (float, float)
            Total expected surviving photon count

        """
        if use_directionality is None:
            use_directionality = self.use_directionality

        string_idx = string - 1
        depth_idx = dom - 1

        dom_coord = self.geom[string_idx, depth_idx]
        if string < 79:
            subdet = 'ic'
        else:
            subdet = 'dc'
        table = self.tables[subdet][depth_idx]
        bin_edges = self.bin_edges[subdet][depth_idx]
        survival_prob = table.survival_prob
        time_indep_survival_prob = table.time_indep_survival_prob
        return pexp_t_r_theta(
            sources=sources,
            hit_time=hit_time,
            dom_coord=dom_coord,
            survival_prob=survival_prob,
            time_indep_survival_prob=time_indep_survival_prob,
            t_min=bin_edges.t[0],
            t_max=bin_edges.t[-1],
            n_t_bins=len(bin_edges.t)-1,
            r_min=bin_edges.r[0],
            r_max=bin_edges.r[-1],
            r_power=2,
            n_r_bins=len(bin_edges.r)-1,
            n_costheta_bins=len(bin_edges.theta)-1
        )
