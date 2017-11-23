#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extract sampled angular sensitivity curves from IceCube software.
"""


from __future__ import absolute_import, division, print_function

from os import listdir
from os.path import expanduser, expandvars, isdir, join

import numpy as np

from icecube.clsim import GetIceCubeDOMAngularSensitivity


__all__ = ['DEFAULT_DIRPATH', 'extract_angular_sensitvity']


DEFAULT_DIRPATH = '$I3_SRC/ice-models/resources/models/angsens'


def expand(p):
    """Fully expand path `p`"""
    return expanduser(expandvars(p))


def extract_angular_sensitvity(outdir='.',
                               dirpath=DEFAULT_DIRPATH,
                               hole_ice_model='as.h2-50cm'):
    """Extract angular sensitivity curve from IceCube software.

    Parameters
    ----------
    dirpath : string
        Path to directory containing the model

    hole_ice_model : string
        Hole ice model. E.g., "as.h2-50cm"

    """
    filepath = expand(join(dirpath, hole_ice_model))
    angsens = GetIceCubeDOMAngularSensitivity(holeIce=filepath)

    cz_vals = np.linspace(-1, 1, 101)
    sampled_angsens = np.array([(cz, angsens.GetValue(cz)) for cz in cz_vals])

    print(
        'Angular sensitivity is {} at {} deg and is {} at {} deg'
        .format(sampled_angsens[0, 1],
                np.rad2deg(np.arccos(sampled_angsens[0, 0])),
                sampled_angsens[-1, 1],
                np.rad2deg(np.arccos(sampled_angsens[-1, 0])))
    )

    average = (
        np.trapz(y=sampled_angsens[:, 1], x=sampled_angsens[:, 0])
        / (sampled_angsens[-1, 0] - sampled_angsens[0, 0])
    )
    print('Average sensitivity = {:.4e} (per unit of coszen)'.format(average))

    outpath = expand(join(
        outdir,
        'sampled_dom_angsens__hole_ice_model__'
        + hole_ice_model.rstrip('.')
        + '.csv'
    ))

    header = (
        'Sampled DOM angular sensitivity curve, {} model\n'
        'coszen,sensitivity'.format(hole_ice_model)
    )
    np.savetxt(outpath, sampled_angsens, delimiter=',', header=header)
    print('Saved sampled angular sensitivity to "{}"'.format(outpath))


def main():
    """Extract all models from the dir"""
    dirpath = DEFAULT_DIRPATH
    for hole_ice_model in sorted(listdir(expand(dirpath))):
        if isdir(expand(join(dirpath, hole_ice_model))):
            continue
        print('Extracting model "{}"'.format(hole_ice_model))
        print('-'*80)
        extract_angular_sensitvity(dirpath=dirpath,
                                   hole_ice_model=hole_ice_model)
        print('-'*80)
        print('')


if __name__ == '__main__':
    main()