#!/usr/bin/env python
# pylint: disable=wrong-import-position

"""
Load retro tables into RAM, then for a given hypothesis calculate llh for an event
"""
from __future__ import absolute_import, division, print_function

import cPickle as pickle
from os.path import abspath, dirname, isdir, join
import sys
import time
import itertools
from copy import deepcopy
from collections import OrderedDict
from scipy import optimize
from argparse import ArgumentParser
import socket

import numba
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Rectangle

if __name__ == '__main__' and __package__ is None:
    PARENT_DIR = dirname(dirname(abspath(__file__)))
    if PARENT_DIR not in sys.path:
        sys.path.append(PARENT_DIR)
from retro import PI 
from retro import HYPO_PARAMS_T
from retro import DETECTOR_GEOM_FILE
from retro import expand
from retro.discrete_hypo import DiscreteHypo
from retro.discrete_muon_kernels import const_energy_loss_muon
from retro.discrete_cascade_kernels import point_cascade
from retro.table_readers import DOMTimePolarTables

SMALL_P = 1e-8
NOISE_Q = 0.1

"""Parse command line arguments"""
parser = ArgumentParser()
parser.add_argument(
    '-f', '--file', type=str, default='benchmark_events.pkl',
    help='''Events pkl file, containing (string, DOM, charge, time per
    event)''',
)
parser.add_argument(
    '--index', default=0, type=int,
    help='''Event index to be processed'''
)
args = parser.parse_args()

def get_neg_llh(pinfo_gen, event, dom_tables):
    """Get log likelihood.

    Parameters
    ----------
    pinfo_gen

    event : retro.Event namedtuple or convertible thereto

    dom_tables

    Returns
    -------
    llh : float
        Negative of the log likelihood

    """
    neg_llh = 0
    small_p_counts = 0
    small_n_counts = 0

    for string_idx in range(86):
        for dom_idx in range(60):
            start_t = time.time()
            total_observed_ps = 0.
            total_expected_ps = 0.
            try:
                hits = event[string_idx+1][dom_idx+1]
            except KeyError:
                hits = None

            if hits is not None:
                weights = hits['weight']
                total_observed_ps = np.sum(weights)
                times = hits['time']
                for hit_time, weight in zip(times, weights):
                    total_expected_ps, expected_p = dom_tables.get_photon_expectation(
                                              pinfo_gen=pinfo_gen,
                                              hit_time=hit_time,
                                              string=string_idx+1,
                                              depth_idx=dom_idx,
                                              )
                    # normalize to get pdf
                    if expected_p > 0.:
                        P = expected_p / total_expected_ps
                    else:
                        P = 0.
                    # make sure they're not zero
                    probability = P + SMALL_P - P*SMALL_P
                    # add them number of hit times
                    neg_llh -= weight * np.log(probability)
            else:
                # just want total_expected_ps, but using thr same function right now
                total_expected_ps, expected_p = dom_tables.get_photon_expectation(
                                              pinfo_gen=pinfo_gen,
                                              hit_time=-1000,
                                              string=string_idx+1,
                                              depth_idx=dom_idx,
                                              )
            

            # add a noise floor
            N = total_expected_ps + NOISE_Q
            # for 0,0 poisson is i think not well defined
            if not (total_observed_ps == 0. and total_expected_ps == 0.):
                # add only non-constant part
                neg_llh -= (total_observed_ps * np.log(N) - N)
            stop_t = time.time()

    return neg_llh

def zenith_astro_to_reco(zenith):
    return PI - zenith

def azimuth_astro_to_reco(azimuth):
    return (azimuth - PI) % (2*PI)



discrete_hypo = DiscreteHypo(hypo_kernels=[point_cascade, const_energy_loss_muon])
geom_file = DETECTOR_GEOM_FILE
# ET:
if socket.gethostname() in ['schwyz', 'uri', 'unterwalden']:
    tables_dir = '/data/icecube/retro_tables/full1000/'
# ACI
else:
    tables_dir = '/gpfs/group/dfc13/default/retro_tables/full1000/'

# Load detector geometry array
print('Loading detector geometry from "%s"...' % expand(geom_file))
detector_geometry = np.load(expand(geom_file))

# Load tables
print('Loading DOM tables...')
dom_tables = DOMTimePolarTables(
    tables_dir=tables_dir,
    hash_val=None,
    geom=detector_geometry,
    use_directionality=False,
    naming_version=0,
)
dom_tables.load_tables()

# create scan dimensions disctionary
scan_dims = {}
scan_dims['t'] = {'scan_points':np.linspace(-200, 200, 21)}
scan_dims['x'] = {'scan_points':np.linspace(-100, 100, 21)}
scan_dims['y'] = {'scan_points':np.linspace(-100, 100, 21)}
scan_dims['z'] = {'scan_points':np.linspace(-430, -370, 21)}
scan_dims['track_zenith'] = {'scan_points':np.linspace(0, PI, 21)}
scan_dims['track_azimuth'] = {'scan_points':np.linspace(0, 2*PI, 21)}
scan_dims['track_energy'] = {'scan_points':np.linspace(0, 40, 21)}
scan_dims['cascade_energy'] = {'scan_points':np.linspace(0, 40, 21)}

#open events file:
with open(args.file, 'rb') as f:
    events = pickle.load(f)


def edges(centers):
    d = np.diff(centers)/2
    e = np.append(centers[0]-d[0], centers[:-1]+d)
    return np.append(e, centers[-1]+d[-1])



## OPTIMIZE

if True:

    # multinest
    import pymultinest

    #for i,event in enumerate(events):
    event = events[args.index]

    truth = {'t':event['MC']['t'],
             'x':event['MC']['x'],
             'y':event['MC']['y'],
             'z':event['MC']['z'],
             'track_zenith':zenith_astro_to_reco(event['MC']['zenith']),
             'track_azimuth':azimuth_astro_to_reco(event['MC']['azimuth']),
             'track_energy': event['MC']['energy'] if event['MC']['type'] == 'NuMu' else 0,
             'cascade_energy': event['MC']['energy'] if event['MC']['type'] == 'NuE' else 0,
             }
    true_params = HYPO_PARAMS_T(**truth)
    pinfo_gen = discrete_hypo.get_pinfo_gen(true_params)
    print('Truth at llh=%.3f'%get_neg_llh(pinfo_gen, event, dom_tables))




    def prior(cube, ndim, nparams):
        '''
        function needed by pymultinest in order to transform
        the so called `cube` into the actual dimesnion of the parameter space.
        The cube is [0,1]^n
        '''
        # t
        cube[0] = (cube[0] * 2000) - 1000
        # x, y, z
        cube[1] = (cube[1] * 1000) - 500
        cube[2] = (cube[2] * 1000) - 500
        cube[3] = (cube[3] * 1000) - 500
        # zenith
        cube[4] = cube[4] * PI
        # azimuth
        cube[5] = cube[5] * 2 * PI
        # energy
        #cube[6] = cube[6] * 200
        # log uniform prior between 10^0 and 10^3
        cube[6] = 10**(cube[6]*3)
        # tarck fraction already (0,1)

    def loglike(cube, ndim, nparams):
        '''
        callable function for multinest to get llh values
        the cube here is after the prior function has been applied
        to it - i.e. it alsready contains the actual parameter values
        '''
        hypo_params = HYPO_PARAMS_T(t=cube[0],
                                    x=cube[1],
                                    y=cube[2],
                                    z=cube[3],
                                    track_zenith=cube[4],
                                    track_azimuth=cube[5],
                                    cascade_energy=cube[6]*(1-cube[7]),
                                    track_energy=cube[6]*cube[7]
                                    )
        pinfo_gen = discrete_hypo.get_pinfo_gen(hypo_params)
        return -get_neg_llh(pinfo_gen, event, dom_tables)

    n_params = 8

    if socket.gethostname() in ['schwyz', 'uri', 'unterwalden']:
        outname = 'out/tol0.1_evt%i-'%args.index
    else:
        outname = '/gpfs/scratch/pde3/retro/out/tol0.1_evt%i-'%args.index
    pymultinest.run(loglike, prior, n_params,
                    verbose=True,
                    outputfiles_basename=outname,
                    resume=False,
                    n_live_points=160,
                    evidence_tolerance=0.1,
                    sampling_efficiency=0.8,
                    max_modes=10,
                    seed=0,
                    max_iter=100000,
                    )



    #def minimizer_callable(x):
    #    '''
    #    x: [t, x, y, z, track_zenith, track_azimuth, energy, track_fraction]
    #    '''
    #    hypo_params = HYPO_PARAMS_T(t=x[0], x=x[1], y=x[2], z=x[3], track_zenith=x[4], track_azimuth=x[5], cascade_energy=x[6]*(1-x[7]), track_energy=x[6]*x[7])
    #    pinfo_gen = discrete_hypo.get_pinfo_gen(hypo_params)
    #    llh = get_neg_llh(pinfo_gen, event, dom_tables)
    #    #print('%.2f, %s'%(llh, x))
    #    return llh


    #bounds = OrderedDict()
    #bounds['t'] = (-1000, 1000)
    #bounds['x'] = (-500, 500)
    #bounds['y'] = (-500, 500)
    #bounds['z'] = (-500, 500)
    #bounds['track_zenith'] = (0, PI)
    #bounds['track_azimuth'] = (0, 2*PI)
    #bounds['energy'] = (0, 200)
    #bounds['track_fraction'] = (0, 1)


    # scipy
    #res = optimize.differential_evolution(minimizer_callable, bounds=bounds.values(),
    #                                      popsize=20,
    #                                      disp=True,
    #                                      polish=True,
    #                                      tol=0.1,
    #                                      strategy='best2bin',
    #                                      )

    #print(res)


    # particle swarm:
    #from pyswarm import pso

    #lower_bounds = []
    #upper_bounds = []
    #for bound in bounds.values():
    #    lower_bounds.append(bound[0])
    #    upper_bounds.append(bound[1])
    #xopt, fopt = pso(minimizer_callable, lower_bounds, upper_bounds,\
    #                 debug=True,
    #                 omega=10,
    #                 phip=10,
    #                 phig=10,
    #                 )

    #print(xopt, fopt)

    # Bayesian optimization:
    #def minimizer_callable(t,x,y,z,track_zenith,track_azimuth, energy, track_fraction):
    #    hypo_params = HYPO_PARAMS_T(t=t, x=x, y=y, z=z, track_zenith=track_zenith, track_azimuth=track_azimuth, cascade_energy=energy*(1-track_fraction), track_energy=energy*track_fraction)
    #    pinfo_gen = discrete_hypo.get_pinfo_gen(hypo_params)
    #    return -get_neg_llh(pinfo_gen, event, dom_tables)

    #from bayes_opt import BayesianOptimization
    #bo = BayesianOptimization(minimizer_callable, bounds)
    #gp_params = {'kernel': None,
    #             'alpha': 1e-3,
    #             'xi': 0.,
    #             'acq': 'ei',
    #             }

    #bo.maximize(init_points=100, n_iter=50, **gp_params)
    #print(bo.res['max'])


## SCAN


if False:
    # 1-d scan:
    scan_dim = 'track_zenith'
    # scan multiple events
    llhs = []
    for i, event in enumerate(events[0:10]):
        print('\nevent ',i)

        hypo = {'t':event['MC']['t'],
                'x':event['MC']['x'],
                'y':event['MC']['y'],
                'z':event['MC']['z'],
                'track_zenith':zenith_astro_to_reco(event['MC']['zenith']),
                'track_azimuth':azimuth_astro_to_reco(event['MC']['azimuth']),
                'track_energy': event['MC']['energy'] if event['MC']['type'] == 'NuMu' else 0,
                'cascade_energy': event['MC']['energy'] if event['MC']['type'] == 'NuE' else 0,
                }

        llhs.append([])
        for point in scan_dims[scan_dim]['scan_points']: 
            hypo[scan_dim] = point
            hypo_params = HYPO_PARAMS_T(**hypo)
            pinfo_gen = discrete_hypo.get_pinfo_gen(hypo_params)
            llhs[-1].append(get_neg_llh(pinfo_gen, event, dom_tables))

    llhs = np.array(llhs)

    # plot them
    for llh in llhs:
        plt.plot(scan_dims[scan_dim]['scan_points'], llh)
        plt.gca().set_xlabel(scan_dim)
        plt.gca().set_ylabel('-llh')
    plt.savefig('scans/%s.png'%scan_dim)


if False:
    # 2-d scan:

    cmap = 'YlGnBu_r'

    #scan_dim_x = 'track_energy'
    #scan_dim_y = 'cascade_energy'

    event = events[0]

    truth = {'t':event['MC']['t'],
            'x':event['MC']['x'],
            'y':event['MC']['y'],
            'z':event['MC']['z'],
            'track_zenith':zenith_astro_to_reco(event['MC']['zenith']),
            'track_azimuth':azimuth_astro_to_reco(event['MC']['azimuth']),
            'track_energy': event['MC']['energy'] if event['MC']['type'] == 'NuMu' else 0,
            'cascade_energy': event['MC']['energy'] if event['MC']['type'] == 'NuE' else 0,
            }

    for scan_dim_x, scan_dim_y in itertools.combinations(truth.keys(), 2):
        print('scanning %s vs. %s'%(scan_dim_x, scan_dim_y))
        hypo = deepcopy(truth)

        llhs = []
        for point_x in scan_dims[scan_dim_x]['scan_points']: 
            llhs.append([])
            for point_y in scan_dims[scan_dim_y]['scan_points']: 
                hypo[scan_dim_x] = point_x
                hypo[scan_dim_y] = point_y
                hypo_params = HYPO_PARAMS_T(**hypo)
                pinfo_gen = discrete_hypo.get_pinfo_gen(hypo_params)
                llhs[-1].append(get_neg_llh(pinfo_gen, event, dom_tables))
                
        llhs = np.array(llhs)
        x = edges(scan_dims[scan_dim_x]['scan_points'])
        y = edges(scan_dims[scan_dim_y]['scan_points'])
        xx, yy = np.meshgrid(x, y)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        mg = ax.pcolormesh(xx, yy, llhs.T, cmap=cmap)
        fig.colorbar(mg)
        ax.set_xlabel(scan_dim_x)
        ax.set_ylabel(scan_dim_y)
        ax.plot(truth[scan_dim_x], truth[scan_dim_y], c='g', marker='*')
        
        #truth
        #ax.axvline(theta_true, color='r')
        #ax.axhline(z_v_true, color='r')
        #ax.set_title('Event %i, E_cscd = %.2f GeV, E_trck = %.2f GeV'%(evt, cscd_energy_true, trck_energy_true)) 

        # plot minimum:
        m_x, m_y = np.unravel_index(llhs.argmin(), llhs.shape)
        ax.add_patch(Rectangle((x[m_x], y[m_y]), (x[m_x+1] - x[m_x]), (y[m_y+1] - y[m_y]),alpha=1, fill=False, edgecolor='red', linewidth=2))

        plt.savefig('scans/2d_%s_%s.png'%(scan_dim_x, scan_dim_y),dpi=150)

