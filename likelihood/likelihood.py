import numpy as np
import pyfits
import h5py
import sys, os
from scipy.special import gammaln
from scipy.stats import norm
from argparse import ArgumentParser
from hypothesis.hypo_fast import hypo, PowerAxis
from particles import particle, particle_array
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numba
from pyswarm import pso

'''
This module is loading up retor tables into RAM first, and then tales icecube hdf5 files with events (hits series) inthem as inputs and calculates lilkelihoods
These likelihoods can be single points or 1d or 2d scans at the moment.

'''

# cmd line arguments
parser = ArgumentParser(description='''make 2d event pictures''')
parser.add_argument('-f', '--file', metavar='H5_FILE', type=str, help='input HDF5 file',
                    default='/fastio/icecube/deepcore/data/MSU_sample/level5pt/numu/14600/icetray_hdf5/Level5pt_IC86.2013_genie_numu.014600.000000.hdf5')
parser.add_argument('-i', '--index', default=0, type=int, help='index offset for event to start with')
args = parser.parse_args()


# --- load tables ---
# tables are not binned in phi, but we will do so for the hypo, therefore need to apply norm
n_phi_bins = 20.
norm = 1./n_phi_bins
# correct for DOM wavelength acceptance, given cherenkov spectrum (source photons)
# see tables/wavelegth.py
#norm *= 0.0544243061857
# compensate for costheta bins (40) and wavelength accepet.
#norm = 0.0544243061857 * 40.
# add in quantum efficiencies (?)
dom_eff_ic = 0.25
dom_eff_dc = 0.35

# define phi bin edges, as these are ignored in the tables (symmetry)
phi_bin_edges = np.linspace(0, 2*np.pi, n_phi_bins+1)

# dictionarries with DOM depth number as key, separately for IceCube (IC) and DeepCore (DC)
IC_n_phot = {}
IC_p_theta = {}
IC_p_phi = {}
IC_p_length = {}
DC_n_phot = {}
DC_p_theta = {}
DC_p_phi = {}
DC_p_length = {}
# read in the actual tables
for dom in range(60):
    # IC tables
    fname = 'tables/tables/full1000/retro_nevts1000_IC_DOM%i_r_cz_t_angles.fits'%dom
    if os.path.isfile(fname):
        table = pyfits.open(fname)
        IC_n_phot[dom] = table[0].data * (norm * dom_eff_ic)
        IC_p_theta[dom] = table[1].data
        IC_p_phi[dom] = table[2].data
        IC_p_length[dom] = table[3].data
    else:
        print'No table for IC DOM %i'%dom
    if dom == 0:
        # first dom used to get the bin edges:
        t_bin_edges = table[4].data
        r_bin_edges = table[5].data
        theta_bin_edges = table[6].data
    else:
        assert np.array_equal(t_bin_edges, table[4].data)
        assert np.array_equal(r_bin_edges, table[5].data)
        assert np.array_equal(theta_bin_edges, table[6].data)
    # DC tables
    fname = 'tables/tables/full1000/retro_nevts1000_DC_DOM%i_r_cz_t_angles.fits'%dom
    if os.path.isfile(fname):
        table = pyfits.open(fname)
        DC_n_phot[dom] = table[0].data * (norm * dom_eff_dc)
        DC_p_theta[dom] = table[1].data
        DC_p_phi[dom] = table[2].data
        DC_p_length[dom] = table[3].data
        assert np.array_equal(t_bin_edges, table[4].data)
        assert np.array_equal(r_bin_edges, table[5].data)
        assert np.array_equal(theta_bin_edges, table[6].data)
    else:
        print'No table for DC DOM %i'%dom

# ToDo...not very nice to invert the time direction here
t_bin_edges = - t_bin_edges[::-1]



# --- load events ---
f = h5py.File(args.file)
name = args.file.split('/')[-1][:-5]
#pulses
var = 'SRTInIcePulses'
p_evts = f[var]['Event']
p_string = f[var]['string']
p_om = f[var]['om']
p_time = f[var]['time']
p_charge = f[var]['charge']

# interaction type
int_type = f['I3MCWeightDict']['InteractionType']

# true Neutrino
neutrinos = particle_array(
    f['trueNeutrino']['Event'],
    f['trueNeutrino']['time'],
    f['trueNeutrino']['x'],
    f['trueNeutrino']['y'],
    f['trueNeutrino']['z'],
    f['trueNeutrino']['zenith'],
    f['trueNeutrino']['azimuth'],
    f['trueNeutrino']['energy'],
    None,
    f['trueNeutrino']['type'],
    color='r',
    linestyle=':',
    label='Neutrino')
# true track
tracks = particle_array(
    f['trueMuon']['Event'],
    f['trueMuon']['time'],
    f['trueMuon']['x'],
    f['trueMuon']['y'],
    f['trueMuon']['z'],
    f['trueMuon']['zenith'],
    f['trueMuon']['azimuth'],
    f['trueMuon']['energy'],
    f['trueMuon']['length'],
    forward=True,
    color='b',
    linestyle='-',
    label='track')
# true Cascade
cascades = particle_array(
    f['trueCascade']['Event'],
    f['trueCascade']['time'],
    f['trueCascade']['x'],
    f['trueCascade']['y'],
    f['trueCascade']['z'],
    f['trueCascade']['zenith'],
    f['trueCascade']['azimuth'],
    f['trueCascade']['energy'],
    color='y',
    label='cascade')
# Multinest reco
ML_recos = particle_array(
    f['IC86_Dunkman_L6_PegLeg_MultiNest8D_NumuCC']['Event'],
    f['IC86_Dunkman_L6_PegLeg_MultiNest8D_NumuCC']['time'],
    f['IC86_Dunkman_L6_PegLeg_MultiNest8D_NumuCC']['x'],
    f['IC86_Dunkman_L6_PegLeg_MultiNest8D_NumuCC']['y'],
    f['IC86_Dunkman_L6_PegLeg_MultiNest8D_NumuCC']['z'],
    f['IC86_Dunkman_L6_PegLeg_MultiNest8D_NumuCC']['zenith'],
    f['IC86_Dunkman_L6_PegLeg_MultiNest8D_NumuCC']['azimuth'],
    f['IC86_Dunkman_L6_PegLeg_MultiNest8D_NumuCC']['energy'],
    color='g',
    label='Multinest')
# SPE fit
SPE_recos = particle_array(
    f['SPEFit2']['Event'],
    f['SPEFit2']['time'],
    f['SPEFit2']['x'],
    f['SPEFit2']['y'],
    f['SPEFit2']['z'],
    f['SPEFit2']['zenith'],
    f['SPEFit2']['azimuth'],
    color='m',
    label='SPE')

# --- load detector geometry array ---
geo = np.load('likelihood/geo_array.npy')

#@profile
def get_llh(hypo, t, x, y, z, q, string, om):
    #llhs = []
    tot_llh = 0.
    n_noise = 0
    # loop over hits
    for hit in range(len(t)):
        # for every DOM + hit:
        #print 'getting matrix for hit %i'%hit
        # t, r ,cz, phi 
        #print 'hit at %.2f ns at (%.2f, %.2f, %.2f)'%(t[hit], x[hit], y[hit], z[hit])

        # get the photon expectations of the hypothesis in the DOM-hit coordinates
        n_phot, p_theta, p_phi, p_length = hypo.get_matrices(t[hit], x[hit], y[hit], z[hit])
        # and also get the retro table for that hit
        if string[hit] < 79:
            # these are ordinary icecube strings
            n_phot_map = IC_n_phot[om[hit] - 1]
            p_theta_map = IC_p_theta[om[hit] - 1]
            p_phi_map = IC_p_phi[om[hit] - 1]
            p_length_map = IC_p_length[om[hit] - 1]
        else:
            # these are deepocre strings
            n_phot_map = DC_n_phot[om[hit] - 1]
            p_theta_map = DC_p_theta[om[hit] - 1]
            p_phi_map = DC_p_phi[om[hit] - 1]
            p_length_map = DC_p_length[om[hit] - 1]

        # get max llh between z_matrix and gamma_map
        # noise probability?
        q_noise = 0.00000025
        expected_q = 0.
        for element in n_phot:
            # get hypo
            idx, hypo_count = element
            # these two agles need to be inverted, because we're backpropagating but want to match to forward propagating photons
            hypo_theta = np.pi - p_theta[idx]
            hypo_phi = np.pi - p_phi[idx]
            hypo_legth = p_length[idx]
            # get map
            map_count = n_phot_map[idx[0:3]]
            map_theta = p_theta_map[idx[0:3]]
            map_phi = p_phi_map[idx[0:3]]
            map_length = p_length_map[idx[0:3]]

            # assume now source is totally directed at 0.73 (cherenkov angle)

            # accept this fraction as isotropic light
            dir_fraction = map_length**2
            print 'map length = ',dir_fraction
            iso_fraction = (1. - dir_fraction)

            # whats the cos(psi) between track direction and map?
            # accept this fraction of directional light
            # this is wrong i think...
	    #proj_dir = np.arccos((np.cos(hypo_theta)*np.cos(map_theta) + np.sin(hypo_theta)*np.sin(map_theta)*np.cos(hypo_phi - map_phi)))
	    proj_dir = (np.cos(hypo_theta)*np.cos(map_theta) + np.sin(hypo_theta)*np.sin(map_theta)*np.cos(hypo_phi - map_phi))
            #print proj_dir
            # how close to 0.754 is it?
            # get a weight from a gaussian
            delta = -proj_dir - 0.754
            accept_dir = np.exp(- delta**2 / 0.1) * dir_fraction
            accept_iso = iso_fraction

            # acceptance directed light
            total_q = hypo_count * map_count
            directional_q = hypo_legth * total_q 
            isotropic_q = (1. - hypo_legth) * total_q

            # factor betwen isotropic and direction light
            #f_iso_dir = 10.

            #expected_q += directional_q * (accept_iso/f_iso_dir + accept_dir) + isotropic_q * (accept_iso + accept_dir/f_iso_dir)
            #expected_q += directional_q * accept_dir + isotropic_q * accept_iso
            expected_q += total_q * accept_dir

            #print 'total q ',total_q
            #print 'directional q = ,', directional_q
            #print 'hypo count: ', hypo_count
            #print 'map count: ',map_count
            #print 'accept iso = ', accept_iso
            #print 'accept dir = ', accept_dir

        #print 'expected q = %.4f'%expected_q
        #print 'observed q = %.4f'%q[hit]
        #print ''

        if q_noise > expected_q:
            n_noise += 1
        expected_q = max(q_noise, expected_q)
        llh = -(q[hit]*np.log(expected_q) - expected_q  - gammaln(q[hit]+1.))
        #print llh
        tot_llh += llh
    return tot_llh, n_noise

def get_6d_llh(params_6d, trck_energy, cscd_energy, trck_e_scale, cscd_e_scale, t, x, y, z, q, string, om, t_bin_edges, r_bin_edges, theta_bin_edges, phi_bin_edges):
    ''' minimizer callable for 6d (vertex + angle) minimization)
        params_6d : list
            [t, x, y, z, theta, phi]
        '''
    my_hypo = hypo(params_6d[0], params_6d[1], params_6d[2], params_6d[3], theta=params_6d[4], phi=params_6d[5], trck_energy=trck_energy, cscd_energy=cscd_energy, cscd_e_scale=cscd_e_scale, trck_e_scale=trck_e_scale)
    my_hypo.set_binning(t_bin_edges, r_bin_edges, theta_bin_edges, phi_bin_edges)
    llh, _ = get_llh(my_hypo, t, x, y, z, q, string, om)
    #print 'llh=%.2f at t=%.2f, x=%.2f, y=%.2f, z=%.2f, theta=%.2f, phi=%.2f'%tuple([llh] + [p for p in params_6d])
    return llh


# a super quirky way to define what to run
do_true = False
do_x =  False
do_y =  False
do_z =  False
do_t =  False
do_theta =  False
do_phi =  False
do_cscd_energy =  False
do_trck_energy =  False
do_xz = False
do_thetaphi = False
do_thetaz = False
do_minimize = False
#do_true = True
#do_x =  True
#do_y =  True
#do_z =  True
do_t =  True
#do_theta =  True
#do_phi =  True
#do_cscd_energy =  True
#do_trck_energy =  True
#do_xz = True
#do_thetaphi = True
#do_thetaz = True
#do_minimize = True

if do_minimize:
    outfile_name = name+'.csv'
    if os.path.isfile(outfile_name):
        print 'File %s exists'%outfile_name
        sys.exit()
    with open(outfile_name, 'w') as outfile:
        outfile.write('#event, t_true, x_true, y_true, z_true, theta_true, phi_true, trck_energy_true, cscd_energy_true, llh_true, \
t_retro, x_retro, y_retro, z_retro, theta_retro, phi_retro, llh_retro, \
t_mn, x_mn, y_mn, z_mn, theta_mn, phi_mn, llh_mn, \
t_spe, x_spe, y_spe, z_spe, theta_spe, phi_spe, llh_spe\n')

# iterate through events
for idx in xrange(args.index, len(neutrinos)):

    # ---------- read event ------------
    evt = neutrinos[idx].evt
    print 'working on %i'%evt
    # find first index
    first = np.where(p_evts == evt)[0][0]
    # read in DOM hits
    string = []
    om = []
    t = []
    q = []
    current_event = p_evts[first]
    while p_evts[first] == evt:
        string.append(p_string[first])
        om.append(p_om[first])
        t.append(p_time[first])
        q.append(p_charge[first])
        first += 1
    string = np.array(string)
    om = np.array(om)
    t = np.array(t)
    q = np.array(q)
    # convert to x,y,z hit positions
    x = []
    y = []
    z = []
    #print geo.shape
    for s, o in zip(string, om):
        x.append(geo[s-1, o-1, 0])
        y.append(geo[s-1, o-1, 1])
        z.append(geo[s-1, o-1, 2])
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    # -------------------------------
    # scan vertex z-positions, eerything else at truth

    # truth values
    t_v_true = neutrinos[idx].v[0]
    x_v_true = neutrinos[idx].v[1]
    y_v_true = neutrinos[idx].v[2]
    z_v_true = neutrinos[idx].v[3]
    theta_true = neutrinos[idx].theta
    # azimuth?
    phi_true = neutrinos[idx].phi
    cscd_energy_true = cascades[idx].energy
    trck_energy_true = tracks[idx].energy

    print 'True event info:'
    print 'time = %.2f ns'%t_v_true
    print 'vertex = (%.2f, %.2f, %.2f)'%(x_v_true, y_v_true, z_v_true)
    print 'theta, phi = (%.2f, %.2f)'%(theta_true, phi_true)
    print 'E_cscd, E_trck (GeV) = %.2f, %.2f'%(cscd_energy_true, trck_energy_true)
    print 'n hits = %i'%len(t)
    print 't hits: .',t

    # some factors by pure choice, since we don't have directionality yet...
    cscd_e_scale=cscd_e_scale = 10. #2.
    trck_e_scale=trck_e_scale = 10. #20.

    n_scan_points = 21
    #cmap = 'afmhot'
    cmap = 'YlGnBu_r'

    if do_minimize:
        kwargs = {}
        kwargs['t_bin_edges'] = t_bin_edges
        kwargs['r_bin_edges'] = r_bin_edges
        kwargs['theta_bin_edges'] = theta_bin_edges
        kwargs['phi_bin_edges'] = phi_bin_edges
        kwargs['trck_energy'] = trck_energy_true
        kwargs['trck_e_scale'] = trck_e_scale
        kwargs['cscd_energy'] = cscd_energy_true
        kwargs['cscd_e_scale'] = cscd_e_scale
        kwargs['t'] = t
        kwargs['x'] = x
        kwargs['y'] = y
        kwargs['z'] = z
        kwargs['q'] = q
        kwargs['string'] = string
        kwargs['om'] = om

        # [t, x, y, z, theta, phi]
        #lower_bounds = [t_v_true - 100, x_v_true - 50, y_v_true - 50, z_v_true - 50, max(0, theta_true - 0.5), max(0, phi_true - 1.)]
        #upper_bounds = [t_v_true + 100, x_v_true + 50, y_v_true + 50, z_v_true + 50, min(np.pi, theta_true + 0.5), min(2*np.pi, phi_true + 1.)]
        lower_bounds = [t_v_true - 300, x_v_true - 100, y_v_true - 100, z_v_true - 100, 0., 0.]
        upper_bounds = [t_v_true + 300, x_v_true + 100, y_v_true + 100, z_v_true + 100, np.pi, 2*np.pi]
        
        truth = [t_v_true, x_v_true, y_v_true, z_v_true, theta_true, phi_true]
        llh_truth = get_6d_llh(truth, **kwargs)
        print 'llh at truth = %.2f'%llh_truth

        mn = [ML_recos[idx].v[0], ML_recos[idx].v[1], ML_recos[idx].v[2], ML_recos[idx].v[3], ML_recos[idx].theta, ML_recos[idx].phi]
        llh_mn = get_6d_llh(mn, **kwargs)

        spe = [SPE_recos[idx].v[0], SPE_recos[idx].v[1], SPE_recos[idx].v[2], SPE_recos[idx].v[3], SPE_recos[idx].theta, SPE_recos[idx].phi]
        llh_spe = get_6d_llh(spe, **kwargs)

        xopt1, fopt1 = pso(get_6d_llh, lower_bounds, upper_bounds, kwargs=kwargs, minstep=1e-5, minfunc=1e-1, debug=True)

        print 'truth at   t=%.2f, x=%.2f, y=%.2f, z=%.2f, theta=%.2f, phi=%.2f'%tuple(truth)
        print('with llh = %.2f'%llh_truth)
        print 'optimum at t=%.2f, x=%.2f, y=%.2f, z=%.2f, theta=%.2f, phi=%.2f'%tuple([p for p in xopt1])
        print('with llh = %.2f\n'%fopt1)

        outlist = [evt] + truth + [trck_energy_true, cscd_energy_true] + [llh_truth] + [p for p in xopt1] + [fopt1] + mn + [llh_mn] + spe + [llh_spe]
        string = ", ".join([str(e) for e in outlist])
        with open(outfile_name, 'a') as outfile:
            outfile.write(string)
            outfile.write('\n')


    if do_true:
        my_hypo = hypo(t_v_true, x_v_true, y_v_true, z_v_true, theta=theta_true, phi=phi_true, trck_energy=trck_energy_true, cscd_energy=cscd_energy_true, cscd_e_scale=cscd_e_scale, trck_e_scale=trck_e_scale)
        my_hypo.set_binning(t_bin_edges, r_bin_edges, theta_bin_edges, phi_bin_edges)
        llh, noise  = get_llh(my_hypo, t, x, y, z, q, string, om)

    if do_z:
        # scan z pos
        z_vs = np.linspace(z_v_true - 50, z_v_true + 50, n_scan_points)
        llhs = []
        noises = []
        for z_v in z_vs:
            print 'testing z = %.2f'%z_v
            my_hypo = hypo(t_v_true, x_v_true, y_v_true, z_v, theta=theta_true, phi=phi_true, trck_energy=trck_energy_true, cscd_energy=cscd_energy_true, cscd_e_scale=cscd_e_scale, trck_e_scale=trck_e_scale)
            my_hypo.set_binning(t_bin_edges, r_bin_edges, theta_bin_edges, phi_bin_edges)
            llh, noise = get_llh(my_hypo, t, x, y, z, q, string, om)
            llhs.append(llh)
            noises.append(noise)
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(z_vs, llhs)
        ax.set_ylabel('llh')
        ax.set_xlabel('Vertex z (m)')
        #truth
        ax.axvline(z_v_true, color='r')
        ax.axvline(ML_recos[idx].v[3], color='g')
        ax.axvline(SPE_recos[idx].v[3], color='m')
        ax.set_title('Event %i, E_cscd = %.2f GeV, E_trck = %.2f GeV'%(evt, cscd_energy_true, trck_energy_true)) 
        plt.savefig('z_%s.png'%evt,dpi=150)

    if do_x:
        # scan x pos
        x_vs = np.linspace(x_v_true - 50, x_v_true + 50, n_scan_points)
        llhs = []
        for x_v in x_vs:
            print 'testing x = %.2f'%x_v
            my_hypo = hypo(t_v_true, x_v, y_v_true, z_v_true, theta=theta_true, phi=phi_true, trck_energy=trck_energy_true, cscd_energy=cscd_energy_true, cscd_e_scale=cscd_e_scale, trck_e_scale=trck_e_scale)
            my_hypo.set_binning(t_bin_edges, r_bin_edges, theta_bin_edges, phi_bin_edges)
            llh, noise = get_llh(my_hypo, t, x, y, z, q, string, om)
            llhs.append(llh)
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x_vs, llhs)
        ax.set_ylabel('llh')
        ax.set_xlabel('Vertex x (m)')
        #truth
        ax.axvline(x_v_true, color='r')
        ax.axvline(ML_recos[idx].v[1], color='g')
        ax.axvline(SPE_recos[idx].v[1], color='m')
        ax.set_title('Event %i, E_cscd = %.2f GeV, E_trck = %.2f GeV'%(evt, cscd_energy_true, trck_energy_true)) 
        plt.savefig('x_%s.png'%evt,dpi=150)

    if do_y:
        # scan y pos
        y_vs = np.linspace(y_v_true - 50, y_v_true + 50, n_scan_points)
        llhs = []
        for y_v in y_vs:
            print 'testing y = %.2f'%y_v
            my_hypo = hypo(t_v_true, x_v_true, y_v, z_v_true, theta=theta_true, phi=phi_true, trck_energy=trck_energy_true, cscd_energy=cscd_energy_true, cscd_e_scale=cscd_e_scale, trck_e_scale=trck_e_scale)
            my_hypo.set_binning(t_bin_edges, r_bin_edges, theta_bin_edges, phi_bin_edges)
            llh, noise = get_llh(my_hypo, t, x, y, z, q, string, om)
            llhs.append(llh)
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(y_vs, llhs)
        ax.set_ylabel('llh')
        ax.set_xlabel('Vertex y (m)')
        #truth
        ax.axvline(y_v_true, color='r')
        ax.axvline(ML_recos[idx].v[2], color='g')
        ax.axvline(SPE_recos[idx].v[2], color='m')
        ax.set_title('Event %i, E_cscd = %.2f GeV, E_trck = %.2f GeV'%(evt, cscd_energy_true, trck_energy_true)) 
        plt.savefig('y_%s.png'%evt,dpi=150)

    if do_t:
        # scan t pos
        t_vs = np.linspace(t_v_true - 200, t_v_true + 200, n_scan_points)
        llhs = []
        noises = []
        for t_v in t_vs:
            print 'testing t = %.2f'%t_v
            my_hypo = hypo(t_v, x_v_true, y_v_true, z_v_true, theta=theta_true, phi=phi_true, trck_energy=trck_energy_true, cscd_energy=cscd_energy_true, cscd_e_scale=cscd_e_scale, trck_e_scale=trck_e_scale)
            my_hypo.set_binning(t_bin_edges, r_bin_edges, theta_bin_edges, phi_bin_edges)
            llh, noise = get_llh(my_hypo, t, x, y, z, q, string, om)
            llhs.append(llh)
            noises.append(noise)
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(t_vs, llhs)
        #ax.plot(t_vs, noises)
        ax.set_ylabel('llh')
        ax.set_xlabel('Vertex t (ns)')
        #truth
        ax.axvline(t_v_true, color='r')
        ax.axvline(ML_recos[idx].v[0], color='g')
        ax.axvline(SPE_recos[idx].v[0], color='m')
        ax.set_title('Event %i, E_cscd = %.2f GeV, E_trck = %.2f GeV'%(evt, cscd_energy_true, trck_energy_true)) 
        plt.savefig('t_%s.png'%evt,dpi=150)

    if do_theta:
        # scan theta
        thetas = np.linspace(0, np.pi, n_scan_points)
        llhs = []
        for theta in thetas:
            print 'testing theta = %.2f'%theta
            my_hypo = hypo(t_v_true, x_v_true, y_v_true, z_v_true, theta=theta, phi=phi_true, trck_energy=trck_energy_true, cscd_energy=cscd_energy_true, cscd_e_scale=cscd_e_scale, trck_e_scale=trck_e_scale)
            my_hypo.set_binning(t_bin_edges, r_bin_edges, theta_bin_edges, phi_bin_edges)
            llh, noise = get_llh(my_hypo, t, x, y, z, q, string, om)
            llhs.append(llh)
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(thetas, llhs)
        ax.set_ylabel('llh')
        ax.set_xlabel('theta (rad)')
        #truth
        ax.axvline(theta_true, color='r')
        ax.axvline(ML_recos[idx].theta, color='g')
        ax.axvline(SPE_recos[idx].theta, color='m')
        ax.set_title('Event %i, E_cscd = %.2f GeV, E_trck = %.2f GeV'%(evt, cscd_energy_true, trck_energy_true)) 
        plt.savefig('theta_%s.png'%evt,dpi=150)

    if do_phi:
        # scan phi
        phis = np.linspace(0, 2*np.pi, n_scan_points)
        llhs = []
        for phi in phis:
            print 'testing phi = %.2f'%phi
            my_hypo = hypo(t_v_true, x_v_true, y_v_true, z_v_true, theta=theta_true, phi=phi, trck_energy=trck_energy_true, cscd_energy=cscd_energy_true, cscd_e_scale=cscd_e_scale, trck_e_scale=trck_e_scale)
            my_hypo.set_binning(t_bin_edges, r_bin_edges, theta_bin_edges, phi_bin_edges)
            llh, noise = get_llh(my_hypo, t, x, y, z, q, string, om)
            llhs.append(llh)
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(phis, llhs)
        ax.set_ylabel('llh')
        ax.set_xlabel('phi (rad)')
        #truth
        ax.axvline(phi_true, color='r')
        ax.axvline(ML_recos[idx].phi, color='g')
        ax.axvline(SPE_recos[idx].phi, color='m')
        ax.set_title('Event %i, E_cscd = %.2f GeV, E_trck = %.2f GeV'%(evt, cscd_energy_true, trck_energy_true)) 
        plt.savefig('phi_%s.png'%evt,dpi=150)

    if do_cscd_energy:
        # scan cscd_energy
        cscd_energys = np.linspace(0, 5.*cscd_energy_true, n_scan_points)
        llhs = []
        for cscd_energy in cscd_energys:
            print 'testing cscd_energy = %.2f'%cscd_energy
            my_hypo = hypo(t_v_true, x_v_true, y_v_true, z_v_true, theta=theta_true, phi=phi_true, trck_energy=trck_energy_true, cscd_energy=cscd_energy, cscd_e_scale=cscd_e_scale, trck_e_scale=trck_e_scale)
            my_hypo.set_binning(t_bin_edges, r_bin_edges, theta_bin_edges, phi_bin_edges)
            llh, noise = get_llh(my_hypo, t, x, y, z, q, string, om)
            llhs.append(llh)
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(cscd_energys, llhs)
        ax.set_ylabel('llh')
        ax.set_xlabel('cscd_energy (GeV)')
        #truth
        ax.axvline(cscd_energy_true, color='r')
        ax.set_title('Event %i, E_cscd = %.2f GeV, E_trck = %.2f GeV'%(evt, cscd_energy_true, trck_energy_true)) 
        plt.savefig('cscd_energy_%s.png'%evt,dpi=150)

    if do_trck_energy:
        # scan trck_energy
        trck_energys = np.linspace(0, 50, n_scan_points)
        llhs = []
        for trck_energy in trck_energys:
            print 'testing trck_energy = %.2f'%trck_energy
            my_hypo = hypo(t_v_true, x_v_true, y_v_true, z_v_true, theta=theta_true, phi=phi_true, trck_energy=trck_energy, cscd_energy=cscd_energy_true, cscd_e_scale=cscd_e_scale, trck_e_scale=trck_e_scale)
            my_hypo.set_binning(t_bin_edges, r_bin_edges, theta_bin_edges, phi_bin_edges)
            llh, noise = get_llh(my_hypo, t, x, y, z, q, string, om)
            llhs.append(llh)
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(trck_energys, llhs)
        ax.set_ylabel('llh')
        ax.set_xlabel('trck_energy (GeV)')
        #truth
        ax.axvline(trck_energy_true, color='r')
        ax.set_title('Event %i, E_cscd = %.2f GeV, E_trck = %.2f GeV'%(evt, cscd_energy_true, trck_energy_true)) 
        plt.savefig('trck_energy_%s.png'%evt,dpi=150)


    if do_xz:
        x_points = 51
        y_points = 51
        x_vs = np.linspace(x_v_true - 150, x_v_true + 150, x_points)
        z_vs = np.linspace(z_v_true - 100, z_v_true + 100, y_points)
        llhs = []
        for z_v in z_vs:
            for x_v in x_vs:
                #print 'testing z = %.2f, x = %.2f'%(z_v, x_v)
                my_hypo = hypo(t_v_true, x_v, y_v_true, z_v, theta=theta_true, phi=phi_true, trck_energy=trck_energy_true, cscd_energy=cscd_energy_true, cscd_e_scale=cscd_e_scale, trck_e_scale=trck_e_scale)
                my_hypo.set_binning(t_bin_edges, r_bin_edges, theta_bin_edges, phi_bin_edges)
                llh, noise = get_llh(my_hypo, t, x, y, z, q, string, om)
                print ' z = %.2f, x = %.2f : llh = %.2f'%(z_v, x_v, llh)
                llhs.append(llh)
        plt.clf()
        # [z, x]
        llhs = np.array(llhs)
        llhs = llhs.reshape(y_points, x_points)

        x_edges = np.linspace(x_vs[0] - np.diff(x_vs)[0]/2., x_vs[-1] + np.diff(x_vs)[0]/2., len(x_vs) + 1)
        z_edges = np.linspace(z_vs[0] - np.diff(z_vs)[0]/2., z_vs[-1] + np.diff(z_vs)[0]/2., len(z_vs) + 1)

        xx, yy = np.meshgrid(x_edges, z_edges)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        mg = ax.pcolormesh(xx, yy, llhs, cmap=cmap)
        ax.set_ylabel('Vertex z (m)')
        ax.set_xlabel('Vertex x (m)')
        ax.set_xlim((x_edges[0], x_edges[-1]))
        ax.set_ylim((z_edges[0], z_edges[-1]))
        #truth
        ax.axvline(x_v_true, color='r')
        ax.axhline(z_v_true, color='r')
        ax.set_title('Event %i, E_cscd = %.2f GeV, E_trck = %.2f GeV'%(evt, cscd_energy_true, trck_energy_true)) 
        plt.savefig('xz_%s.png'%evt,dpi=150)
        
    if do_thetaphi:
        x_points = 50
        y_points = 50
        theta_edges = np.linspace(0, np.pi, x_points + 1)
        phi_edges = np.linspace(0, 2*np.pi, y_points + 1)
        
        thetas = 0.5 * (theta_edges[:-1] + theta_edges[1:])
        phis = 0.5 * (phi_edges[:-1] + phi_edges[1:])

        llhs = []
        for phi in phis:
            for theta in thetas:
                print 'testing phi = %.2f, theta = %.2f'%(phi, theta)
                my_hypo = hypo(t_v_true, x_v_true, y_v_true, z_v_true, theta=theta, phi=phi, trck_energy=trck_energy_true, cscd_energy=cscd_energy_true, cscd_e_scale=cscd_e_scale, trck_e_scale=trck_e_scale)
                my_hypo.set_binning(t_bin_edges, r_bin_edges, theta_bin_edges, phi_bin_edges)
                llh, noise = get_llh(my_hypo, t, x, y, z, q, string, om)
                print 'llh = %.2f'%llh
                llhs.append(llh)
        plt.clf()
        llhs = np.array(llhs)
        # will be [phi, theta]
        llhs = llhs.reshape(y_points, x_points)
        

        xx, yy = np.meshgrid(theta_edges, phi_edges)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        mg = ax.pcolormesh(xx, yy, llhs, cmap=cmap)
        ax.set_xlabel(r'$\theta$ (rad)')
        ax.set_ylabel(r'$\phi$ (rad)')
        ax.set_xlim((theta_edges[0], theta_edges[-1]))
        ax.set_ylim((phi_edges[0], phi_edges[-1]))
        #truth
        ax.axvline(theta_true, color='r')
        ax.axhline(phi_true, color='r')
        ax.set_title('Event %i, E_cscd = %.2f GeV, E_trck = %.2f GeV'%(evt, cscd_energy_true, trck_energy_true)) 
        plt.savefig('thetaphi_%s.png'%evt,dpi=150)

    if do_thetaz:
        x_points = 51
        y_points = 51
        theta_edges = np.linspace(0, np.pi, x_points + 1)
        z_edges = np.linspace(z_v_true - 20, z_v_true + 20, y_points + 1)
        
        thetas = 0.5 * (theta_edges[:-1] + theta_edges[1:])
        zs = 0.5 * (z_edges[:-1] + z_edges[1:])

        llhs = []
        for z_v in zs:
            for theta in thetas:
                print 'testing z = %.2f, theta = %.2f'%(z_v, theta)
                my_hypo = hypo(t_v_true, x_v_true, y_v_true, z_v, theta=theta, phi=phi_true, trck_energy=trck_energy_true, cscd_energy=cscd_energy_true, cscd_e_scale=cscd_e_scale, trck_e_scale=trck_e_scale)
                my_hypo.set_binning(t_bin_edges, r_bin_edges, theta_bin_edges, phi_bin_edges)
                llh, noise = get_llh(my_hypo, t, x, y, z, q, string, om)
                print 'llh = %.2f'%llh
                llhs.append(llh)
        plt.clf()
        llhs = np.array(llhs)
        # will be [z, theta]
        llhs = llhs.reshape(y_points, x_points)
        

        xx, yy = np.meshgrid(theta_edges, z_edges)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        mg = ax.pcolormesh(xx, yy, llhs, cmap=cmap)
        ax.set_xlabel(r'$\theta$ (rad)')
        ax.set_ylabel(r'Vertex z (m)')
        ax.set_xlim((theta_edges[0], theta_edges[-1]))
        ax.set_ylim((z_edges[0], z_edges[-1]))
        #truth
        ax.axvline(theta_true, color='r')
        ax.axhline(z_v_true, color='r')
        ax.set_title('Event %i, E_cscd = %.2f GeV, E_trck = %.2f GeV'%(evt, cscd_energy_true, trck_energy_true)) 
        plt.savefig('thetaz_%s.png'%evt,dpi=150)

    # exit after one event
    #sys.exit()
outfile.close()
