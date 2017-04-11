import numpy as np
import pyfits
import h5py
import sys
from argparse import ArgumentParser
from hypothesis.hypo import hypo, PowerAxis
#from likelihood.particles import particle, particle_array
from particles import particle, particle_array
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

parser = ArgumentParser(description='''make 2d event pictures''')
parser.add_argument('-f', '--file', metavar='H5_FILE', type=str, help='input HDF5 file',
                    default='/fastio/icecube/deepcore/data/MSU_sample/level5pt/numu/14600/icetray_hdf5/Level5pt_IC86.2013_genie_numu.014600.000000.hdf5')
parser.add_argument('-i', '--index', default=0, type=int, help='index offset for event to start with')
args = parser.parse_args()

# --- load tables ---
# tables are not binned in phi, but we will do so for the hypo, therefore need to apply norm
n_phi_bins = 20.
norm = 1./n_phi_bins

# load photon tables (r, cz, t) -> (t, r, cz)
IC = {}
DC = {}
for dom in range(60):
    table = pyfits.open('tables/tables/summed/retro_nevts1000_IC_DOM%i_r_cz_t.fits'%dom)
    IC[dom] = np.rollaxis(table[0].data, 2, 0) * norm
    table = pyfits.open('tables/tables/summed/retro_nevts1000_DC_DOM%i_r_cz_t.fits'%dom)
    DC[dom] = np.rollaxis(table[0].data, 2, 0) * norm

# need to change the tables into expecte n-photons:

#for key, val in IC.items():
#    print 'IC DOM %i, sum = %.2f'%(key, val.sum())
#for key, val in DC.items():
#    print 'DC DOM %i, sum = %.2f'%(key, val.sum())

# construct binning: same as tables (ToDo: should assert that)
t_bin_edges = np.linspace(0, 3e3, 301)
r_bin_edges = PowerAxis(0, 400, 200, 2)
theta_bin_edges = np.arccos(np.linspace(-1, 1, 41))[::-1]
phi_bin_edges = np.linspace(0, 2*np.pi, n_phi_bins)



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
    phi_true = neutrinos[idx].phi
    cscd_energy_true = cascades[idx].energy
    trck_energy_true = tracks[idx].energy

    z_vs = np.linspace(-500, 0, 101)
    print 'True event info:'
    print 'time = %.2f ns'%t_v_true
    print 'vertex = (%.2f, %.2f, %.2f)'%(x_v_true, y_v_true, z_v_true)
    print 'theta, phi = (%.2f, %.2f)'%(theta_true, phi_true)
    print 'E_cscd, E_trck (GeV) = %.2f, %.2f'%(cscd_energy_true, trck_energy_true)
    llhs = []
    # construct hypo
    for z_v in z_vs:
        print 'testing z = %.2f'%z_v
	my_hypo = hypo(t_v_true, x_v_true, y_v_true, z_v, theta=theta_true, phi=phi_true, trck_energy=trck_energy_true, cscd_energy=cscd_energy_true)
	my_hypo.set_binning(t_bin_edges, r_bin_edges, theta_bin_edges, phi_bin_edges)
	llh = 0
	# loop over hits
	for hit in range(len(t)):
	    # for every DOM + hit:
            #print 'getting matrix for hit %i'%hit
            # t, r ,cz, phi 
            #print 'hit at %.2f ns at (%.2f, %.2f, %.2f)'%(t[hit], x[hit], y[hit], z[hit])
	    z_matrix = my_hypo.get_z_matrix(t[hit], x[hit], y[hit], z[hit])
            if string[hit] < 78:
                gamma_map = IC[om[hit]]
            else:
                gamma_map = DC[om[hit]]
            # get max llh between z_matrix and gamma_map
            min_delta = 1e7
            for phi_idx in range(z_matrix.shape[3]):
	        hypo_gammas = z_matrix[:, :, :, phi_idx]
                #print 'non-zero hypo = ', (hypo_gammas != 0.).sum()
                #print 'non-zero map  = ', (gamma_map != 0.).sum()
                non_zero = (hypo_gammas != 0.) * (gamma_map != 0.)
                if non_zero.any():
                    #print 'found non-zero'
                    deltas = abs(hypo_gammas[non_zero] - 1./gamma_map[non_zero])
                    #hit_gammas = 1./gamma_map 
                    min_delta = min(min_delta, np.min(deltas))
            if min_delta > 0:
                llh += np.log(min_delta)
	llhs.append(llh)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(z_vs, llhs)
    plt.axvline(z_v_true, color='r')
    plt.savefig('zvs.png',dpi=150)
    print z_vs
    print llhs
    # exit after one event
    sys.exit()
