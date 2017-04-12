import numpy as np
import pyfits
import h5py
import sys
from argparse import ArgumentParser
from hypothesis.hypo_fast import hypo, PowerAxis
from particles import particle, particle_array
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numba
#from line_profiler import profile

parser = ArgumentParser(description='''make 2d event pictures''')
parser.add_argument('-f', '--file', metavar='H5_FILE', type=str, help='input HDF5 file',
                    default='/fastio/icecube/deepcore/data/MSU_sample/level5pt/numu/14600/icetray_hdf5/Level5pt_IC86.2013_genie_numu.014600.000000.hdf5')
parser.add_argument('-i', '--index', default=0, type=int, help='index offset for event to start with')
args = parser.parse_args()

# --- load tables ---
# tables are not binned in phi, but we will do so for the hypo, therefore need to apply norm
n_phi_bins = 20.
norm = 1./n_phi_bins

# load photon tables (r, cz, t) -> (-t, r, cz)
IC = {}
DC = {}
for dom in range(60):
    table = pyfits.open('tables/tables/summed/retro_nevts1000_IC_DOM%i_r_cz_t.fits'%dom)
    IC[dom] = np.flipud(np.rollaxis(table[0].data, 2, 0)) * norm
    table = pyfits.open('tables/tables/summed/retro_nevts1000_DC_DOM%i_r_cz_t.fits'%dom)
    DC[dom] = np.flipud(np.rollaxis(table[0].data, 2, 0)) * norm

# need to change the tables into expecte n-photons:

#for key, val in IC.items():
#    print 'IC DOM %i, sum = %.2f'%(key, val.sum())
#for key, val in DC.items():
#    print 'DC DOM %i, sum = %.2f'%(key, val.sum())

# construct binning: same as tables (ToDo: should assert that)
t_bin_edges = np.linspace(-3e3, 0, 301)
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

#@profile
def get_llh(hypo, t, x, y, z, q, string, om):
    llh = 0
    # loop over hits
    for hit in range(len(t)):
        # for every DOM + hit:
        #print 'getting matrix for hit %i'%hit
        # t, r ,cz, phi 
        #print 'hit at %.2f ns at (%.2f, %.2f, %.2f)'%(t[hit], x[hit], y[hit], z[hit])
        z_matrix = hypo.get_z_matrix(t[hit], x[hit], y[hit], z[hit])
        if string[hit] < 78:
            gamma_map = IC[om[hit] - 1]
        else:
            gamma_map = DC[om[hit] - 1]
        # get max llh between z_matrix and gamma_map
        chi2 = 1e9
        for element in z_matrix:
            idx, hypo_count = element
            map_count = gamma_map[idx[0:3]]
            if map_count > 0:
                g = map_count / q[hit]
                new_chi2 = (hypo_count - 1./g)**2*g
                chi2 = min(chi2, new_chi2)
        llh += chi2
    return llh

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
    # azimuth?
    phi_true = neutrinos[idx].phi
    cscd_energy_true = cascades[idx].energy
    trck_energy_true = tracks[idx].energy

    print 'True event info:'
    print 'time = %.2f ns'%t_v_true
    print 'vertex = (%.2f, %.2f, %.2f)'%(x_v_true, y_v_true, z_v_true)
    print 'theta, phi = (%.2f, %.2f)'%(theta_true, phi_true)
    print 'E_cscd, E_trck (GeV) = %.2f, %.2f'%(cscd_energy_true, trck_energy_true)

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
    #do_x =  True
    #do_y =  True
    #do_z =  True
    #do_t =  True
    #do_theta =  True
    #do_phi =  True
    #do_cscd_energy =  True
    #do_trck_energy =  True
    do_xz = True
    #do_thetaphi = True

    n_scan_points = 7


    if do_z:
        # scan z pos
        z_vs = np.linspace(z_v_true - 20, z_v_true + 20, n_scan_points)
        llhs = []
        for z_v in z_vs:
            print 'testing z = %.2f'%z_v
            my_hypo = hypo(t_v_true, x_v_true, y_v_true, z_v, theta=theta_true, phi=phi_true, trck_energy=trck_energy_true, cscd_energy=cscd_energy_true)
            my_hypo.set_binning(t_bin_edges, r_bin_edges, theta_bin_edges, phi_bin_edges)
            llh = get_llh(my_hypo, t, x, y, z, q, string, om)
            llhs.append(llh)
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(z_vs, llhs)
        ax.set_ylabel('llh')
        ax.set_xlabel('Vertex z (m)')
        #truth
        ax.axvline(z_v_true, color='r')
        plt.savefig('z_%s.png'%evt,dpi=150)

    if do_x:
        # scan x pos
        x_vs = np.linspace(x_v_true - 50, x_v_true + 50, n_scan_points)
        llhs = []
        for x_v in x_vs:
            print 'testing x = %.2f'%x_v
            my_hypo = hypo(t_v_true, x_v, y_v_true, z_v_true, theta=theta_true, phi=phi_true, trck_energy=trck_energy_true, cscd_energy=cscd_energy_true)
            my_hypo.set_binning(t_bin_edges, r_bin_edges, theta_bin_edges, phi_bin_edges)
            llh = get_llh(my_hypo, t, x, y, z, q, string, om)
            llhs.append(llh)
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x_vs, llhs)
        ax.set_ylabel('llh')
        ax.set_xlabel('Vertex x (m)')
        #truth
        ax.axvline(x_v_true, color='r')
        plt.savefig('x_%s.png'%evt,dpi=150)

    if do_y:
        # scan y pos
        y_vs = np.linspace(y_v_true - 50, y_v_true + 50, n_scan_points)
        llhs = []
        for y_v in y_vs:
            print 'testing y = %.2f'%y_v
            my_hypo = hypo(t_v_true, x_v_true, y_v, z_v_true, theta=theta_true, phi=phi_true, trck_energy=trck_energy_true, cscd_energy=cscd_energy_true)
            my_hypo.set_binning(t_bin_edges, r_bin_edges, theta_bin_edges, phi_bin_edges)
            llh = get_llh(my_hypo, t, x, y, z, q, string, om)
            llhs.append(llh)
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(y_vs, llhs)
        ax.set_ylabel('llh')
        ax.set_xlabel('Vertex y (m)')
        #truth
        ax.axvline(y_v_true, color='r')
        plt.savefig('y_%s.png'%evt,dpi=150)

    if do_t:
        # scan t pos
        t_vs = np.linspace(t_v_true - 200, t_v_true + 200, n_scan_points)
        llhs = []
        for t_v in t_vs:
            print 'testing t = %.2f'%t_v
            my_hypo = hypo(t_v, x_v_true, y_v_true, z_v_true, theta=theta_true, phi=phi_true, trck_energy=trck_energy_true, cscd_energy=cscd_energy_true)
            my_hypo.set_binning(t_bin_edges, r_bin_edges, theta_bin_edges, phi_bin_edges)
            llh = get_llh(my_hypo, t, x, y, z, q, string, om)
            llhs.append(llh)
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(t_vs, llhs)
        ax.set_ylabel('llh')
        ax.set_xlabel('Vertex t (ns)')
        #truth
        ax.axvline(t_v_true, color='r')
        plt.savefig('t_%s.png'%evt,dpi=150)

    if do_theta:
        # scan theta
        thetas = np.linspace(0, np.pi, n_scan_points)
        llhs = []
        for theta in thetas:
            print 'testing theta = %.2f'%theta
            my_hypo = hypo(t_v_true, x_v_true, y_v_true, z_v_true, theta=theta, phi=phi_true, trck_energy=trck_energy_true, cscd_energy=cscd_energy_true)
            my_hypo.set_binning(t_bin_edges, r_bin_edges, theta_bin_edges, phi_bin_edges)
            llh = get_llh(my_hypo, t, x, y, z, q, string, om)
            llhs.append(llh)
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(thetas, llhs)
        ax.set_ylabel('llh')
        ax.set_xlabel('theta (rad)')
        #truth
        ax.axvline(theta_true, color='r')
        plt.savefig('theta_%s.png'%evt,dpi=150)

    if do_phi:
        # scan phi
        phis = np.linspace(0, 2*np.pi, n_scan_points)
        llhs = []
        for phi in phis:
            print 'testing phi = %.2f'%phi
            my_hypo = hypo(t_v_true, x_v_true, y_v_true, z_v_true, theta=theta_true, phi=phi, trck_energy=trck_energy_true, cscd_energy=cscd_energy_true)
            my_hypo.set_binning(t_bin_edges, r_bin_edges, theta_bin_edges, phi_bin_edges)
            llh = get_llh(my_hypo, t, x, y, z, q, string, om)
            llhs.append(llh)
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(phis, llhs)
        ax.set_ylabel('llh')
        ax.set_xlabel('phi (rad)')
        #truth
        ax.axvline(phi_true, color='r')
        plt.savefig('phi_%s.png'%evt,dpi=150)

    if do_cscd_energy:
        # scan cscd_energy
        cscd_energys = np.linspace(0, 500, n_scan_points)
        llhs = []
        for cscd_energy in cscd_energys:
            print 'testing cscd_energy = %.2f'%cscd_energy
            my_hypo = hypo(t_v_true, x_v_true, y_v_true, z_v_true, theta=theta_true, phi=phi_true, trck_energy=trck_energy_true, cscd_energy=cscd_energy)
            my_hypo.set_binning(t_bin_edges, r_bin_edges, theta_bin_edges, phi_bin_edges)
            llh = get_llh(my_hypo, t, x, y, z, q, string, om)
            llhs.append(llh)
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(cscd_energys, llhs)
        ax.set_ylabel('llh')
        ax.set_xlabel('cscd_energy (GeV)')
        #truth
        ax.axvline(cscd_energy_true, color='r')
        plt.savefig('cscd_energy_%s.png'%evt,dpi=150)

    if do_trck_energy:
        # scan trck_energy
        trck_energys = np.linspace(0, 50, n_scan_points)
        llhs = []
        for trck_energy in trck_energys:
            print 'testing trck_energy = %.2f'%trck_energy
            my_hypo = hypo(t_v_true, x_v_true, y_v_true, z_v_true, theta=theta_true, phi=phi_true, trck_energy=trck_energy, cscd_energy=cscd_energy_true)
            my_hypo.set_binning(t_bin_edges, r_bin_edges, theta_bin_edges, phi_bin_edges)
            llh = get_llh(my_hypo, t, x, y, z, q, string, om)
            llhs.append(llh)
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(trck_energys, llhs)
        ax.set_ylabel('llh')
        ax.set_xlabel('trck_energy (GeV)')
        #truth
        ax.axvline(trck_energy_true, color='r')
        plt.savefig('trck_energy_%s.png'%evt,dpi=150)


    if do_xz:
        x_points = 21
        y_points = 21
        x_vs = np.linspace(x_v_true - 150, x_v_true + 150, x_points)
        z_vs = np.linspace(z_v_true - 100, z_v_true + 100, y_points)
        llhs = []
        for z_v in z_vs:
            for x_v in x_vs:
                #print 'testing z = %.2f, x = %.2f'%(z_v, x_v)
                my_hypo = hypo(t_v_true, x_v, y_v_true, z_v, theta=theta_true, phi=phi_true, trck_energy=trck_energy_true, cscd_energy=cscd_energy_true)
                my_hypo.set_binning(t_bin_edges, r_bin_edges, theta_bin_edges, phi_bin_edges)
                llh = get_llh(my_hypo, t, x, y, z, q, string, om)
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
        mg = ax.pcolormesh(xx, yy, llhs, cmap='YlGnBu_r')
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
        x_points = 7
        y_points = 7
        theta_edges = np.linspace(0, np.pi, x_points + 1)
        phi_edges = np.linspace(0, 2*np.pi, y_points + 1)
        
        thetas = 0.5 * (theta_edges[:-1] + theta_edges[1:])
        phis = 0.5 * (phi_edges[:-1] + phi_edges[1:])

        llhs = []
        for phi in phis:
            for theta in thetas:
                print 'testing phi = %.2f, theta = %.2f'%(phi, theta)
                my_hypo = hypo(t_v_true, x_v_true, y_v_true, z_v_true, theta=theta, phi=phi, trck_energy=trck_energy_true, cscd_energy=cscd_energy_true)
                my_hypo.set_binning(t_bin_edges, r_bin_edges, theta_bin_edges, phi_bin_edges)
                llh = get_llh(my_hypo, t, x, y, z, q, string, om)
                print 'llh = %.2f'%llh
                llhs.append(llh)
        plt.clf()
        llhs = np.array(llhs)
        # will be [phi, theta]
        llhs = llhs.reshape(y_points, x_points)
        

        xx, yy = np.meshgrid(theta_edges, phi_edges)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        mg = ax.pcolormesh(xx, yy, llhs, cmap='YlGnBu_r')
        ax.set_xlabel(r'$\theta$ (rad)')
        ax.set_ylabel(r'$\phi$ (rad)')
        ax.set_xlim((theta_edges[0], theta_edges[-1]))
        ax.set_ylim((phi_edges[0], phi_edges[-1]))
        #truth
        ax.axvline(theta_true, color='r')
        ax.axhline(phi_true, color='r')
        ax.set_title('Event %i, E_cscd = %.2f GeV, E_trck = %.2f GeV'%(evt, cscd_energy_true, trck_energy_true)) 
        plt.savefig('thetaphi_%s.png'%evt,dpi=150)

    # exit after one event
    #sys.exit()
