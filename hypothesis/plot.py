from track_hypo import PowerAxis
from track_hypo import get_bin_index
from track_hypo import get_track_lengths
from hypo_fast import hypo
import numpy as np
import math

if __name__ == '__main__':

    # for plotting
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from mpl_toolkits.mplot3d import Axes3D
    import time
    # plot setup
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(221,projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    plt_lim = 50

    ax.set_xlim((-plt_lim,plt_lim))
    ax.set_ylim((-plt_lim,plt_lim))
    ax.set_zlim((-plt_lim,plt_lim))
    ax.grid(True)


    # same as CLsim
    t_bin_edges = np.linspace(0, 500, 51)
    r_bin_edges = PowerAxis(0, 200, 20, 2)
    theta_bin_edges = np.arccos(np.linspace(-1, 1, 51))[::-1]
    phi_bin_edges = np.linspace(0, 2*np.pi, 37)

    my_hypo = hypo(10., 0., 4., 0., theta=0.57, phi=5.3, trck_energy=25., cscd_energy=0.)
    my_hypo.set_binning(t_bin_edges, r_bin_edges, theta_bin_edges, phi_bin_edges)

    # kevin array
    t0 = time.time()
    z_kevin_sparse = get_track_lengths(10e-9, 0., 4., 0., 0.57, 5.3, 113.636)
    print 'took%.2f ms to calculate z_kevin-matrix'%((time.time() - t0)*1000)
    z_kevin = np.zeros((len(t_bin_edges) - 1, len(r_bin_edges) - 1, len(theta_bin_edges) - 1, len(phi_bin_edges) - 1))
    for hit in z_kevin_sparse:
        #print hit
        idx, count = hit
        z_kevin[idx] = count * 2451.4544553
    print 'total number of photons in kevin matrix = %i (%.2f %%)'%(z_kevin.sum(), z_kevin.sum()/my_hypo.tot_photons*100.)

    # plot the track as a line
    x_0, y_0, z_0 = my_hypo.track.point(my_hypo.track.t0)
    #print 'track vertex', x_0, y_0, z_0
    x_e, y_e, z_e  = my_hypo.track.point(my_hypo.track.t0 + my_hypo.track.dt)
    ax.plot([x_0,x_e],[y_0,y_e],zs=[z_0,z_e])
    ax.plot([-plt_lim,-plt_lim],[y_0,y_e],zs=[z_0,z_e],alpha=0.3,c='k')
    ax.plot([x_0,x_e],[plt_lim,plt_lim],zs=[z_0,z_e],alpha=0.3,c='k')
    ax.plot([x_0,x_e],[y_0,y_e],zs=[-plt_lim,-plt_lim],alpha=0.3,c='k')
    
    t0 = time.time()
    hits, n_t, n_p, n_l = my_hypo.get_matrices(0., 0., 0., 0.)
    print 'took %.2f ms to calculate z-matrix'%((time.time() - t0)*1000)
    z = np.zeros((len(t_bin_edges) - 1, len(r_bin_edges) - 1, len(theta_bin_edges) - 1, len(phi_bin_edges) - 1))
    for hit in hits:
        #print hit
        idx, count = hit
        z[idx] = count
    print 'total number of photons in matrix = %i (%.2f %%)'%(z.sum(), z.sum()/my_hypo.tot_photons*100.)

    print 'total_residual = ',(z - z_kevin).sum()/z.sum()

    #cmap = 'gnuplot2_r'
    cmap = mpl.cm.get_cmap('gnuplot_r')
    cmap.set_under('w')
    cmap.set_bad('w')

    tt, yy = np.meshgrid(t_bin_edges, r_bin_edges)
    zz = z_kevin.sum(axis=(2,3)) - z.sum(axis=(2,3))
    mg = ax2.pcolormesh(tt, yy, zz.T, vmin=1e-7, cmap=cmap)
    ax2.set_xlabel('t')
    ax2.set_ylabel('r')

    tt, yy = np.meshgrid(t_bin_edges, theta_bin_edges)
    zz = z_kevin.sum(axis=(1,3)) - z.sum(axis=(1,3))
    mg = ax3.pcolormesh(tt, yy, zz.T, vmin=1e-7, cmap=cmap)
    ax3.set_xlabel('t')
    ax3.set_ylabel(r'$\theta$')
    ax3.set_ylim((0,np.pi))

    tt, yy = np.meshgrid(t_bin_edges, phi_bin_edges)
    zz = z_kevin.sum(axis=(1,2)) - z.sum(axis=(1,2))
    mg = ax4.pcolormesh(tt, yy, zz.T, vmin=1e-7, cmap=cmap)
    ax4.set_xlabel('t')
    ax4.set_ylabel(r'$\phi$')
    ax4.set_ylim((0,2*np.pi))

    plt.show()
    plt.savefig('hypo_diff3.png',dpi=300)
