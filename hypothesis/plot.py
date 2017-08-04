#!/usr/bin/env python

from track_hypo import PowerAxis
from track_hypo import segment_hypo
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

    my_hypo = hypo(10., 0., 40., 45., theta=2.57, phi=5.3, trck_energy=25., cscd_energy=25.)
    my_hypo.set_binning(t_bin_edges, r_bin_edges, theta_bin_edges, phi_bin_edges)

    # kevin array
    t0 = time.time()
    kevin_hypo = segment_hypo(10., 0., 40., 45., 2.57, 5.3, 25., 25.)
    kevin_hypo.set_binning(50., 20., 50., 36., 500., 200.)
    kevin_hypo.set_dom_location(0., 10., -10., 40.)
    #kevin_hypo.use_scaled_time_increments()
    #kevin_hypo.create_photon_matrix()
    kevin_hypo.vector_photon_matrix()
    print 'took %.2f ms to calculate z_kevin-matrix'%((time.time() - t0)*1000)
    #print 'number of segments: %i'%kevin_hypo.number_of_segments
    print 'number of segments: %i'%kevin_hypo.number_of_increments
    #z_kevin_sparse = kevin_hypo.z_kevin
    z_kevin_vector = kevin_hypo.indices_array
    z_kevin = np.zeros((len(t_bin_edges) - 1, len(r_bin_edges) - 1, len(theta_bin_edges) - 1, len(phi_bin_edges) - 1))
    #for hit in z_kevin_sparse:
    #    #print hit
    #    idx, count = hit
    
    #debug prints
    #    z_kevin[idx] = count
    #print z_kevin_vector
    #print kevin_hypo.variables_array
    #print kevin_hypo.t_array
    print kevin_hypo.variables_array[0, :]
    #print kevin_hypo.t_index_array
    print kevin_hypo.indices_array[0, :]
    #print kevin_hypo.variables_array[1, :]
    #print kevin_hypo.variables_array[2, :]    
    #print kevin_hypo.variables_array[6, :]
    #print kevin_hypo.indices_array[3, :]
   
    for col in xrange(kevin_hypo.number_of_increments):
        idx = (int(z_kevin_vector[0, col]), int(z_kevin_vector[1, col]), int(z_kevin_vector[2, col]), int(z_kevin_vector[3, col]))
        if z_kevin_vector[1, col] < kevin_hypo.r_max:
            z_kevin[idx] += kevin_hypo.variables_array[7, col]
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
    hits, n_t, n_p, n_l = my_hypo.get_matrices(0., 10., -10., 40.)
    print 'took %.2f ms to calculate z-matrix'%((time.time() - t0)*1000)
    z = np.zeros((len(t_bin_edges) - 1, len(r_bin_edges) - 1, len(theta_bin_edges) - 1, len(phi_bin_edges) - 1))
    for hit in hits:
        #print hit
        idx, count = hit
        z[idx] = count
    print 'total number of photons in matrix = %i (%.2f %%)'%(z.sum(), z.sum()/my_hypo.tot_photons*100.)

    print 'total_residual = ',(z - z_kevin).sum()/z.sum()

    #create differential matrix
    z_diff = z_kevin - z

    #create percent differnt matrix
    z_per = np.zeros_like(z_kevin)
    mask = z != 0
    z_per[mask] = z_kevin[mask] / z[mask] -1

    #cmap = 'gnuplot_r'
    cmap = mpl.cm.get_cmap('bwr')
    cmap.set_under('w')
    cmap.set_bad('w')

    tt, yy = np.meshgrid(t_bin_edges, r_bin_edges)
    zz = z_diff.sum(axis=(2,3))
    z_vmax = np.maximum(np.abs(np.min(zz)), np.max(zz))
    mg = ax2.pcolormesh(tt, yy, zz.T, vmin=-z_vmax, vmax=z_vmax, cmap=cmap)
    ax2.set_xlabel('t')
    ax2.set_ylabel('r')
    plt.colorbar(mg, ax=ax2)

    tt, yy = np.meshgrid(t_bin_edges, theta_bin_edges)
    zz = z_diff.sum(axis=(1,3))
    z_vmax = np.maximum(np.abs(np.min(zz)), np.max(zz))
    mg = ax3.pcolormesh(tt, yy, zz.T, vmin=-z_vmax, vmax=z_vmax, cmap=cmap)
    ax3.set_xlabel('t')
    ax3.set_ylabel(r'$\theta$')
    ax3.set_ylim((0,np.pi))
    plt.colorbar(mg, ax=ax3)

    tt, yy = np.meshgrid(t_bin_edges, phi_bin_edges)
    zz = z_diff.sum(axis=(1,2))
    z_vmax = np.maximum(np.abs(np.min(zz)), np.max(zz))
    mg = ax4.pcolormesh(tt, yy, zz.T, vmin=-z_vmax, vmax=z_vmax, cmap=cmap)
    ax4.set_xlabel('t')
    ax4.set_ylabel(r'$\phi$')
    ax4.set_ylim((0,2*np.pi))
    plt.colorbar(mg, ax=ax4)
    
    plt.show()
    plt.savefig('hypo_vector2.png',dpi=300)
