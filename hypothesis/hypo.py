import numpy as np
from icecube.clsim.traysegments.common import parseIceModel
from icecube.clsim import NumberOfPhotonsPerMeter
from icecube import clsim
from os.path import expandvars
from icecube.icetray import I3Units


class track(object):
    ''' class to calculate track positons for a given track hypo '''
    def __init__(self, t_v, x_v, y_v, z_v, theta, phi, length):
        ''' initialize with track parameters :
            t_v : time (ns)
            t_x, t_y, t_z : vertex position (m)
            theta : zenith (rad)
            phi : azimuth (rad)
            length : track length (m)
            '''
        # speed of light
        self.c = 0.299792 # m/ns
        # vertex position, angle and length
        self.t_v = t_v
        self.x_v = x_v
        self.y_v = y_v
        self.z_v = z_v
        self.phi = phi
        self.theta = theta
        self.length = length
        self.dt = self.length / self.c
        # precalculated quantities
        self.sinphi = np.sin(phi)
        self.cosphi = np.cos(phi)
        self.tanphi = np.tan(phi)
        self.sintheta = np.sin(theta)
        self.costheta = np.cos(theta)
        self.set_origin(0, 0, 0, 0)

    def set_origin(self, t, x, y, z):
        ''' displace track '''
        self.t_o = t
        self.x_o = x
        self.y_o = y
        self.z_o = z

    @property
    def t0(self):
        ''' transalted t '''
        return self.t_v - self.t_o
    @property
    def x0(self):
        ''' translated x '''
        return self.x_v - self.x_o
    @property
    def y0(self):
        ''' translated y '''
        return self.y_v - self.y_o
    @property
    def z0(self):
        ''' translated z '''
        return self.z_v - self.z_o

    def point(self, t):
        '''return point on track for a given time'''
        # make sure time is valid
        assert (self.t0 <= t) and (t <= self.t0 + self.dt)
        dt = (t - self.t0)
        dr = self.c * dt
        x = self.x0 + dr * self.sintheta * self.cosphi
        y = self.y0 + dr * self.sintheta * self.sinphi
        z = self.z0 + dr * self.costheta
        return (x, y, z)

    def extent(self, t_low, t_high):
        '''return maximum time extent of track in a given time interval'''
        if (self.t0 + self.dt >= t_low) and (t_high >= self.t0):
            # then we have overlap
            t_0 = max(t_low, self.t0)
            t_1 = min(t_high, self.t0+self.dt)
            return (t_0, t_1)
        else:
            return None

    @property
    def tb(self):
        '''closest time to origin (smallest R)'''
        return self.t0 - (self.x0*self.sintheta*self.cosphi + self.y0*self.sintheta*self.sinphi + self.z0*self.costheta) / self.c

    @property
    def ts(self):
        '''time with smallest theta'''
        if self.x0 == 0. and self.y0 == 0. and self.z0 == 0.:
           return self.t0
        rho = ( \
            - self.costheta*(self.x0**2 + self.y0**2) \
            + self.sintheta*self.z0*(self.x0*self.cosphi + self.y0*self.sinphi)) \
        /( \
            + self.sintheta*self.costheta*(self.x0*self.cosphi + self.y0*self.sinphi) \
            - self.z0*(self.sintheta**2) \
            )
        return self.t0 + rho / self.c

    def rho_of_t(self, t):
        if t < self.t0 or t > self.t0 + self.dt:
            return -1
        else:
            return (t - self.t0) * self.c

    def rho_of_phi(self, phi):
        '''track parameter rho for a give phi'''
        sin = np.sin(phi)
        cos = np.cos(phi)
        return (sin*self.x0 - cos*self.y0)/(cos*self.sinphi*self.sintheta - sin*self.cosphi*self.sintheta)

    def get_M(self, T):
        '''helper function'''
        S = - self.x0**2*self.sintheta**2*self.sinphi**2 \
            - self.y0**2*self.cosphi**2*self.sintheta**2 \
            + 2*self.x0*self.y0*self.sinphi*self.sintheta**2*self.cosphi \
            + np.tan(T)**2*( \
                + (self.x0**2 + self.y0**2)*self.costheta**2 \
                + self.z0**2*self.sintheta**2 \
                - 2*self.z0*self.sintheta*self.costheta*(self.x0*self.cosphi + self.y0*self.sinphi) \
            )
        if S < 0:
            return 0.
        else:
            return np.sqrt(S)

    def rho_of_theta_neg(self, T):
        '''track parameter rho for a given theta, solution 1'''
        M = self.get_M(T)
        d = -self.sintheta**2 + self.costheta**2*np.tan(T)**2
        if d == 0: return np.inf
        rho = ( \
            + self.x0*self.sintheta*self.cosphi \
            + self.y0*self.sinphi*self.sintheta \
            - self.z0*self.costheta*np.tan(T)**2 \
            + M)/ \
            d
        return rho

    def rho_of_theta_pos(self, T):
        '''track parameter rho for a given theta, solution 2'''
        M = self.get_M(T)
        d = -self.sintheta**2 + self.costheta**2*np.tan(T)**2
        if d == 0: return np.inf
        rho = ( \
            + self.x0*self.sintheta*self.cosphi \
            + self.y0*self.sinphi*self.sintheta \
            - self.z0*self.costheta*np.tan(T)**2 \
            - M)/ \
            d
        return rho

    def get_A(self, R):
        '''helper function'''
        S = R**2 \
            + self.x0**2*(self.cosphi**2*self.sintheta**2 - 1) \
            + self.y0**2*(self.sinphi**2*self.sintheta**2 - 1) \
            + self.z0**2*(self.costheta**2 - 1) \
            + 2*self.x0*self.y0*self.sinphi*self.sintheta**2*self.cosphi \
            + 2*self.x0*self.z0*self.cosphi*self.sintheta*self.costheta \
            + 2*self.y0*self.z0*self.sinphi*self.sintheta*self.costheta
        if S < 0:
            return 0.
        else:
            return np.sqrt(S)

    def rho_of_r_pos(self, R):
        '''track parameter rho for a given r, solution 1'''
        A = self.get_A(R)
        return A - self.x0*self.sintheta*self.cosphi - self.y0*self.sinphi*self.sintheta - self.z0*self.costheta

    def rho_of_r_neg(self, R):
        '''track parameter rho for a given r, solution 2'''
        A = self.get_A(R)
        return - A - self.x0*self.sintheta*self.cosphi - self.y0*self.sinphi*self.sintheta - self.z0*self.costheta



class hypo(object):
    ''' craate the hypo for a given set of parameters and then retrieve maps (z-matrix) that contain
        the expected photon sources in each cell for a spherical coordinate system that has its origin
        at the DOM-hit point.
        The cells are devided up according to the binning that must be set.'''

    def __init__(self, t_v, x_v, y_v, z_v, theta, phi, trck_energy, cscd_energy):
        ''' provide hypo with vertex (_v), direction and trck and cscd energy
            
            t_v : time (ns)
            x_v, y_v, z_v : vertex pos. (m)
            theta : zenith (rad)
            phi : azimuth (rad)
            trck_energy : track energy (GeV)
            cscd_energy : cascade energy (GeV)
            '''

        # convert trck energy to length by (ToDo check this)
        length = 15. / 3.3 * trck_energy
        self.track = track(t_v, x_v, y_v, z_v, theta, phi, length)
	mediumProperties = parseIceModel(expandvars("$I3_SRC/clsim/resources/ice/spice_mie"), disableTilt=True)
	domAcceptance = clsim.GetIceCubeDOMAcceptance()
	self.photons_per_meter = NumberOfPhotonsPerMeter(mediumProperties.GetPhaseRefractiveIndex(0),
						    domAcceptance,
						    mediumProperties.GetMinWavelength(),
						    mediumProperties.GetMaxWavelength()
						    )
	density = mediumProperties.GetMediumDensity()
	# PPC parametrerization
	nph = 5.21 * 0.924 / density
        self.cscd_photons = cscd_energy * nph * self.photons_per_meter * I3Units.g / I3Units.cm3
        self.t_bin_edges = None
        self.r_bin_edges = None
        self.phi_bin_edges = None
        self.theta_bin_edges = None

        print 'total number of photons from cscd (%s GeV) = %i'%(cscd_energy, self.cscd_photons)
        print 'total number of photons from trck (%s GeV) = %i'%(trck_energy, (self.track.length * self.photons_per_meter))

    def set_binning(self, t_bin_edges, r_bin_edges, theta_bin_edges, phi_bin_edges):
        ''' set the binning of the spherical coordinates with bin_edges'''
        self.t_bin_edges = t_bin_edges
        self.r_bin_edges = r_bin_edges
        self.theta_bin_edges = theta_bin_edges
        self.phi_bin_edges = phi_bin_edges

    # coord. transforms
    @staticmethod
    def cr(x, y, z):
        '''r in spehrical coord., given x,y,z'''
        return np.sqrt(x**2 + y**2 + z**2)
    @staticmethod
    def cphi(x, y, z):
        '''phi in spehrical coord., given x,y,z'''
        v = np.arctan2(y, x)
        if v < 0: v += 2*np.pi
        return v
    @staticmethod
    def ctheta(x, y, z):
        '''theta in spehrical coord., given x,y,z'''
        if x == 0 and y == 0 and z == 0:
           return 0
        return np.arccos(z/np.sqrt(x**2 + y**2 + z**2))

    # bin correlators:
    def correlate_theta(self, bin_edges, t, rho_extent):
        '''get the track parameter rho intervals, which lie inside bins'''
        intervals = []
        for i in range(len(bin_edges) - 1):
            # get interval overlaps
            # from these two intervals:
            if t is None:
                intervals.append(None)
                continue
            b = (bin_edges[i], bin_edges[i+1])
            if abs(t[0] - t[1]) < 1e-7 and (b[0] <= t[0]) and (t[0] < b[1]):
                # along coordinate axis
                intervals.append(rho_extent)
            elif (b[0] <= t[1]) and (t[0] < b[1]) and (t[0] < t[1]):
                val_high = min(b[1], t[1])
                val_low = max(b[0], t[0])
                if b[0] < np.pi/2.:
                    theta_low = self.track.rho_of_theta_pos(val_low)
                    theta_high = self.track.rho_of_theta_pos(val_high)
                else:
                    theta_low = self.track.rho_of_theta_neg(val_low)
                    theta_high = self.track.rho_of_theta_neg(val_high)
                intervals.append(sorted((theta_low, theta_high)))
            elif (b[0] <= t[0]) and (t[1] < b[1]) and (t[1] < t[0]):
                val_high = min(b[1], t[0])
                val_low = max(b[0], t[1])
                if b[0] < np.pi/2.:
                    theta_low = self.track.rho_of_theta_neg(val_low)
                    theta_high = self.track.rho_of_theta_neg(val_high)
                else:
                    theta_low = self.track.rho_of_theta_pos(val_low)
                    theta_high = self.track.rho_of_theta_pos(val_high)
                intervals.append(sorted((theta_low, theta_high)))
            else:
                intervals.append(None)
        return intervals

    def correlate_phi(self, bin_edges, t, rho_extent):
        '''get the track parameter rho intervals, which lie inside bins'''
        # phi intervals
        intervals = []
        for i in range(len(bin_edges) - 1):
            # get interval overlaps
            # from these two intervals:
            b = (bin_edges[i], bin_edges[i+1])
            if abs(t[0] - t[1]) < 1e-7 and (b[0] <= t[0]) and (t[0] < b[1]):
                # along coordinate axis
                intervals.append(rho_extent)
            elif t[0] <= t[1] and (b[0] <= t[1]) and (t[0] < b[1]):
                phi_high = min(b[1], t[1])
                phi_low = max(b[0], t[0])
                r_low = self.track.rho_of_phi(phi_low)
                r_high = self.track.rho_of_phi(phi_high)
                intervals.append(sorted((r_low, r_high)))
            # crossing the 0/2pi point
            elif t[1] < t[0]:
                if b[1] >= 0 and t[1] >= b[0]:
                    phi_high = min(b[1], t[1])
                    phi_low = max(b[0], 0)
                elif 2*np.pi > b[0] and b[1] >= t[0]:
                    phi_high = min(b[1], 2*np.pi)
                    phi_low = max(b[0], t[0])
                elif b[0] <= t[1] and t[0] <= t[1]:
                    phi_high = min(b[1], t[1])
                    phi_low = max(b[0], t[0])
                else:
                    intervals.append(None)
                    continue
                r_low = self.track.rho_of_phi(phi_low)
                r_high = self.track.rho_of_phi(phi_high)
                intervals.append(sorted((r_low, r_high)))
            else:
                intervals.append(None)
        return intervals

    def correlate_r(self, bin_edges, t, pos):
        '''get the track parameter rho intervals, which lie inside bins'''
        intervals = []
        for i in range(len(bin_edges) - 1):
            # get interval overlaps
            # from these two intervals:
            b = (bin_edges[i], bin_edges[i+1])
            if (b[0] <= t[1]) and (t[0] < b[1]):
                val_high = min(b[1], t[1])
                val_low = max(b[0], t[0])
                if ((val_low >= val_high) and not pos) or ((val_low <= val_high) and pos):
                    r_low = self.track.rho_of_r_neg(val_low)
                    r_high = self.track.rho_of_r_neg(val_high)
                else:
                    r_low = self.track.rho_of_r_pos(val_low)
                    r_high = self.track.rho_of_r_pos(val_high)
                intervals.append(sorted((r_low, r_high)))
            else:
                intervals.append(None)
        return intervals

    def get_z_matrix(self, Dt=0, Dx=0, Dy=0, Dz=0):
        ''' calculate the z-matrix for a given DOM hit:
            i.e. dom 3d position + hit time
            
            Dt : DOM hit time (ns)
            Dx, Dy, Dz : DOM position (m)
            '''

        # set the origin of the spherical coords. to the DOM hit pos.
        self.track.set_origin(Dt, Dx, Dy, Dz)

        # closest point (r inflection point)
        tb = self.track.tb
        if tb >= self.track.t0 and tb < self.track.t0 + self.track.dt:
            xb, yb, zb = self.track.point(tb)
            r_inflection_point = self.cr(xb, yb, zb)

        # theta inflection point
        ts = self.track.ts
        if ts >= self.track.t0 and ts < self.track.t0 + self.track.dt:
            xs, ys, zs = self.track.point(ts)
            theta_inflection_point = self.ctheta(xs, ys, zs)

        # the big matrix z
        z = np.zeros((len(self.t_bin_edges) - 1, len(self.r_bin_edges) - 1, len(self.theta_bin_edges) - 1, len(self.phi_bin_edges) - 1))

        start_t = time.time()
        inner_loop = 0
        # iterate over time bins
        for k in range(len(self.t_bin_edges) - 1):
            time_bin = (self.t_bin_edges[k], self.t_bin_edges[k+1])
            # maximum extent:
            t_extent = self.track.extent(*time_bin)

            # only continue if there is anything in that time bins
            if t_extent is not None:
                #print 'sth in time bin ',k
                # caluculate the maximal values for each coordinate (r, theta and phi)
                # over which the track spans in the given time bins
                extent = [self.track.point(t_extent[0]), self.track.point(t_extent[1])]
                rho_extent = (self.track.rho_of_t(t_extent[0]), self.track.rho_of_t(t_extent[1]))

                # r
                track_r_extent = (self.cr(*extent[0]), self.cr(*extent[1]))
                if tb <= t_extent[0]:
                    track_r_extent_neg = sorted(track_r_extent)
                    track_r_extent_pos = [-1, -1]
                elif tb > t_extent[1]:
                    track_r_extent_pos = sorted(track_r_extent)
                    track_r_extent_neg = [-1, -1]
                else:
                    track_r_extent_pos = sorted([track_r_extent[0], r_inflection_point])
                    track_r_extent_neg = sorted([r_inflection_point, track_r_extent[1]])

                #print track_r_extent_pos
                #print track_r_extent_neg

                # theta
                track_theta_extent = (self.ctheta(*extent[0]), self.ctheta(*extent[1]))
                if ts <= t_extent[0]:
                    track_theta_extent_neg = track_theta_extent
                    track_theta_extent_pos = None
                elif ts >= t_extent[1]:
                    track_theta_extent_pos = track_theta_extent
                    track_theta_extent_neg = None
                else:
                    track_theta_extent_neg = (track_theta_extent[0], theta_inflection_point)
                    track_theta_extent_pos = (theta_inflection_point, track_theta_extent[1])

                # phi
                track_phi_extent = sorted([self.cphi(*extent[0]), self.cphi(*extent[1])])
                if np.abs(track_phi_extent[1] - track_phi_extent[0]) > np.pi:
                    track_phi_extent.append(track_phi_extent.pop(0))

                # clalculate intervals with bin edges:
                theta_inter_neg = self.correlate_theta(self.theta_bin_edges, track_theta_extent_neg, rho_extent)
                theta_inter_pos = self.correlate_theta(self.theta_bin_edges, track_theta_extent_pos, rho_extent)
                phi_inter = self.correlate_phi(self.phi_bin_edges, track_phi_extent, rho_extent)
                r_inter_neg = self.correlate_r(self.r_bin_edges, track_r_extent_neg, False)
                r_inter_pos = self.correlate_r(self.r_bin_edges, track_r_extent_pos, True)
                
                start_t2 = time.time()
                # Fill in the Z matrix the length of the track if there is overlap of the track with a bin
                # ugly nested loops
                for i, phi in enumerate(phi_inter):
                    if phi is None:
                        continue
                    for m, theta in enumerate(theta_inter_neg):
                        if theta is None:
                            continue
                        for j, r in enumerate(r_inter_neg):
                            if r is None:
                                continue
                            # interval of r theta and phi
                            A = max(r[0], theta[0], phi[0])
                            B = min(r[1], theta[1], phi[1])
                            if A <= B:
                                length = B - A
                                z[k][j][m][i] += length * self.photons_per_meter
                        for j, r in enumerate(r_inter_pos):
                            if r is None:
                                continue
                            # interval of r theta and phi
                            A = max(r[0], theta[0], phi[0])
                            B = min(r[1], theta[1], phi[1])
                            if A <= B:
                                length = B - A
                                z[k][j][m][i] += length * self.photons_per_meter
                    for m, theta in enumerate(theta_inter_pos):
                        if theta is None:
                            continue
                        for j, r in enumerate(r_inter_neg):
                            if r is None:
                                continue
                            # interval of r theta and phi
                            A = max(r[0], theta[0], phi[0])
                            B = min(r[1], theta[1], phi[1])
                            if A <= B:
                                length = B - A
                                z[k][j][m][i] += length * self.photons_per_meter
                        for j, r in enumerate(r_inter_pos):
                            if r is None:
                                continue
                            # interval of r theta and phi
                            A = max(r[0], theta[0], phi[0])
                            B = min(r[1], theta[1], phi[1])
                            if A <= B:
                                length = B - A
                                z[k][j][m][i] += length * self.photons_per_meter
                end_t2 = time.time()
                inner_loop += end_t2 - start_t2


        end_t = time.time()

        print 'inner loop took %.1f ms'%(inner_loop*1000)
        print 'outer loop took %.1f ms'%((end_t - start_t)*1000)

        # add cascade as point:
        # get bin at self.track.t0, ...
        t0 = self.track.t0
        x0 = self.track.x0
        y0 = self.track.y0
        z0 = self.track.z0
        r0 = self.cr(x0, y0, z0)
        theta0 = self.ctheta(x0, y0, z0)
        phi0 = self.cphi(x0, y0, z0)
        # find bins
        t_bin = self.get_bin(t0, self.t_bin_edges)
        r_bin = self.get_bin(r0, self.r_bin_edges)
        theta_bin = self.get_bin(theta0, self.theta_bin_edges)
        phi_bin = self.get_bin(phi0, self.phi_bin_edges)
        if not None in (t_bin, r_bin, theta_bin, phi_bin):
            z[t_bin, r_bin, theta_bin, phi_bin] += self.cscd_photons

        return z

    @staticmethod
    def get_bin(val, bin_edges):
        ''' find bin for value val in bin_edges
            return None if val is outside binning
        '''
        for k in range(len(bin_edges) - 1):
            if bin_edges[k] <= val and val < bin_edges[k+1]:
                return k
        return None


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


    def PowerAxis(minval, maxval, n_bins, power):
        '''JVSs Power axis reverse engeneered'''
        l = np.linspace(np.power(minval, 1./power), np.power(maxval, 1./power), n_bins+1)
        bin_edges = np.power(l, power)
        return bin_edges

    # same as CLsim
    t_bin_edges = np.linspace(0, 500, 51)
    r_bin_edges = PowerAxis(0, 100, 20, 2)
    theta_bin_edges = np.arccos(np.linspace(-1, 1, 51))[::-1]
    phi_bin_edges = np.linspace(0, 2*np.pi, 37)

    my_hypo = hypo(50., 0., 0., 0., theta=1.0, phi=-0.45, trck_energy=10., cscd_energy=10.)
    my_hypo.set_binning(t_bin_edges, r_bin_edges, theta_bin_edges, phi_bin_edges)


    # plot the track as a line
    x_0, y_0, z_0 = my_hypo.track.point(my_hypo.track.t0)
    print 'track vertex', x_0, y_0, z_0
    x_e, y_e, z_e  = my_hypo.track.point(my_hypo.track.t0 + my_hypo.track.dt)
    ax.plot([x_0,x_e],[y_0,y_e],zs=[z_0,z_e])
    ax.plot([-plt_lim,-plt_lim],[y_0,y_e],zs=[z_0,z_e],alpha=0.3,c='k')
    ax.plot([x_0,x_e],[plt_lim,plt_lim],zs=[z_0,z_e],alpha=0.3,c='k')
    ax.plot([x_0,x_e],[y_0,y_e],zs=[-plt_lim,-plt_lim],alpha=0.3,c='k')
    
    t0 = time.time()
    z = my_hypo.get_z_matrix(10., 0., 0., 0.)
    print 'took %.2f ms to calculate matrix with %i bins'%((time.time() - t0)*1000, z.size)
    print 'total number of photons %i'%z.sum()

    #cmap = 'gnuplot2_r'
    cmap = mpl.cm.get_cmap('gnuplot_r')
    cmap.set_under('w')
    cmap.set_bad('w')

    tt, yy = np.meshgrid(t_bin_edges, r_bin_edges)
    zz = z.sum(axis=(2,3))
    mg = ax2.pcolormesh(tt, yy, zz.T, vmin=1e-7, cmap=cmap)
    ax2.set_xlabel('t')
    ax2.set_ylabel('r')

    tt, yy = np.meshgrid(t_bin_edges, theta_bin_edges)
    zz = z.sum(axis=(1,3))
    mg = ax3.pcolormesh(tt, yy, zz.T, vmin=1e-7, cmap=cmap)
    ax3.set_xlabel('t')
    ax3.set_ylabel(r'$\theta$')
    ax3.set_ylim((0,np.pi))

    tt, yy = np.meshgrid(t_bin_edges, phi_bin_edges)
    zz = z.sum(axis=(1,2))
    mg = ax4.pcolormesh(tt, yy, zz.T, vmin=1e-7, cmap=cmap)
    ax4.set_xlabel('t')
    ax4.set_ylabel(r'$\phi$')
    ax4.set_ylim((0,2*np.pi))

    plt.show()
    plt.savefig('hypo.png',dpi=300)
