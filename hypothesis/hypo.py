import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=15, bitrate=20000)


# plot setup
fig = plt.figure(figsize=(10,10))
#ax = fig.add_subplot(221,adjustable='box', aspect='equal')
ax = fig.add_subplot(221,projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
#ax2 = fig.add_subplot(122, projection='polar')
ax.set_xlim((-10,10))
ax.set_ylim((-10,10))
ax.set_zlim((-10,10))
ax.grid(True)


# --- track (= hypothesis) ---
# the track is w.r.t the IC coordinates and is specified given a vertex 
# position, plus one (two) angles and a length

# set_origin is used to displace the track so it will be w.r.t. to DOM coordinate system

class track(object):
    def __init__(self, t_v, x_v, y_v, z_v, phi, theta, length):
        # speed of light
        self.c = 0.1
        # vertex position, angle and length
        self.t_v = t_v
        self.x_v = x_v
        self.y_v = y_v
        self.z_v = z_v
        self.phi = phi
        self.theta = theta
        self.sinphi = np.sin(phi)
        self.cosphi = np.cos(phi)
        self.tanphi = np.tan(phi)
        self.sintheta = np.sin(theta)
        self.costheta = np.cos(theta)
        self.length = length
        self.dt = self.length / self.c
        self.set_origin(0,0,0,0)

    def set_origin(self, t, x, y, z):
        self.t_o = t
        self.x_o = x
        self.y_o = y
        self.z_o = z

    @property
    def t0(self):
        return self.t_v - self.t_o
    @property
    def x0(self):
        return self.x_v - self.x_o
    @property
    def y0(self):
        return self.y_v - self.y_o
    @property
    def z0(self):
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
        return (x,y,z)

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
        rho = ( \
            - self.costheta*(self.x0**2 + self.y0**2) \
            + self.sintheta*self.z0*(self.x0*self.cosphi + self.y0*self.sinphi)) \
        /( \
            + self.sintheta*self.costheta*(self.x0*self.cosphi + self.y0*self.sinphi) \
            - self.z0*(self.sintheta**2) \
            )
        return self.t0 + rho / self.c

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

    def get_A(self,R):
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


# T, X, Y, Z, Phi, Theta, L
#my_track = track(0, -4.0, -2.1, 0.05, 5.5)
#my_track = track(0, -8.0, -5.1, 0.3, 15.)
#my_track = track(0, -6.0, 4.1, 2.0, -0.45, 1., 20.)
my_track = track(0, 0, 0, 2.0, -0.45, np.pi, 20.)
#my_track = track(0, 6.0, 4.1, -2.0, 2.45, 1., 20.)
#my_track.set_origin(5,0,-1)
#my_track = track(0, 3.5, -2.1, np.pi/2., 5.5)

# plot
x_0, y_0, z_0 = my_track.point(my_track.t0)
x_e, y_e, z_e  = my_track.point(my_track.t0 + my_track.dt)

# projections?
#ax.arrow(x_0, y_0, x_e - x_0, y_e - y_0, head_width=0.05, head_length=0.1, fc='k', ec='k')
#ax.quiver(x_0, y_0, z_0, x_e - x_0, y_e - y_0, z_e -z_0)
m = -10
ax.plot([x_0,x_e],[y_0,y_e],zs=[z_0,z_e])
ax.plot([m,m],[y_0,y_e],zs=[z_0,z_e],alpha=0.3,c='k')
ax.plot([x_0,x_e],[-m,-m],zs=[z_0,z_e],alpha=0.3,c='k')
ax.plot([x_0,x_e],[y_0,y_e],zs=[m,m],alpha=0.3,c='k')

def PowerAxis(minval, maxval, n_bins, power):
    '''JVSs Power axis reverse engeneered'''
    l = np.linspace(np.power(minval,1./power),np.power(maxval,1./power), n_bins+1)
    bin_edges = np.power(l,power)
    return bin_edges

# same as CLsim
#t_bin_edges = PowerAxis(0, 7e3, 105, 2)
t_bin_edges = np.linspace(0, 200, 101)
r_bin_edges = PowerAxis(0, 15, 200, 2)
#r_bin_edges = PowerAxis(0, 580, 200, 2)
phi_bin_edges = np.linspace(0, 2*np.pi, 37)
theta_bin_edges = np.arccos(np.linspace(-1 , 1, 101))[::-1]

# coord. transforms
def cr(x,y,z):
    '''r in spehrical coord., given x,y,z'''
    return np.sqrt(x**2 + y**2 + z**2)
def cphi(x,y,z):
    '''phi in spehrical coord., given x,y,z'''
    v = np.arctan2(y,x)
    if v < 0: v += 2*np.pi
    return v
def ctheta(x,y,z):
    '''theta in spehrical coord., given x,y,z'''
    return np.arccos(z/np.sqrt(x**2 + y**2 + z**2))

# bin correlators:
def correlate_theta(bin_edges, t, rho_extent, my_track):
    '''get the track parameter rho intervals, which lie inside bins'''
    intervals = []
    for i in range(len(bin_edges) - 1):
        # get interval overlaps
        # from these two intervals:
        if t is None:
            intervals.append(None)
            continue
        b = (bin_edges[i],bin_edges[i+1])
        if t[0] == t[1] and (b[0] <= t[0]) and (t[0] <= b[1]):
            # along coordinate axis
            intervals.append(rho_extent)
        elif (b[0] <= t[1]) and (t[0] < b[1]) and (t[0] < t[1]):
            val_high = min(b[1], t[1])
            val_low = max(b[0], t[0])
            if b[0] < np.pi/2.:
                theta_low = my_track.rho_of_theta_pos(val_low)
                theta_high = my_track.rho_of_theta_pos(val_high)
            else:
                theta_low = my_track.rho_of_theta_neg(val_low)
                theta_high = my_track.rho_of_theta_neg(val_high)
            intervals.append(sorted((theta_low,theta_high)))
        elif (b[0] <= t[0]) and (t[1] < b[1]) and (t[1] < t[0]):
            val_high = min(b[1], t[0])
            val_low = max(b[0], t[1])
            if b[0] < np.pi/2.:
                theta_low = my_track.rho_of_theta_neg(val_low)
                theta_high = my_track.rho_of_theta_neg(val_high)
            else:
                theta_low = my_track.rho_of_theta_pos(val_low)
                theta_high = my_track.rho_of_theta_pos(val_high)
            intervals.append(sorted((theta_low,theta_high)))
        else:
            intervals.append(None)
    return intervals

def correlate_phi(bin_edges, t, rho_extent, my_track):
    '''get the track parameter rho intervals, which lie inside bins'''
    # phi intervals
    intervals = []
    for i in range(len(bin_edges) - 1):
        # get interval overlaps
        # from these two intervals:
        b = (bin_edges[i],bin_edges[i+1])
        if t[0] == t[1] and (b[0] <= t[0]) and (t[0] < b[1]):
            # along coordinate axis
            intervals.append(rho_extent)
        elif t[0] <= t[1] and (b[0] <= t[1]) and (t[0] < b[1]):
            phi_high = min(b[1], t[1])
            phi_low = max(b[0], t[0])
            r_low = my_track.rho_of_phi(phi_low)
            r_high = my_track.rho_of_phi(phi_high)
            intervals.append(sorted((r_low,r_high)))
        # crossing the 0/2pi point 
        elif t[1] < t[0]:
            if b[1] >= 0 and t[1] >= b[0]:
                phi_high = min(b[1], t[1])
                phi_low = max(b[0],0)
            elif 2*np.pi > b[0] and b[1] >= t[0]:
                phi_high = min(b[1], 2*np.pi)
                phi_low = max(b[0],t[0])
            elif b[0] <= t[1] and t[0] <= t[1]:
                phi_high = min(b[1], t[1])
                phi_low = max(b[0], t[0])
            else:
                intervals.append(None)
                continue
            r_low = my_track.rho_of_phi(phi_low)
            r_high = my_track.rho_of_phi(phi_high)
            intervals.append(sorted((r_low,r_high)))
        else:
            intervals.append(None)
    return intervals

def correlate_r(bin_edges, t, pos, my_track):
    '''get the track parameter rho intervals, which lie inside bins'''
    intervals = []
    for i in range(len(bin_edges) - 1):
        # get interval overlaps
        # from these two intervals:
        b = (bin_edges[i],bin_edges[i+1])
        if (b[0] <= t[1]) and (t[0] < b[1]):
            val_high = min(b[1], t[1])
            val_low = max(b[0], t[0])
            if ((val_low > val_high) and not pos) or ((val_low <= val_high) and pos):
                r_low = my_track.rho_of_r_neg(val_low)
                r_high = my_track.rho_of_r_neg(val_high)
            else:
                r_low = my_track.rho_of_r_pos(val_low)
                r_high = my_track.rho_of_r_pos(val_high)
            intervals.append(sorted((r_low,r_high)))
        else:
            intervals.append(None)
    return intervals

# closest point
tb = my_track.tb
if tb > my_track.t0 and tb < my_track.t0 + my_track.dt:
    xb, yb, zb = my_track.point(tb)
    rb = cr(xb,yb,zb)

# closest point to z-axis:
ts = my_track.ts
if ts > my_track.t0 and ts < my_track.t0 + my_track.dt:
    xs, ys, zs = my_track.point(ts)
    thetas = ctheta(xs,ys,zs)

#ims = []

# the big matrix z
z = np.zeros((len(t_bin_edges) - 1, len(r_bin_edges) - 1, len(theta_bin_edges) - 1, len(phi_bin_edges) - 1))

# terate over time bins
for k in range(len(t_bin_edges) - 1):
    time_bin = (t_bin_edges[k], t_bin_edges[k+1])
    # maximum extent:
    t_extent = my_track.extent(*time_bin)

    # only continue if there is anything in that time bins
    if t_extent is not None:
        # caluculate the maximal values for each coordinate (r, theta and phi)
        # over which the track spans in the given time bins
        extent = [my_track.point(t_extent[0]), my_track.point(t_extent[1])]
        rho_extent = (my_track.c * t_extent[0], my_track.c * t_extent[1])

        # r
        track_r_extent = (cr(*extent[0]), cr(*extent[1]))
        if tb <= t_extent[0]:
            track_r_extent_neg = sorted(track_r_extent)
            track_r_extent_pos = [0,0]
        elif tb >= t_extent[1]:
            track_r_extent_pos = sorted(track_r_extent)
            track_r_extent_neg = [0,0]
        else:
            track_r_extent_pos = sorted([track_r_extent[0], rb])
            track_r_extent_neg = sorted([rb, track_r_extent[1]])

        # theta
        track_theta_extent = (ctheta(*extent[0]), ctheta(*extent[1]))
        if ts <= t_extent[0]:
            track_theta_extent_neg = track_theta_extent
            track_theta_extent_pos = None
        elif ts >= t_extent[1]:
            track_theta_extent_pos = track_theta_extent
            track_theta_extent_neg = None
        else:
            track_theta_extent_neg = [track_theta_extent[0], thetas]
            track_theta_extent_pos = [thetas, track_theta_extent[1]]

        # phi
        track_phi_extent = sorted([cphi(*extent[0]), cphi(*extent[1])])
        if np.abs(track_phi_extent[1] - track_phi_extent[0])>np.pi:
            track_phi_extent.append(track_phi_extent.pop(0))

        # clalculate intervals with bin edges:
        theta_inter_neg = correlate_theta(theta_bin_edges, track_theta_extent_neg, rho_extent, my_track)
        theta_inter_pos = correlate_theta(theta_bin_edges, track_theta_extent_pos, rho_extent, my_track)
        
        phi_inter = correlate_phi(phi_bin_edges, track_phi_extent, rho_extent, my_track)

        r_inter_neg = correlate_r(r_bin_edges, track_r_extent_neg, False, my_track)
        r_inter_pos = correlate_r(r_bin_edges, track_r_extent_pos, True, my_track)

        # Fill in the Z matrix the length of the track if there is overlap of the track with a bin
        # ugly nested loops
        for i,phi in enumerate(phi_inter):
            if phi is None: continue
            for m,theta in enumerate(theta_inter_neg):
                if theta is None: continue
                for j,r in enumerate(r_inter_neg):
                    if r is None: continue
                    # interval of r theta and phi
                    A = max(r[0],theta[0],phi[0])
                    B = min(r[1],theta[1],phi[1])
                    if A <= B:
                        length = B-A
                        z[k][j][m][i] += length
                for j,r in enumerate(r_inter_pos):
                    if r is None: continue
                    # interval of r theta and phi
                    A = max(r[0],theta[0],phi[0])
                    B = min(r[1],theta[1],phi[1])
                    if A <= B:
                        length = B-A
                        z[k][j][m][i] += length
            for m,theta in enumerate(theta_inter_pos):
                if theta is None: continue
                for j,r in enumerate(r_inter_neg):
                    if r is None: continue
                    # interval of r theta and phi
                    A = max(r[0],theta[0],phi[0])
                    B = min(r[1],theta[1],phi[1])
                    if A <= B:
                        length = B-A
                        z[k][j][m][i] += length
                for j,r in enumerate(r_inter_pos):
                    if r is None: continue
                    # interval of r theta and phi
                    A = max(r[0],theta[0],phi[0])
                    B = min(r[1],theta[1],phi[1])
                    if A <= B:
                        length = B-A
                        z[k][j][m][i] += length

    #p = phiphi[:,0,:]
    #r = rr[:,0,:]
    #nz = z.sum(axis=1)

    #im = ax2.pcolormesh(p, r, nz,cmap='Purples')
    #ax2.grid(True)
    #ims.append([im])

#im_ani = animation.ArtistAnimation(fig, ims, interval=200, repeat_delay=3000, blit=True)
#im_ani.save('im.mp4', writer=writer)
vmax=0.2
cmap = 'bone_r'

tt, yy = np.meshgrid(t_bin_edges, r_bin_edges)
zz = z.sum(axis=(2,3))
mg = ax2.pcolormesh(tt, yy, zz.T, vmin=0., vmax=vmax, cmap=cmap)
ax2.set_xlabel('t')
ax2.set_ylabel('r')

tt, yy = np.meshgrid(t_bin_edges, theta_bin_edges)
zz = z.sum(axis=(1,3))
mg = ax3.pcolormesh(tt, yy, zz.T, vmin=0., vmax=vmax, cmap=cmap)
ax3.set_xlabel('t')
ax3.set_ylabel(r'$\theta$')
ax3.set_ylim((0,np.pi))

tt, yy = np.meshgrid(t_bin_edges, phi_bin_edges)
zz = z.sum(axis=(1,2))
mg = ax4.pcolormesh(tt, yy, zz.T, vmin=0., vmax=vmax, cmap=cmap)
ax4.set_xlabel('t')
ax4.set_ylabel(r'$\phi$')
ax4.set_ylim((0,2*np.pi))

plt.show()
plt.savefig('3d.png',dpi=300)
