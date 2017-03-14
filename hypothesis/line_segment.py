import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, bitrate=20000)


# plot setup
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(121,adjustable='box', aspect='equal')
ax2 = fig.add_subplot(122, projection='polar')
ax.set_xlim((-10,10))
ax.set_ylim((-10,10))
ax.grid(True)


# --- track (= hypothesis) ---
# the track is w.r.t the IC coordinates and is specified given a vertex 
# position, plus one (two) angles and a length

# set_origin is used to displace the track so it will be w.r.t. to DOM coordinate system

class track(object):
    def __init__(self, t_v, x_v, y_v, phi, length):
        # speed of light
        self.c = 1.
        # vertex position, angle and length
        self.t_v = t_v
        self.x_v = x_v
        self.y_v = y_v
        self.phi = phi
        self.sin = np.sin(phi)
        self.cos = np.cos(phi)
        self.tan = np.tan(phi)
        self.length = length
        self.dt = self.length / self.c
        self.set_origin(0,0,0)

    def set_origin(self, t, x, y):
        self.t_o = t
        self.x_o = x
        self.y_o = y

    @property
    def tb(self):
        # closest time to origin
        return self.t0 - (self.x0*self.cos + self.y0*self.sin) / self.c

    @property
    def t0(self):
        return self.t_v - self.t_o
    @property
    def x0(self):
        return self.x_v - self.x_o
    @property
    def y0(self):
        return self.y_v - self.y_o

    def point(self, t):
        # make sure time is valid
        assert (self.t0 <= t) and (t <= self.t0 + self.dt) 
        dt = (t - self.t0)
        dr = self.c * dt
        x = self.x0 + dr * self.cos
        y = self.y0 + dr * self.sin
        return (x,y)

    def extent(self, t_low, t_high):
        # get full extend of track in time interval
        if (self.t0 + self.dt >= t_low) and (t_high >= self.t0):
            # then we have overlap
            t_0 = max(t_low, self.t0)
            t_1 = min(t_high, self.t0+self.dt)
            return (t_0, t_1)
        else:
            return None

    def r_of_x(self, x):
        return (x - self.x0)/self.cos

    def r_of_y(self, y):
        return (y - self.y0)/self.sin

    def r_of_phi(self, phi):
        sin = np.sin(phi)
        cos = np.cos(phi)
        return (sin*self.x0 - cos*self.y0)/(cos*self.sin - sin*self.cos)

    def r_of_r_pos(self, r):
        S = r**2 - (self.x0*self.sin)**2 - (self.y0*self.cos)**2 + 2*self.x0*self.y0*self.sin*self.cos
        if S < 0:
            A = 0
        else:
            A = np.sqrt(S)
        return A - self.x0*self.cos - self.y0*self.sin
    def r_of_r_neg(self, r):
        S = r**2 - (self.x0*self.sin)**2 - (self.y0*self.cos)**2 + 2*self.x0*self.y0*self.sin*self.cos
        if S < 0:
            A = 0
        else:
            A = np.sqrt(S)
        return - A - self.x0*self.cos - self.y0*self.sin
 
#my_track = track(0, -4.0, -2.1, 0.05, 5.5)
#my_track = track(0, -8.0, -5.1, 0.3, 15.)
my_track = track(5, -6.0, -5.1, 0.45, 15.)
my_track.set_origin(5,0,-1)
#my_track = track(0, 3.5, -2.1, np.pi/2., 5.5)

# plot
x_0, y_0 = my_track.point(my_track.t0)
x_e, y_e = my_track.point(my_track.t0 + my_track.dt)
ax.arrow(x_0, y_0, x_e - x_0, y_e - y_0, head_width=0.05, head_length=0.1, fc='k', ec='k')


# plot the DOM
ax.plot(0,0,'+',markersize=10,c='b')
ax.plot(0,0,'o',markersize=10,mfc='none',c='b')

# binning
t_bin_edges = np.linspace(0,20,41)
r_bin_edges = np.linspace(0,10,21)
phi_bin_edges = np.linspace(0,2*np.pi,41)

for r in r_bin_edges:
    circle = plt.Circle((0,0), r, color='g', alpha=0.2, fill=False)
    ax.add_artist(circle)
for phi in phi_bin_edges:
    dx = 100 * np.cos(phi)
    dy = 100 * np.sin(phi)
    ax.plot([0,0 + dx],[0,dy], ls='-', color='g', alpha=0.2)

# coord. transforms
def cr(x,y):
    return np.sqrt(x**2 + y**2)
def cphi(x,y):
    v = np.arctan2(y,x)
    if v < 0: v += 2*np.pi
    return v


# ---- the actual calculation:


# closest point
tb = my_track.tb

if tb > my_track.t0 and tb < my_track.t0 + my_track.dt:
    xb, yb = my_track.point(my_track.tb)
    rb = cr(xb,yb)

ims = []
for k in range(len(t_bin_edges) - 1):
    # time bin
    time_bin = (t_bin_edges[k], t_bin_edges[k+1])
    phiphi, rr = np.meshgrid(phi_bin_edges, r_bin_edges)
    z = np.zeros(phiphi.size).reshape(phiphi.shape)

    # maximum extent:
    t_extent = my_track.extent(*time_bin)

    if t_extent is not None:
        r_extent = (my_track.c * t_extent[0], my_track.c * t_extent[1])
        extent = [my_track.point(t_extent[0]), my_track.point(t_extent[1])]
        track_phi_extent = sorted([cphi(*extent[0]), cphi(*extent[1])])
        if np.abs(track_phi_extent[1] - track_phi_extent[0])>np.pi:
            track_phi_extent.append(track_phi_extent.pop(0))
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

        
        # phi intervals
        phi_inter = []
        for i in range(len(phi_bin_edges) - 1):
            # get interval overlaps
            # from these two intervals:
            t = track_phi_extent
            b = (phi_bin_edges[i],phi_bin_edges[i+1])
            if t[0] == t[1]:
                # along coordinate axis
                phi_inter.append(r_extent)
            elif t[0] <= t[1] and (b[0] <= t[1]) and (t[0] < b[1]):
                phi_h = min(b[1], t[1])
                phi_l = max(b[0], t[0])
                r_l = my_track.r_of_phi(phi_l)
                r_h = my_track.r_of_phi(phi_h)
                phi_inter.append(sorted((r_l,r_h)))
            # crossing the 0/2pi point 
            elif t[1] < t[0]:
                if b[1] >= 0 and t[1] >= b[0]:
                    phi_h = min(b[1], t[1])
                    phi_l = max(b[0],0)
                elif 2*np.pi > b[0] and b[1] >= t[0]:
                    phi_h = min(b[1], 2*np.pi)
                    phi_l = max(b[0],t[0])
                elif b[0] <= t[1] and t[0] <= t[1]:
                    phi_h = min(b[1], t[1])
                    phi_l = max(b[0], t[0])
                else:
                    phi_inter.append(None)
                    continue
                r_l = my_track.r_of_phi(phi_l)
                r_h = my_track.r_of_phi(phi_h)
                phi_inter.append(sorted((r_l,r_h)))
            else:
                phi_inter.append(None)

        # also need two r extents!
        r_inter_neg = []
        for i in range(len(r_bin_edges) - 1):
            # get interval overlaps
            # from these two intervals:
            t = track_r_extent_neg
            b = (r_bin_edges[i],r_bin_edges[i+1])
            if (b[0] <= t[1]) and (t[0] < b[1]):
                ro_h = min(b[1], t[1])
                ro_l = max(b[0], t[0])
                if ro_l > ro_h:
                    r_l = my_track.r_of_r_neg(ro_l)
                    r_h = my_track.r_of_r_neg(ro_h)
                else:
                    r_l = my_track.r_of_r_pos(ro_l)
                    r_h = my_track.r_of_r_pos(ro_h)
                r_inter_neg.append(sorted((r_l,r_h)))
            else:
                r_inter_neg.append(None)
        r_inter_pos = []
        for i in range(len(r_bin_edges) - 1):
            # get interval overlaps
            # from these two intervals:
            t = track_r_extent_pos
            b = (r_bin_edges[i],r_bin_edges[i+1])
            if (b[0] <= t[1]) and (t[0] < b[1]):
                ro_h = min(b[1], t[1])
                ro_l = max(b[0], t[0])
                if ro_l > ro_h:
                    r_l = my_track.r_of_r_pos(ro_l)
                    r_h = my_track.r_of_r_pos(ro_h)
                else:
                    r_l = my_track.r_of_r_neg(ro_l)
                    r_h = my_track.r_of_r_neg(ro_h)
                r_inter_pos.append(sorted((r_l,r_h)))
            else:
                r_inter_pos.append(None)

        # get bins with interval overlaps
        for i,phi in enumerate(phi_inter):
            if phi is None: continue
            for j,r in enumerate(r_inter_neg):
                if r is None: continue
                if (r[0] < phi[1]) and (phi[0] < r[1]):
                    # we have oberlap
                    length = min(phi[1], r[1]) - max(phi[0], r[0])
                    z[j][i] += length
            for j,r in enumerate(r_inter_pos):
                if r is None: continue
                if (r[0] < phi[1]) and (phi[0] < r[1]):
                    # we have oberlap
                    length = min(phi[1], r[1]) - max(phi[0], r[0])
                    z[j][i] += length

    im = ax2.pcolormesh(phiphi, rr, z,cmap='Purples')
    ax2.grid(True)
    ims.append([im])

im_ani = animation.ArtistAnimation(fig, ims, interval=200, repeat_delay=3000, blit=True)
im_ani.save('im.mp4', writer=writer)
#plt.show()
#plt.savefig('test.png',dpi=150)
