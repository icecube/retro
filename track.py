import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# plot setup
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.set_xlim((-10,10))
ax.set_ylim((-10,10))
ax.grid(True)


# we live in the IC coordinate system with (t,x,y,z)

# --- track (= hypothesis) ---
# the track is w.r.t the IC coordinates and is specified given a vertex 
# position, plus one (two) angles and a length

class track(object):
    def __init__(self, t_v, x_v, y_v, phi, length):
        # speed of light
        self.c = 1.
        # vertex position, angle and length
        self.t_v = t_v
        self.x_v = x_v
        self.y_v = y_v
        self.phi = phi
        self.length = length
        # end time
        self.t_end = t_v + (self.length / self.c)

    def extent(self, t_low, t_high):
        # get full extend of track in time interval
        if (self.t_end >= t_low) and (t_high >= self.t_v):
            # then we have overlap
            t_0 = max(t_low, self.t_v)
            t_1 = min(t_high, self.t_end)
            return [self.point(t_0), self.point(t_1)]
        else:
            return None

    def point(self, t):
        # make sure time is valid
        assert (self.t_v <= t) and (t <= self.t_end) 
        dt = (t - self.t_v)
        dr = self.c * dt
        # pos
        x = self.x_v + dr * np.cos(self.phi)
        y = self.y_v + dr * np.sin(self.phi)
        return (x,y)

    def r_of_x(self, x):
        return (x - self.x_v)/np.cos(self.phi)

    def r_of_y(self, y):
        return (y - self.y_v)/np.sin(self.phi)

 
my_track = track(1.2, 3.0, -1.7, np.pi/3., 4.1)

# this binning is in coordinates w.r.t the DOM!
# which is sitting at this IC position (DOM coordinates origin)
# and looking at a hit at this time w.r.t the global event time
DOM_origin = (0., 3.3, -1.05, 0.)
ax.plot(DOM_origin[1],DOM_origin[2],'+',markersize=10,c='b')
ax.plot(DOM_origin[1],DOM_origin[2],'o',markersize=10,mfc='none',c='b')


def dom_to_ic_t(x):
    return x + DOM_origin[0]
def dom_to_ic_x(x):
    return x + DOM_origin[1]
def dom_to_ic_y(x):
    return x + DOM_origin[2]
def dom_to_ic_z(x):
    return x + DOM_origin[3]

def ic_to_dom_t(x):
    return x - DOM_origin[0]
def ic_to_dom_x(x):
    return x - DOM_origin[1]
def ic_to_dom_y(x):
    return x - DOM_origin[2]
def ic_to_dom_z(x):
    return x - DOM_origin[3]

# binning
DOM_x_bin_edges = np.linspace(-5,5,11)
DOM_y_bin_edges = np.linspace(-6,6,11)


# plot DOM grid
for DOM_x in DOM_x_bin_edges:
    x = dom_to_ic_x(DOM_x)
    ax.axvline(x,color='b', linestyle='-',alpha=0.2)
for DOM_y in DOM_y_bin_edges:
    y = dom_to_ic_y(DOM_y)
    ax.axhline(y,color='b', linestyle='-',alpha=0.2)

# time bin
time_bin = (0,10)

# track interval in both dimensions
extent = my_track.extent(*time_bin)
track_x_inter = sorted((extent[0][0], extent[1][0]))
track_y_inter = sorted((extent[0][1], extent[1][1]))
# go to DOM coords
DOM_track_x_inter = (ic_to_dom_x(track_x_inter[0]), ic_to_dom_x(track_x_inter[1]))
DOM_track_y_inter = (ic_to_dom_y(track_y_inter[0]), ic_to_dom_y(track_y_inter[1]))

# for every dimension, get track interval in every bin
x_inter = []
for i in range(len(DOM_x_bin_edges) - 1):
    # get interval overlaps
    # from these two intervals:
    t = DOM_track_x_inter
    b = (DOM_x_bin_edges[i],DOM_x_bin_edges[i+1])
    if (b[0] <= t[1]) and (t[0] <= b[1]):
        x_h = min(b[1], t[1])
        x_l = max(b[0], t[0])
        # case of parallel track with coordinate
        if x_l == x_h:
            x_inter.append(track_x_inter)
        else:
            r_l = my_track.r_of_x(dom_to_ic_x(x_l))
            r_h = my_track.r_of_x(dom_to_ic_x(x_h))
            x_inter.append(sorted((r_l,r_h)))
    else:
        x_inter.append(None)
print x_inter


y_inter = []
for i in range(len(DOM_y_bin_edges) - 1):
    # get interval overlaps
    # from these two intervals:
    t = DOM_track_y_inter
    b = (DOM_y_bin_edges[i],DOM_y_bin_edges[i+1])
    if (b[0] <= t[1]) and (t[0] <= b[1]):
        y_h = min(b[1], t[1])
        y_l = max(b[0], t[0])
        if y_l == y_h:
            y_inter.append(track_y_inter)
        else:
            r_l = my_track.r_of_y(dom_to_ic_y(y_l))
            r_h = my_track.r_of_y(dom_to_ic_y(y_h))
            y_inter.append(sorted((r_l,r_h)))
    else:
        y_inter.append(None)
print y_inter


# get bins with interval overlaps
for i,x in enumerate(x_inter):
    if x is None: continue
    for j,y in enumerate(y_inter):
        if y is None: continue
        if (y[0] < x[1]) and (x[0] < y[1]):
            # we have oberlap
            length = min(x[1], y[1]) - max(x[0], y[0])
            print 'length r = %.2f'%length
            print 'at bin %i, %i'%(i,j)

# polt
x_0, y_0 = my_track.point(my_track.t_v)
x_e, y_e = my_track.point(my_track.t_end)
ax.arrow(x_0, y_0, x_e - x_0, y_e - y_0, head_width=0.05, head_length=0.1, fc='k', ec='k')
plt.show()
plt.savefig('test.png')
