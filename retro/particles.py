"""
"""


from __future__ import absolute_import, division

import numpy as np


__all__ = ['SPEED_OF_LIGHT_M_PER_NS', 'Particle', 'ParticleArray']


SPEED_OF_LIGHT_M_PER_NS = 0.299792458
"""Speed of light in units of m/ns"""


class Particle(object):
    """
    class to contain a particle and useful properties

    forward : bool
        if the particle should be plotted forward or backards in time
    """
    def __init__(self, evt, t, x, y, z, zenith, azimuth, energy=None, length=None,
                 pdg=None, interaction=None, forward=False, color='r',
                 linestyle='--', label=''):
        self.evt = evt
        self.t = t
        self.x = x
        self.y = y
        self.z = z
        self.zenith = zenith
        self.azimuth = azimuth
        self.energy = energy
        self.length = length
        self.pdg = pdg
        self.interaction = interaction
        self.forward = forward
        self.color = color
        self.linestyle = linestyle
        self.label = label

    @property
    def theta(self):
        return np.pi - self.zenith

    @property
    def phi(self):
        return (self.azimuth - np.pi)%(2*np.pi)

    @property
    def vertex(self):
        return np.array([self.t, self.x, self.y, self.z])

    @property
    def dt(self):
        # distance traveled, if no length is given set to 1000 m
        return 100. if self.length is None else self.length

    # deltas
    @property
    def dx(self):
        return np.sin(self.theta)*np.cos(self.phi)

    @property
    def dy(self):
        return np.sin(self.theta)*np.sin(self.phi)

    @property
    def dz(self):
        return np.cos(self.theta)

    @property
    def d(self):
        return np.array([self.dt, self.dx, self.dy, self.dz])

    # lines to plot
    @property
    def lt(self):
        if self.forward:
            return [self.t, self.t + self.dt / SPEED_OF_LIGHT_M_PER_NS]
        return [self.t - self.dt / SPEED_OF_LIGHT_M_PER_NS, self.t]

    @property
    def lx(self):
        if self.forward:
            return [self.x, self.x + self.dt*self.dx]
        return [self.x - self.dt * self.dx, self.x]
    @property
    def ly(self):
        if self.forward:
            return [self.y, self.y + self.dt*self.dy]
        return [self.y - self.dt * self.dy, self.y]
    @property
    def lz(self):
        if self.forward:
            return [self.z, self.z + self.dt*self.dz]
        return [self.z - self.dt * self.dz, self.z]

    @property
    def line(self):
        return [self.lt, self.lx, self.ly, self.lz]


class ParticleArray(object):
    """
    Container class for particles from arrays
    get_item will just return a particle object at that position
    """
    def __init__(self, evt, t, x, y, z, zenith, azimuth, energy=None, length=None,
                 pdg=None, interaction=None, forward=False, color='r',
                 linestyle='--', label=''):
        self.evt = evt
        self.t = t
        self.x = x
        self.y = y
        self.z = z
        self.zenith = zenith
        self.azimuth = azimuth
        self.energy = energy
        self.length = length
        self.pdg = pdg
        self.interaction = interaction
        self.forward = forward
        self.color = color
        self.linestyle = linestyle
        self.label = label

    def __getitem__(self, idx):
        energy = None if self.energy is None else self.energy[idx]
        length = None if self.length is None else self.length[idx]
        pdg = None if self.pdg is None else self.pdg[idx]
        if self.interaction is None:
            interaction = None
        else:
            interaction = self.interaction[idx]
        return Particle(self.evt[idx],
                        self.t[idx],
                        self.x[idx],
                        self.y[idx],
                        self.z[idx],
                        self.zenith[idx],
                        self.azimuth[idx],
                        energy,
                        length,
                        pdg,
                        interaction,
                        self.forward,
                        self.color,
                        self.linestyle,
                        self.label)

    def __len__(self):
        return len(self.evt)
