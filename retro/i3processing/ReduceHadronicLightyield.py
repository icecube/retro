# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

__all__ = '''
    HadLightyield
'''.split()

from icecube import icetray, dataclasses, clsim
from numpy import random


class HadLightyield(icetray.I3Module):
    """
    Parametrizations taken from PPC Revision 97875 Only the mean value is
    taken. The variation was done by PPC the first time it was used.

    """
    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddOutBox('OutBox')
        self.AddParameter('InputPhotonSeries',
                          'Name of the photon series',
                          'UnweightedPhotons')
        self.AddParameter('MCTree',
                          'Name of the MC tree',
                          'I3MCTree')
        self.AddParameter('OutputPhotonSeries',
                          'Name for the photon series to output',
                          'UnweightedPhotons2')
        self.AddParameter('Lightyield',
                          'Lightyield of the hadrons',
                          0.95)

    def Configure(self):
        self.hs_in  = self.GetParameter('InputPhotonSeries')
        self.hs_out = self.GetParameter('OutputPhotonSeries')
        self.counter = 0
        self.lightyield = self.GetParameter('Lightyield')
        self.skip_particles = [dataclasses.I3Particle.MuMinus,
                               dataclasses.I3Particle.MuPlus,
                               dataclasses.I3Particle.TauMinus,
                               dataclasses.I3Particle.TauPlus,
                               dataclasses.I3Particle.EMinus,
                               dataclasses.I3Particle.EPlus,
                               dataclasses.I3Particle.Gamma,
                               dataclasses.I3Particle.Brems,
                               dataclasses.I3Particle.DeltaE,
                               dataclasses.I3Particle.PairProd,
                               dataclasses.I3Particle.Pi0]

    def DAQ(self, frame):
        mctree = frame['I3MCTree']

        # Define the particles whose light will be modified
        mctree_particles = mctree.get_daughters(mctree.most_energetic_neutrino)
        hadrons_ids = []
        for one_particle in mctree_particles:
            if not one_particle.type in self.skip_particles:
                hadrons_ids.append(one_particle.minor_id)

        if len(hadrons_ids) == 0:
            frame[self.hs_out] = frame[self.hs_in]
            # Do nothing, pass the unweighed photons as they were
            return True

        photonseries = frame[self.hs_in]

        # The new I3PhotonSeriesMap needs to be initializaed with a list of tuples
        photonseries_tuples = []

        # Go DOM by DOM
        for one_dom, dom_hits in photonseries:

            # Initialize a new hit series for every DOM
            new_dom_photons = clsim.I3PhotonSeries()

            # Go hit by hit
            for index, one_hit in enumerate(dom_hits):
                # Decide if the photon weight will be modified
                if one_hit.particleMinorID in hadrons_ids:
                    new_photon = clsim.I3Photon(one_hit)
                    new_photon.weight *= self.lightyield
                    #print('Reduced! ', new_photon.weight)
                    new_dom_photons.append(new_photon)
                    self.counter += 1
                else:
                    new_dom_photons.append(one_hit)

            photonseries_tuples.append((one_dom, new_dom_photons))

        # Create a new UnweightedPhotons Map from my tuple
        new_hitseries = clsim.I3PhotonSeriesMap(photonseries_tuples)

        # frame.Delete(self.hs_in)
        # Push it to the frame
        frame[self.hs_out] = new_hitseries
        #print('DONE!')
        self.PushFrame(frame)
        return True

    def Finish(self):
        # Inform how many hits were dropped in this file
        print('CorrectHadronicLightYield worked, applied ', self.counter)
