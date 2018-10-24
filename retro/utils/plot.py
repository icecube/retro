# -*- coding: utf-8 -*-

"""
Plotting tools and utilities.
"""

from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt


COLOR_CYCLE_ORTHOG = (
    '#000000', #  0  Black
    '#803E75', #  2  Strong Purple
    '#FF6800', #  3  Vivid Orange
    '#8A9DD7', #  4  Very Light Blue
    '#FFB300', #  1  Vivid Yellow
    '#C10020', #  5  Vivid Red
    '#CEA262', #  6  Grayish Yellow
    '#817066', #  7  Medium Gray

    #The following will not be good for people with defective color vision
    '#007D34', #  8  Vivid Green
    '#F6768E', #  9  Strong Purplish Pink
    '#00538A', # 10  Strong Blue
    '#93AA00', # 11  Vivid Yellowish Green
    '#593315', # 12  Deep Yellowish Brown
    '#F14AD3', # 13  PINK/Magenta!  (used to be: #F13A13, Vivid Reddish Orange
    '#53377A', # 14  Strong Violet
    '#FF8E00', # 15  Vivid Orange Yellow
    '#54BF00', # 16  Vivid Greenish Yellow
    '#0000A5', # 17  BLUE!
    '#7F180D', # 18  Strong Reddish Brown

    #'#F13A13', # 13  Vivid Reddish Orange
    #'#B32851', # 16  Strong Purplish Red
    #'#FF7A5C', # 19  Strong Yellowish Pink
)
"""Use via: ``mpl.rc('axes', color_cycle=colorCycleOrthog)``
Modified from
http://stackoverflow.com/questions/470690/how-to-automatically-generate-n-distinct-colors"""


def remove_border(axes=None, top=False, right=False, left=True, bottom=True):
    """
    from ChrisBeaumont, https://github.com/cs109/content/blob/master/README.md
    """
    ax = axes or plt.gca()
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)

    # Turn off all ticks
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')

    # Re-enable visibles
    if top:
        ax.xaxis.tick_top()
    if bottom:
        ax.xaxis.tick_bottom()
    if left:
        ax.yaxis.tick_left()
    if right:
        ax.yaxis.tick_right()

    ax.spines['bottom'].set_position(('axes', -0.02))
    ax.spines['left'].set_position(('axes', -0.02))
