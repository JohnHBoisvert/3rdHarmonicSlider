from pylab import *
from scipy import *
from scipy.optimize import fsolve
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.stats import gaussian_kde
import csv
import math
from astropy.io import ascii
from operator import itemgetter, attrgetter
import operator
###This is for debugging
### sys.exit("Terminating program now, as requested...")
import sys

def dotdiffbtwen2and3(K,e,M,omega):
    return K * (9.0/8.0) * e * e * np.sin(3.0 * M + omega) - K * e * np.sin(2.0 * M + omega)

#S= np.arange(0,2.0*np.pi,0.001)*100.
S= np.arange(0,2.0*np.pi,0.01)
#S= np.arange(0,100,1)*100.
KA = 100.0
omegaA = 0.0
eA = 1.
eB = 0.05

ax1 = plt.subplot(111)
subplots_adjust(left=0.15, bottom=0.35)
l, = ax1.plot(S,dotdiffbtwen2and3(KA,eA,S,omegaA))
grid(True)

ax2 = plt.subplot(111)
l2, = ax2.plot(S,dotdiffbtwen2and3(KA,eB,S,omegaA))

xlabel('M [rad]')
ylabel('Dot Diff Btwn 2nd $&$ 3rd Harm. [m s$^{-1}$]')

xlim([0,2.0*np.pi])
ylim([-250,250])

axcolor = 'lightgoldenrodyellow'
axKA = axes([0.15, 0.08, 0.65, 0.03], axisbg=axcolor)
axoA = axes([0.15, 0.13, 0.65, 0.03], axisbg=axcolor)
axeA = axes([0.15, 0.23, 0.65, 0.03], axisbg=axcolor)
axeB = axes([0.15, 0.18, 0.65, 0.03], axisbg=axcolor)

sKA = Slider(axKA, 'K', 0.5, 100.0, valinit=100.)
seA = Slider(axeA, 'e_red', 0.001, 1.0, valinit=eA)
seB = Slider(axeB, 'e_blue', 0.001, 1.0, valinit=eB)
soA = Slider(axoA, '$\omega$', 0.001, 2.0*np.pi, valinit=0.0)

def update(val):
    l.set_ydata(dotdiffbtwen2and3(sKA.val,seA.val,S,soA.val))
    l2.set_ydata(dotdiffbtwen2and3(sKA.val,seB.val,S,soA.val))

sKA.on_changed(update)
seA.on_changed(update)
seB.on_changed(update)
soA.on_changed(update)

resetax = plt.axes([0.15, 0.03, 0.65, 0.03])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def reset(event):
    sKA.reset()
    seA.reset()
    seB.reset()
    soA.reset()
button.on_clicked(reset)

plt.show()