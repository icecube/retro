import glob
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

flist = glob.glob('*.csv')
data = []
for fname in flist:
    with open(fname) as file:
        for line in file:
            if line.startswith('#'):
                continue
            else:
                l = eval('['+line+']')
                data.append(l)

data = np.array(data)
print data.shape
#event, t_true, x_true, y_true, z_true, theta_true, phi_true, trck_energy_true, cscd_energy_true, llh_true, t_retro, x_retro, y_retro, z_retro, theta_retro, phi_retro, llh_retro, t_mn, x_mn, y_mn, z_mn, theta_mn, phi_mn, llh_mn, t_spe, x_spe, y_spe, z_spe, theta_spe, phi_spe, llh_spe

#theta
bins = np.linspace(-np.pi,np.pi,51)
idx = 14
h = plt.hist(data[:,5] - data[:,idx+14],color='m', alpha=0.3, bins=bins)
h = plt.hist(data[:,5] - data[:,idx],color='b', alpha=0.3, bins=bins)
h = plt.hist(data[:,5] - data[:,idx+7],color='g', alpha=0.3, bins=bins)
plt.gca().set_xlabel(r'$\Delta(\vartheta_{true},\vartheta_{retro})$')
plt.savefig('delta_theta.png')
plt.clf()

#phi
bins = np.linspace(-2*np.pi,2*np.pi,51)
idx = 15
h = plt.hist(data[:,6] - data[:,idx+14],color='m', alpha=0.3, bins=bins)
h = plt.hist(data[:,6] - data[:,idx],color='b', alpha=0.3, bins=bins)
h = plt.hist(data[:,6] - data[:,idx+7],color='g', alpha=0.3, bins=bins)
plt.gca().set_xlabel(r'$\Delta(\phi_{true},\phi_{retro})$')
plt.savefig('delta_phi.png')
plt.clf()

#time
bins = np.linspace(-300,300,51)
idx = 10
h = plt.hist(data[:,1] - data[:,idx+14],color='m', alpha=0.3, bins=bins)
h = plt.hist(data[:,1] - data[:,idx],color='b', alpha=0.3, bins=bins)
h = plt.hist(data[:,1] - data[:,idx+7],color='g', alpha=0.3, bins=bins)
plt.gca().set_xlabel(r'$\Delta(t_{true},t_{retro})$')
plt.savefig('delta_t.png')
plt.clf()

#x
bins = np.linspace(-100,100,51)
idx = 11
h = plt.hist(data[:,2] - data[:,idx+14],color='m', alpha=0.3, bins=bins)
h = plt.hist(data[:,2] - data[:,idx],color='b', alpha=0.3, bins=bins)
h = plt.hist(data[:,2] - data[:,idx+7],color='g', alpha=0.3, bins=bins)
plt.gca().set_xlabel(r'$\Delta(x_{true},x_{retro})$')
plt.savefig('delta_x.png')
plt.clf()

#y
bins = np.linspace(-100,100,51)
idx = 12
h = plt.hist(data[:,3] - data[:,idx+14],color='m', alpha=0.3, bins=bins)
h = plt.hist(data[:,3] - data[:,idx],color='b', alpha=0.3, bins=bins)
h = plt.hist(data[:,3] - data[:,idx+7],color='g', alpha=0.3, bins=bins)
plt.gca().set_xlabel(r'$\Delta(y_{true},y_{retro})$')
plt.savefig('delta_y.png')
plt.clf()

#z
bins = np.linspace(-100,100,51)
idx = 13
h = plt.hist(data[:,4] - data[:,idx+14],color='m', alpha=0.3, bins=bins)
h = plt.hist(data[:,4] - data[:,idx],color='b', alpha=0.3, bins=bins)
h = plt.hist(data[:,4] - data[:,idx+7],color='g', alpha=0.3, bins=bins)
plt.gca().set_xlabel(r'$\Delta(z_{true},z_{retro})$')
plt.savefig('delta_z.png')
plt.clf()

#llh
bins = np.linspace(-30,30,51)
idx = 16
h = plt.hist(data[:,9] - data[:,idx+14],color='m', alpha=0.3, bins=bins)
h = plt.hist(data[:,9] - data[:,idx],color='b', alpha=0.3, bins=bins)
h = plt.hist(data[:,9] - data[:,idx+7],color='g', alpha=0.3, bins=bins)
plt.gca().set_xlabel(r'$\Delta(llh_{true},llh_{retro})$')
plt.savefig('delta_llh.png')
plt.clf()

