import hoomd
import hoomd.md
import gsd
import gsd.hoomd
import numpy as np
import time
import random
import math
import sys
from sklearn.cluster import DBSCAN
from collections import Counter

pi = np.pi

class PrintTimestep(hoomd.custom.Action):

    def __init__(self, t_start):
        self._t_start = t_start

    def act(self, timestep):
        global old_time, old_step, meas_period
        current_time = time.time()
        current_time = current_time - self._t_start

        if( (current_time - old_time) > 10 ):

            if( (timestep - old_step) > meas_period/10):

                old_step = np.floor(timestep/meas_period)*meas_period

                for writer in simulation.operations.writers:
                    if hasattr(writer, 'flush'):
                        writer.flush()

            old_time = current_time

            tH = int(current_time/3600)
            tM = int( (current_time - tH*3600)/60 )
            tS = int( (current_time - tH*3600 - tM*60) )

            TPS = timestep/(current_time)

            ETA = nsteps/TPS  - current_time

            tH2 = int(ETA/3600)
            tM2 = int( (ETA - tH2*3600)/60 )
            tS2 = int( (ETA - tH2*3600- tM2*60) )

            print("Elapsed time {:02d}:{:02d}:{:02d}".format(tH,tM,tS) + " | Step {:d}/{:d} | TPS={:.6g} ".format(timestep,nsteps,TPS) + " | ETA {:02d}:{:02d}:{:02d}".format(tH2,tM2,tS2), flush=True )



def computeCOM(xx, Lx):

    th = xx / Lx * 2 * pi

    Xi = np.cos(th)

    Zeta = np.sin(th)

    Xi_bar = Xi.mean()

    Zeta_bar = Zeta.mean()

    th_bar = np.arctan2(-Zeta_bar, -Xi_bar) + pi

    com = Lx * th_bar / 2 / pi

    return com

strFormat = '{:g}'

##################################################
### Set parameters and read initial conditions ###
##################################################

args = str(sys.argv[1])

paramInd = int(args)

paramSpace = np.genfromtxt('ParameterSpacePoiseuille.csv', delimiter=',')

params = paramSpace[paramInd-1,:]

dir = "../slabPrep/initialConditions/"
fileroot_in = 'liqSlabSecond'

fileroot_out = "poiseuilleFlow"

buf = 0.075

animate = 1; # 0 - Do note print equilibration trajectory; 1 - print equilibration trajectory

sig_b = 1.0; # Wall LJ length scale parameter
eps_b = 1.0; # Wall LJ potential depth
T = 1.0; # temperature for Langevin Thermostat
k = 20.0; # bond spring constant
r0 = 1.0; # bond equilibrium position if harmonic/maximum extension if FENE
rc = 1.0; # cutoff radius for dpd attraction
rd = 0.8; # cutoff radius for mdpd repulsion
alpha = 15/(3.1415*rd**3) # coefficient by which HOOMD repulsion parameter B is different from papers

dt = 1e-3 # simulation timestep

i=0
oliLen = int(params[i])

### particle types are: A-grafted monomer, B-brush/gel monomers, C-oligomer monomers, D-fluid particles
### mdpd interaction parameters in order: [AA,AB,AC,AD,BB,BC,BD,CC,CD,DD]
i=i+1
Amm = float(params[i]); # monomer-monomer interaction
i=i+1
B = float(params[i]); # density dependent repulsion strength. Needs to be the same for all species. (See no-go theorem in many-body dissipative particle dynamics)

i=i+1
gamma = float(params[i]); # drag coefficient

i=i+1
Lxin = float(params[i])
i=i+1
Lyin = float(params[i])

i=i+1
Fext = float(params[i]); # pushing force

i=i+1
thick = float(params[i]); # thickness of the slab, choose large number to select the entire slab

i=i+1
freezeThick = float(params[i]); # thickness of the frozen layers at the top and the bottom

i=i+1
nsteps = int(params[i]);
i=i+1
Nmeas = int(params[i]);
meas_period = int(nsteps/Nmeas)

i=i+1
animate = int(params[i]); # 0 - Do note print equilibration trajectory; 1 - print equilibration trajectory

i=i+1
SimID = int(params[i]);

sinP =  '_' + strFormat.format(oliLen) + '_' + strFormat.format(-Amm) + '_' + strFormat.format(B) + '_' + strFormat.format(Lxin) + '_' + strFormat.format(Lyin);

tPend = gsd.hoomd.open(name=dir + fileroot_in + sinP + ".gsd",mode='r')

tempSnapPend = tPend[0]

Ntot = tempSnapPend.particles.N

Noli = int(Ntot/oliLen)

LzP = tempSnapPend.configuration.box[2];

posz = tempSnapPend.particles.position[:,2]

bonds = tempSnapPend.bonds.group


#### using clustering analysis to find the particles in the slab (largest cluster)
#### and finding its top and bottom boundaries

maxDist = 1.5*rc

clustering = DBSCAN(eps=maxDist, min_samples=1).fit(tempSnapPend.particles.position)

# Get cluster labels
labels = clustering.labels_

# Count the occurrences of each cluster label
label_counts = Counter(labels)

# Ignore noise points (label -1)
if -1 in label_counts:
    del label_counts[-1]

# Find the largest cluster (label with the maximum count)
largest_cluster_label = max(label_counts, key=label_counts.get)

# Get indices of points in the largest cluster
largest_cluster_indices = np.where(labels == largest_cluster_label)[0]

# Get the positions of particles in the largest cluster
largest_cluster_positions = tempSnapPend.particles.position[largest_cluster_indices]

bottom = largest_cluster_positions[:,2].min(); # bottom of the slab
top = largest_cluster_positions[:,2].max(); # top of the slab

### compute centers of mass of the chains

comzArr = np.zeros(Noli)

for chain in range(Noli):

    ind1 = chain*oliLen
    ind2 = (chain+1)*oliLen

    comzArr[chain] = computeCOM(posz[ind1:ind2] + LzP/2, LzP) - LzP/2


### choose only chains that are within the slab limits

mask = ( comzArr >  bottom )
mask2 = ( comzArr <  np.min((top,bottom+thick)) )
mask3 = mask & mask2


mask3 = np.repeat(mask3, oliLen) # repeat the mask to apply it to monomers instead of chains


Ntot = int(mask3.sum()) # number of particles remaining
Noli = int(Ntot/oliLen) # number of chains remaining

pos = np.zeros((Ntot,3))
pos[:] = tempSnapPend.particles.position[mask3] # positions of selected particles

comZslab = computeCOM(pos[:,2] + LzP/2,LzP) - LzP/2 # center of mass of the slab


Lx = Lxin
Ly = Lyin
Lz = pos[:,2].max() - pos[:,2].min() + 10*rc


pos[:,2] = (pos[:,2] - comZslab + Lz/2)%Lz - Lz/2 # center the slab in z-direction

if(oliLen > 1): # assign bond pairs to array ([Nbonds,2])


    Nbonds = Noli*(oliLen-1) # total number of bonds


    bond_pairs = np.zeros((Nbonds,2))

    for chain in range(Noli):

        ind1 = chain*oliLen
        ind2 = (chain+1)*oliLen

        indBond1 = chain*(oliLen-1)
        indBond2 = indBond1+(oliLen-1)


        bond_pairs[indBond1:indBond2,0] = np.linspace(ind1,ind2 - 2,oliLen-1);
        bond_pairs[indBond1:indBond2,1] = np.linspace(ind1+1,ind2 - 2 + 1,oliLen-1);


### compute centers of mass of the remaining chains

comzArr = np.zeros(Noli)

for chain in range(Noli):

    ind1 = chain*oliLen
    ind2 = (chain+1)*oliLen

    comzArr[chain] = computeCOM(pos[ind1:ind2,2] + LzP/2, LzP) - LzP/2

### choose the chains that are within 5*rc from the top or bottom
### to provide the fixed boundary


WallPosTop = (comzArr.max() - freezeThick*rc + 1.3*sig_b)
WallPosBot = (comzArr.min() + freezeThick*rc - 1.3*sig_b)

if (oliLen > 1): # in case of chains, shift by an additional 0.7*Rg to avoid blow ups
    WallPosTop = (comzArr.max() - freezeThick*rc + ( 1.3*sig_b + 0.7*oliLen**(3/5) ) )
    WallPosBot = (comzArr.min() + freezeThick*rc - ( 1.3*sig_b + 0.7*oliLen**(3/5) ) )

mask  = comzArr > (comzArr.max() - freezeThick*rc)
mask2 = comzArr < (comzArr.min() + freezeThick*rc)
mask3 = mask | mask2

mask3 = np.repeat(mask3, oliLen)

seed1 = random.randint(1,65535)

##################################
### Start HOOMD Initialization ###
##################################

frame = gsd.hoomd.Frame()

#particle initialization
frame.particles.N = Ntot
frame.particles.position = np.zeros((Ntot, 3))
frame.particles.position[:] = pos
frame.particles.velocity = np.zeros((Ntot, 3))
frame.particles.types = ['A','B']
frame.particles.typeid = np.zeros(Ntot)
frame.particles.typeid[~mask3] = 1
frame.particles.mass = np.ones(Ntot)

#bond initialization
if(oliLen > 1):

    NbondsTot = Nbonds
    frame.bonds.N = NbondsTot
    frame.bonds.types = ['Polymer']
    frame.bonds.typeid = [0]*NbondsTot
    frame.bonds.group = bond_pairs

#box initialization
frame.configuration.dimensions = 3
frame.configuration.box =  [Lx, Ly, Lz, 0, 0, 0]
frame.configuration.step = 0


sout = '_' + strFormat.format(oliLen) + '_' + strFormat.format(-Amm) + '_' + strFormat.format(B) + '_' + strFormat.format(gamma) + '_' + strFormat.format(Fext) + '_' + strFormat.format(thick) + '_' + strFormat.format(SimID);

initialFname = "initialConditions/" + fileroot_out + sout + ".gsd"

#saving the frame into gsd file
with gsd.hoomd.open(name=initialFname, mode='w',) as f:
    f.append(frame)

gpu = hoomd.device.GPU()
simulation = hoomd.Simulation(device=gpu, seed=seed1)
simulation.timestep = 0
simulation.create_state_from_gsd(filename=initialFname)


try:
    bondType = simulation.state.types['bond_types'][0];

except:
    print("No bonds");



#neighbour list via cell
nl = hoomd.md.nlist.Cell(buffer = buf, exclusions = ())

#disipative particle dynamics force
dpd = hoomd.md.pair.DPD(nlist=nl, kT=T, default_r_cut=rc)
dpd.params[('A', 'A')] = dict(A = Amm, gamma = gamma)
dpd.params[('A', 'B')] = dict(A = Amm, gamma = gamma)
dpd.params[('B', 'B')] = dict(A = Amm, gamma = gamma)


#many-body interactions
sqd = hoomd.md.many_body.SquareDensity(nl, default_r_cut = rd)
sqd.params[('A', 'A')] = dict(A=0, B=B/alpha)
sqd.params[('A', 'B')] = dict(A=0, B=B/alpha)
sqd.params[('B', 'B')] = dict(A=0, B=B/alpha)

#harmonic force between bonds
if(oliLen > 1):
    harmonic = hoomd.md.bond.Harmonic()
    harmonic.params[bondType] = dict(k=k, r0=r0)

#wall with forces

walls = [hoomd.wall.Plane(origin=(0, 0, WallPosBot), normal=(0, 0, 1), open=True), hoomd.wall.Plane(origin=(0, 0, WallPosTop), normal=(0, 0, -1), open= True)]
lj = hoomd.md.external.wall.LJ(walls=walls)
lj.params['A'] = {"sigma": sig_b, "epsilon": 0.0, "r_cut": 0.0}
lj.params['B'] = {"sigma": sig_b, "epsilon": eps_b, "r_cut": sig_b*2.0**(1.0/6.0)}

#set the integrator
integrator = hoomd.md.Integrator(dt=dt)


integrator.forces.append(dpd)
integrator.forces.append(sqd)
integrator.forces.append(lj)

if(oliLen > 1):
    integrator.forces.append(harmonic)



### Custom action for printing progress updates
dt_time = 1000
time_start = time.time()
old_time = 0
old_step = 0
time_action = PrintTimestep(time_start)
time_writer = hoomd.write.CustomWriter(action=time_action, trigger=hoomd.trigger.Periodic(dt_time))
simulation.operations.writers.append(time_writer)

all = hoomd.filter.All()
fixed = hoomd.filter.Type(['A'])
mobile = hoomd.filter.Type(['B'])

# defining triggers
traj_trig = hoomd.trigger.Periodic(period=meas_period, phase=0)

if(animate==1):
    trajFname = "trajectories/" + fileroot_out + "Traj" + sout + ".gsd"
    hoomd.write.GSD.write(state=simulation.state, filename=trajFname, mode='wb')
    dump_gsd = hoomd.write.GSD(trigger = traj_trig, filter = all, filename=trajFname, mode='ab', dynamic = ['property', 'momentum', 'attribute', 'topology'])
    simulation.operations.writers.append(dump_gsd)

#periodic single frame dump in case of crashes
restart_gsd = hoomd.write.GSD(trigger = traj_trig, filter = all, filename=initialFname, mode='ab', dynamic = ['property', 'momentum', 'attribute', 'topology'], truncate=True)
simulation.operations.writers.append(restart_gsd)

nve = hoomd.md.methods.ConstantVolume(filter=mobile)
integrator.methods.append(nve)

simulation.operations.integrator = integrator

simulation.state.thermalize_particle_momenta(filter=mobile, kT=T)


if (abs(Fext) > 0):
    constant = hoomd.md.force.Constant( filter=mobile )
    constant.constant_force['B'] = (Fext,0,0)

    integrator.forces.append(constant)

print('Simulation Starting')

simulation.run(nsteps)

hoomd.write.GSD.write(state=simulation.state, filename=initialFname, mode='wb')

print('Simulation ended')
