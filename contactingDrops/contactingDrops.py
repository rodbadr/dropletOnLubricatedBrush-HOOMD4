import math
import hoomd
import hoomd.md
import gsd
import gsd.hoomd
import numpy as np
import time
import random
import sys
from sklearn.cluster import DBSCAN
from collections import Counter
import polymerAnalysisToolbox as pat

pi = np.pi

strFormat = '{:g}'


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


##################################################
### Set parameters and read initial conditions ###
##################################################

args = str(sys.argv[1])

paramInd = int(args)

# paramSpace = np.genfromtxt('ParameterSpaceDroplet.csv', delimiter=',')
paramSpace = np.genfromtxt('ParameterSpaceContactingDrops.csv', delimiter=',')

params = paramSpace[paramInd-1,:]


dir_in = '../slabPrep/' # input simulations directory
fileroot_in = 'liqSlab' # input simulations file root

fileroot_out = 'contactingDrops'


buf = 0.075

animate = 1; # 0 - Do note print equilibration trajectory; 1 - print equilibration trajectory

sig_b = 1.0; # Wall LJ length scale parameter
eps_b = 1.0; # Wall LJ potential depth
T = 1.0; # temperature for Langevin Thermostat
k = 20.0; # bond spring constant
r0 = 1.0; # bond equilibrium position if harmonic/maximum extension if FENE
rc = 1.0; # cutoff radius for dpd attraction
rd = 0.8; # cutoff radius for mdpd repulsion
gamma = 4.5 # drag coefficient
alpha = 15/(3.1415*rd**3) # coefficient by which HOOMD repulsion parameter B is different from papers
dt = 1e-3

i=0
oliLen1 = int(params[i]); # number of monomers per oligomer chain
i=i+1
oliRadius1 = float(params[i]); #radius of the sphere
i=i+1

oliLen2 = int(params[i]); # number of monomers per oligomer chain
i=i+1
oliRadius2 = float(params[i]); #radius of the sphere
i=i+1

### particle types are: A-grafted monomer, B-brush/gel monomers, C-oligomer monomers, D-fluid particles
### mdpd interaction parameters in order: [AA,AB,AC,AD,BB,BC,BD,CC,CD,DD]
A11 = float(params[i]); # monomer-monomer interaction
i=i+1
A22 = float(params[i]); # fluid-fluid interaction
i=i+1
A12 = float(params[i]); # monomer-fluid interaction
i=i+1
B = float(params[i]); # density dependent repulsion strength. Needs to be the same for all species. (See no-go theorem in many-body dissipative particle dynamics)
i=i+1

Lxin = float(params[i]);
i=i+1
Lyin = float(params[i]);
i=i+1

nsteps = int(params[i])
nsteps = int(4e5)
i=i+1
Nmeas = int(params[i])
i=i+1
meas_period = int(nsteps/Nmeas)

animate = int(params[i]) # 0 - Do note print equilibration trajectory; 1 - print equilibration trajectory
animate = int(0) # 0 - Do note print equilibration trajectory; 1 - print equilibration trajectory
i=i+1

SimID = int(params[i])

print(params)

seed1 = random.randint(1,65535)

######################
### process slab 1 ###
######################


sin1 = '_' + strFormat.format(oliLen1)  + '_' + strFormat.format(-A11) + '_' + strFormat.format(B) + '_' + strFormat.format(Lxin) + '_' + strFormat.format(Lyin)

inputFname1 = "initialConditions/" + fileroot_in + "Second" + sin1 + ".gsd"

tempSnap = gsd.hoomd.open(name=f"{dir_in}{inputFname1}",mode='r')

snap1 = tempSnap[0]

Ntot1 = snap1.particles.N
Noli1 = int(Ntot1/oliLen1)

Lx1 = snap1.configuration.box[0]
Ly1 = snap1.configuration.box[1]
Lz1 = snap1.configuration.box[2]


comZslab1 = pat.computeCOM_periodic(snap1.particles.position[:,2] + Lz1/2, Lz1) - Lz1/2 # center of mass of the slab

snap1.particles.position[:,2] = (snap1.particles.position[:,2] - comZslab1 + Lz1/2) % Lz1 - Lz1/2 # center the slab in z-direction

#### using clustering analysis to find the particles in the slab (largest cluster)
#### and finding its top and bottom boundaries

maxDist = 1.5*rc

clustering = DBSCAN(eps=maxDist, min_samples=1).fit(snap1.particles.position)

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
largest_cluster_positions = snap1.particles.position[largest_cluster_indices]

bottom1 = largest_cluster_positions[:,2].min(); # bottom of the slab
top1 = largest_cluster_positions[:,2].max(); # top of the slab

if ( (top1 - bottom1) < 2.0*oliRadius1 ):
    print("WARNING: Oligomer slab 1 thickness less than droplet diameter.")
    

mid1 = 0.5*(bottom1 + top1)


### compute centers of mass of the chains

comxArr1 = np.zeros(Noli1)
comyArr1 = np.zeros(Noli1)
comzArr1 = np.zeros(Noli1)

for chain in range(Noli1):

    ind1 = chain*oliLen1
    ind2 = (chain+1)*oliLen1

    comxArr1[chain] = pat.computeCOM_periodic(snap1.particles.position[ind1:ind2,0] + Lx1/2, Lx1) - Lx1/2
    comyArr1[chain] = pat.computeCOM_periodic(snap1.particles.position[ind1:ind2,1] + Ly1/2, Ly1) - Ly1/2
    comzArr1[chain] = pat.computeCOM_periodic(snap1.particles.position[ind1:ind2,2] + Lz1/2, Lz1) - Lz1/2



rhoCOM1 = np.sqrt( (comxArr1)**2 + (comyArr1)**2 + (comzArr1 - mid1)**2 )

mask = rhoCOM1 < oliRadius1 # select chains within radius

mask = np.repeat(mask, oliLen1) # repeat the mask to apply it to monomers instead of chains


Ntot1 = int(mask.sum()) # number of particles remaining
Noli1 = int(Ntot1/oliLen1) # number of chains remaining

positions1 = np.zeros((Ntot1,3))
positions1[:] = snap1.particles.position[mask] # positions of selected particles

Nbonds1 = Noli1*(oliLen1-1) # total number of bonds

if(oliLen1 > 1): # assign bond pairs to array ([Nbonds,2])


    bondsNew1 = np.zeros((Nbonds1,2))

    for chain in range(Noli1):

        ind1 = chain*oliLen1
        ind2 = (chain+1)*oliLen1

        indBond1 = chain*(oliLen1-1)
        indBond2 = indBond1+(oliLen1-1)


        bondsNew1[indBond1:indBond2,0] = np.linspace(ind1,ind2 - 2,oliLen1-1);
        bondsNew1[indBond1:indBond2,1] = np.linspace(ind1+1,ind2 - 2 + 1,oliLen1-1);

######################
### process slab 2 ###
######################

sin2 = '_' + strFormat.format(oliLen2)  + '_' + strFormat.format(-A22) + '_' + strFormat.format(B) + '_' + strFormat.format(Lxin) + '_' + strFormat.format(Lyin)

inputFname2 = "initialConditions/" + fileroot_in + "Second" + sin2 + ".gsd"

tempSnap = gsd.hoomd.open(name=f"{dir_in}{inputFname2}",mode='r')

snap2 = tempSnap[0]

Ntot2 = snap2.particles.N
Noli2 = int(Ntot2/oliLen2)

Lx2 = snap2.configuration.box[0]
Ly2 = snap2.configuration.box[1]
Lz2 = snap2.configuration.box[2]


comZslab2 = pat.computeCOM_periodic(snap2.particles.position[:,2] + Lz2/2, Lz2) - Lz2/2 # center of mass of the slab

snap2.particles.position[:,2] = (snap2.particles.position[:,2] - comZslab2 + Lz2/2) % Lz2 - Lz2/2 # center the slab in z-direction



#### using clustering analysis to find the particles in the slab (largest cluster)
#### and finding its top and bottom boundaries

maxDist = 1.5*rc

clustering = DBSCAN(eps=maxDist, min_samples=1).fit(snap2.particles.position)

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
largest_cluster_positions = snap2.particles.position[largest_cluster_indices]

bottom2 = largest_cluster_positions[:,2].min(); # bottom of the slab
top2 = largest_cluster_positions[:,2].max(); # top of the slab

if ( (top2 - bottom2) < 2.0*oliRadius2 ):
    print("WARNING: Oligomer slab 2 thickness less than droplet diameter.")

mid2 = 0.5*(bottom2 + top2)

### compute centers of mass of the chains

comxArr2 = np.zeros(Noli2)
comyArr2 = np.zeros(Noli2)
comzArr2 = np.zeros(Noli2)

for chain in range(Noli2):

    ind1 = chain*oliLen2
    ind2 = (chain+1)*oliLen2

    comxArr2[chain] = pat.computeCOM_periodic(snap2.particles.position[ind1:ind2,0] + Lx2/2, Lx2) - Lx2/2
    comyArr2[chain] = pat.computeCOM_periodic(snap2.particles.position[ind1:ind2,1] + Ly2/2, Ly2) - Ly2/2
    comzArr2[chain] = pat.computeCOM_periodic(snap2.particles.position[ind1:ind2,2] + Lz2/2, Lz2) - Lz2/2


rhoCOM2 = np.sqrt( (comxArr2)**2 + (comyArr2)**2 + (comzArr2 - mid2)**2 )

mask = rhoCOM2 < oliRadius2 # select chains within radius

mask = np.repeat(mask, oliLen2) # repeat the mask to apply it to monomers instead of chains

Ntot2 = int(mask.sum()) # number of particles remaining
Noli2 = int(Ntot2/oliLen2) # number of chains remaining

positions2 = np.zeros((Ntot2,3))
positions2[:] = snap2.particles.position[mask] # positions of selected particles

Nbonds2 = Noli2*(oliLen2-1) # total number of bonds

if(oliLen2 > 1): # assign bond pairs to array ([Nbonds,2])

    bondsNew2 = np.zeros((Nbonds2,2))

    for chain in range(Noli2):

        ind1 = chain*oliLen2 + Ntot1
        ind2 = (chain+1)*oliLen2 + Ntot1

        indBond1 = chain*(oliLen2-1) + Nbonds1
        indBond2 = indBond1+(oliLen2-1)


        bondsNew2[indBond1:indBond2,0] = np.linspace(ind1,ind2 - 2,oliLen2-1);
        bondsNew2[indBond1:indBond2,1] = np.linspace(ind1+1,ind2 - 2 + 1,oliLen2-1);


##########################
### finalize positions ###
##########################

# choose the postions of the centers of the droplets along z-axis
# such that the droplets are almost tangent and the center of mass is at z=0

radPos1 = - (oliRadius1 + oliRadius2)/(1 + (oliRadius1/oliRadius2)**3 )
radPos2 = (oliRadius1 + oliRadius2)/(1 + (oliRadius2/oliRadius1)**3 )

positions1[:,2] = positions1[:,2] - (mid1 - radPos1)
positions2[:,2] = positions2[:,2] - (mid2 - radPos2)


Ntot = Ntot1 + Ntot2
NbondsTot = Nbonds1 + Nbonds2

positions = np.zeros((Ntot, 3))
positions[:Ntot1] = positions1[:]
positions[Ntot1:] = positions2[:]

Lx = np.max( [oliRadius1*2, oliRadius2*2])*2 + 5*rc
Ly = np.max( [oliRadius1*2, oliRadius2*2])*2 + 5*rc
Lz = (positions[:,2].max() - positions[:,2].min() )*2


comZslab = pat.computeCOM_periodic(positions[:,2] + Lz/2,Lz) - Lz/2 # center of mass of the slab

positions[:,2] = (positions[:,2] - comZslab + Lz/2)%Lz - Lz/2 # center the slab in z-direction



if NbondsTot>0:

    bond_pairs = np.zeros((NbondsTot,2))

    if Nbonds1>0:
        bond_pairs[:Nbonds1] = bondsNew1

    if Nbonds2>0:
        bond_pairs[Nbonds1:] = bondsNew2





##################################
### Start HOOMD Initialization ###
##################################

frame = gsd.hoomd.Frame()

#particle initialization
frame.particles.N = Ntot
frame.particles.position = np.zeros((Ntot, 3))
frame.particles.position[:] = positions
frame.particles.velocity = np.zeros((Ntot, 3))
frame.particles.types = ['A','B']
frame.particles.typeid = np.zeros(Ntot)
frame.particles.typeid[:Ntot1] = 0
frame.particles.typeid[Ntot1:] = 1
frame.particles.mass = np.ones(Ntot)

#bond initialization
if(NbondsTot > 0):

    frame.bonds.N = NbondsTot
    frame.bonds.types = ['Polymer']
    frame.bonds.typeid = [0]*NbondsTot
    frame.bonds.group = bond_pairs

#box initialization
frame.configuration.dimensions = 3
frame.configuration.box =  [Lx, Ly, Lz, 0, 0, 0]
frame.configuration.step = 0

sout = '_' + strFormat.format(oliLen1) + '_' + strFormat.format(oliRadius1) + '_' + strFormat.format(oliLen2) + '_' + strFormat.format(oliRadius2) + '_' + strFormat.format(-A11) + '_' + strFormat.format(-A22) + '_' + strFormat.format(-A12) + '_' + strFormat.format(B) + '_' + strFormat.format(SimID)


initialFname = "initialConditions/" + fileroot_out + sout + ".gsd"

#saving the frame into gsd file
with gsd.hoomd.open(name=initialFname, mode='w',) as f:
    f.append(frame)


#initializing the simulation with the gsd file
gpu = hoomd.device.GPU()
simulation = hoomd.Simulation(device=gpu, seed=seed1)
simulation.create_state_from_gsd(filename=initialFname)


try:
    bondType = simulation.state.types['bond_types'][0];

except:
    print("No bonds");


#neighbour list via cell
nl = hoomd.md.nlist.Cell(buffer = buf, exclusions = ())

#disipative particle dynamics force
dpd = hoomd.md.pair.DPD(nlist=nl, kT=T, default_r_cut=rc)
dpd.params[('A', 'A')] = dict(A = A11, gamma = gamma)
dpd.params[('A', 'B')] = dict(A = A12, gamma = gamma)
dpd.params[('B', 'B')] = dict(A = A22, gamma = gamma)


#many-body interactions
sqd = hoomd.md.many_body.SquareDensity(nl, default_r_cut = rd)
sqd.params[('A', 'A')] = dict(A=0, B=B/alpha)
sqd.params[('A', 'B')] = dict(A=0, B=B/alpha)
sqd.params[('B', 'B')] = dict(A=0, B=B/alpha)

#harmonic force between bonds
if(NbondsTot > 0):
    harmonic = hoomd.md.bond.Harmonic()
    harmonic.params[bondType] = dict(k=k, r0=r0)

#set the integrator
integrator = hoomd.md.Integrator(dt=dt)


integrator.forces.append(dpd)
integrator.forces.append(sqd)

if(NbondsTot > 0):
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

nve = hoomd.md.methods.ConstantVolume(filter=all)
integrator.methods.append(nve)

simulation.operations.integrator = integrator

simulation.state.thermalize_particle_momenta(filter=all, kT=T)

print('Simulation Starting')

simulation.run(nsteps)


### dump final frame
hoomd.write.GSD.write(state=simulation.state, filename=initialFname, mode='wb')

print('Simulation ended')
