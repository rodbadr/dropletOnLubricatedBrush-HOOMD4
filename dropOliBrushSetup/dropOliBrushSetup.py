import math
import gsd
import gsd.hoomd
import hoomd
import numpy as np
import random
import sys
import time
from sklearn.cluster import DBSCAN
from collections import Counter

strFormat = '{:g}'

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


##################################################
### Set parameters and read initial conditions ###
##################################################

time_start = time.time()

args = str(sys.argv[1]) # read command line argument

paramInd = int(args) # change it to integer

paramSpace = np.genfromtxt('ParameterSpaceDropOliBrushSetup.csv', delimiter=',') # read parameter file

params = paramSpace[paramInd-1,:] # select specific parameter set

dirBrush = "../oliBrushSetup/" # directory of the brush initial condition
filerootBrush = "oliBrush" # file name root for the brush initial condition
dirLiq = "../slabPrep/" # directory of the source liquid slab
filerootLiq  = "liqSlabSecond" # file name root for the source liquid slab

fileroot_out = "dropOliBrush" # output file name root for the simulations

buf = 0.075 # buffer for the neighbor list (model dependent)

sig_b = 1.0; # Wall LJ length scale parameter
eps_b = 1.0; # Wall LJ potential depth
T = 1.0; # temperature for Langevin Thermostat
k = 20.0; # bond spring constant
r0 = 1.0; # bond equilibrium position if harmonic/maximum extension if FENE
rc = 1.0; # cutoff radius for dpd attraction
rd = 0.8; # cutoff radius for mdpd repulsion
gamma = 4.5 # drag coefficient
alpha = 15/(3.1415*rd**3) # coefficient by which HOOMD repulsion parameter B is different from papers

dt = 1e-3 # simulation time step

mPoly = 1 # mass of polymer monomers. mass of PDMS repeat unit: 74 g/mol
mLiq = 1 # mass of liquid particles. mass of water molecule: 18 g/mol

print(params)

i = 0
dropRadius = float(params[i]) # radius of the liquid hemi-sphere

i = i+1
brushLen = int(params[i]) # number of monomers per brush polymer
i = i+1
Nbrush1 = int(params[i]) # number of polymers in each direction. Total number of polymers is Nbrush1*Nbrush2
i = i+1
Nbrush2 = int(params[i]); # number of polymers in each direction. Total number of polymers is Nbrush1*Nbrush2
i = i+1
brushDist = float(params[i])*rc; # distance between brush grafting sites

i=i+1
Noli= int(params[i]) # number of oligomers/lubricant chains
i=i+1
oliLen = int(params[i]) # oligomer length

### particle types are: A-grafted monomer, B-brush/gel monomers
i = i+1
Amm = float(params[i]); # monomer-monomer interaction
i = i+1
All = float(params[i]); # liquid-liquid interaction
i = i+1
Aml = float(params[i]); # monomer-liquid interaction
i = i+1
B = float(params[i]); # density dependent repulsion strength. Needs to be the same for all species. (See no-go theorem in many-body dissipative particle dynamics)

i = i+1
nsteps = int(params[i]) # number of simulation steps
i = i+1
Nmeas = int(params[i]) # number of snapshots/measurements

i = i+1
animate = int(params[i]); # 0 - Do note print equilibration trajectory; 1 - print equilibration trajectory

i = i+1
SimID = int(params[i]) # simulation ID

NmonBrush = int(brushLen*Nbrush1*Nbrush2); # total number of monomers in brush
NmonOli = int(oliLen*Noli); # total number of oligomer monomers
NbondsBrush = int((brushLen-1)*Nbrush1*Nbrush2) # number of brush bonds
NbondsOli = int((oliLen-1)*Noli) # number of oligomer bonds

NmonTot = int(NmonBrush + NmonOli) # total number of polymer particles
NbondsTot = int(NbondsBrush + NbondsOli) # total number of bonds

meas_period = int(nsteps/Nmeas) # period of dumping frames / making measurements

seed1 = random.randint(1,65535) # seed for random number generation


########################################
### Start Reading Initial Conditions ###
########################################

Lx = brushDist*Nbrush1
Ly = brushDist*Nbrush2

### read and setup brush particles

sinBrush = '_' + strFormat.format(brushLen) + '_' + strFormat.format(Nbrush1) + '_' + strFormat.format(Nbrush2) + '_' + strFormat.format(oliLen) +'_' + strFormat.format(Noli) +'_' + strFormat.format(brushDist) +'_' + strFormat.format(-Amm) + '_' + strFormat.format(B) + '_' + strFormat.format(SimID);

inputFnameBrush = "initialConditions/" + filerootBrush + sinBrush + ".gsd"

tempSnap = gsd.hoomd.open(name=f"{dirBrush}{inputFnameBrush}",mode='r')
snapBrush = tempSnap[0]

LzBrush = snapBrush.configuration.box[2];

brushHeight =  snapBrush.particles.position[0:NmonBrush,2].max()-snapBrush.particles.position[0:NmonBrush,2].min()# brush thickness in the initial condition for the substrate

positionsBrush = snapBrush.particles.position
bondsBrush = snapBrush.bonds.group
typeID_Brush = snapBrush.particles.typeid

for i in range(Noli):

    shift = 0

    for j in range(oliLen-1):

        dz = positionsBrush[NmonBrush+i*oliLen+j,2] - positionsBrush[NmonBrush+i*oliLen+j+1,2]

        if (abs(dz)>LzBrush/2):
            shift = 1

    if(shift == 1):

        tempz = np.zeros(oliLen)
        tempz[:] = positionsBrush[NmonBrush+i*oliLen:NmonBrush+(i+1)*oliLen,2]
        tempz[tempz<0] = tempz[tempz<0] + LzBrush
        positionsBrush[NmonBrush+i*oliLen:NmonBrush+(i+1)*oliLen,2] = tempz[:];


sinLiq = '_' + strFormat.format(1) + '_' + strFormat.format(-All) + '_' + strFormat.format(B) + '_' + strFormat.format(100) + '_' + strFormat.format(100)

inputFnameLiq = "initialConditions/" + filerootLiq + sinLiq + ".gsd"

tempSnap = gsd.hoomd.open(name=f"{dirLiq}{inputFnameLiq}",mode='r')
snapLiq = tempSnap[0]

LzLiq = snapLiq.configuration.box[2];


comZslabLiq = computeCOM(snapLiq.particles.position[:,2] + LzLiq/2, LzLiq) - LzLiq/2 # center of mass of the slab

snapLiq.particles.position[:,2] = (snapLiq.particles.position[:,2] - comZslabLiq + LzLiq/2) % LzLiq - LzLiq/2 # center the slab in z-direction

positionsLiq = snapLiq.particles.position


#### using clustering analysis to find the particles in the slab (largest cluster)
#### and finding its top and bottom boundaries

maxDist = 1.5*rc

clustering = DBSCAN(eps=maxDist, min_samples=1).fit(positionsLiq)

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
largest_cluster_positions = positionsLiq[largest_cluster_indices]

bottomLiq = largest_cluster_positions[:,2].min(); # bottom of the slab
topLiq = largest_cluster_positions[:,2].max(); # top of the slab

# calculate radial position of particles
rr2 = positionsLiq[:,0]**2 + positionsLiq[:,1]**2 + (positionsLiq[:,2] - bottomLiq)**2

mask1 = ( rr2 < dropRadius**2 ) # take only particles within the chosen radius
mask2 = ( positionsLiq[:,2] > bottomLiq ) # take only particles within the slab
mask  = mask1 & mask2

positionsLiq = positionsLiq[mask]

Nliq = positionsLiq.shape[0]

Lz = math.ceil(LzBrush + LzLiq + sig_b + 30)

WallPos = -Lz/2 + sig_b # wall Position

# shifting z coordinates to desired positions

positionsBrush[:,2] = positionsBrush[:,2] - positionsBrush[:,2].min() + WallPos
positionsLiq[:,2] = positionsLiq[:,2] - positionsLiq[:,2].min() + WallPos + brushHeight - 2*rc

Ntot = NmonTot + Nliq

positions = np.zeros((Ntot, 3))
positions[:NmonTot] = positionsBrush[:]
positions[NmonTot:] = positionsLiq[:]


bond_pairs = np.zeros((NbondsTot,2))

bond_pairs[:] = bondsBrush

##################################
### Start HOOMD Initialization ###
##################################

frame = gsd.hoomd.Frame()

#particle initialization
frame.particles.N = Ntot
frame.particles.position = np.zeros((Ntot, 3))
frame.particles.position[:] = positions
frame.particles.velocity = np.zeros((Ntot, 3))
frame.particles.types = ['A','B','C']
frame.particles.typeid = np.zeros(Ntot)
frame.particles.typeid[:NmonTot] = typeID_Brush
frame.particles.typeid[NmonTot:] = 2
frame.particles.mass = np.zeros(Ntot)
frame.particles.mass[:NmonTot] = np.ones(NmonTot)*mPoly
frame.particles.mass[NmonTot:] = np.ones(Nliq)*mLiq

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





sout = '_' + strFormat.format(dropRadius) + '_' + strFormat.format(brushLen) + '_' + strFormat.format(Nbrush1) + '_' + strFormat.format(Nbrush2) + '_' + strFormat.format(oliLen) +'_' + strFormat.format(Noli) + '_' + strFormat.format(brushDist) + '_' + strFormat.format(-Amm) + '_' + strFormat.format(-All) + '_' + strFormat.format(-Aml) + '_' + strFormat.format(B) + '_' + strFormat.format(SimID);

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
nl = hoomd.md.nlist.Tree(buffer = buf, exclusions = ())

#disipative particle dynamics force
dpd = hoomd.md.pair.DPD(nlist=nl, kT=T, default_r_cut=rc)
dpd.params[('A', 'A')] = dict(A = Amm, gamma = gamma)
dpd.params[('A', 'B')] = dict(A = Amm, gamma = gamma)
dpd.params[('A', 'C')] = dict(A = Aml, gamma = gamma)
dpd.params[('B', 'B')] = dict(A = Amm, gamma = gamma)
dpd.params[('B', 'C')] = dict(A = Aml, gamma = gamma)
dpd.params[('C', 'C')] = dict(A = All, gamma = gamma)

#many-body interactions
sqd = hoomd.md.many_body.SquareDensity(nl, default_r_cut = rd)
sqd.params[('A', 'A')] = dict(A=0, B=B/alpha)
sqd.params[('A', 'B')] = dict(A=0, B=B/alpha)
sqd.params[('A', 'C')] = dict(A=0, B=B/alpha)
sqd.params[('B', 'B')] = dict(A=0, B=B/alpha)
sqd.params[('B', 'C')] = dict(A=0, B=B/alpha)
sqd.params[('C', 'C')] = dict(A=0, B=B/alpha)

#harmonic force between bonds
if(brushLen > 1 or oliLen > 1):
    harmonic = hoomd.md.bond.Harmonic()
    harmonic.params[bondType] = dict(k=k, r0=r0)

#wall with forces
walls = [hoomd.wall.Plane(origin=(0, 0, WallPos), normal=(0, 0, 1), open=True), hoomd.wall.Plane(origin=(0, 0, WallPos), normal=(0, 0, -1), open= True)]
lj = hoomd.md.external.wall.LJ(walls=walls)
lj.params['A'] = {"sigma": sig_b, "epsilon": 0.0, "r_cut": 0.0}
lj.params['B'] = {"sigma": sig_b, "epsilon": eps_b, "r_cut": sig_b*2.0**(1.0/6.0)}
lj.params['C'] = {"sigma": sig_b, "epsilon": eps_b, "r_cut": sig_b*2.0**(1.0/6.0)}


#set the integrator
integrator = hoomd.md.Integrator(dt=dt)


integrator.forces.append(dpd)
integrator.forces.append(sqd)
integrator.forces.append(lj)

if(brushLen > 1 or oliLen > 1):
    integrator.forces.append(harmonic)


# # Custom action
dt_time = 1000
time_start = time.time()
old_time = 0
old_step = 0
time_action = PrintTimestep(time_start)
time_writer = hoomd.write.CustomWriter(action=time_action, trigger=hoomd.trigger.Periodic(dt_time))
simulation.operations.writers.append(time_writer)

all = hoomd.filter.All()
grafted = hoomd.filter.Type(['A'])
mobile = hoomd.filter.Type(['B','C'])

# defining triggers
traj_trig = hoomd.trigger.Periodic(period=meas_period, phase=0)

if(animate==1):
    trajFname = "trajectories/" + fileroot_out + "Traj" + sout + ".gsd"
    hoomd.write.GSD.write(state=simulation.state, filename=trajFname, mode='wb')
    dump_gsd = hoomd.write.GSD(trigger = traj_trig, filter = all, filename=trajFname, mode='ab', dynamic = ['property', 'momentum', 'attribute', 'topology'])
    simulation.operations.writers.append(dump_gsd)

#periodic single frame dump in case of crashes
restart_gsd = hoomd.write.GSD(trigger = traj_trig, filter = all, filename=initialFname, mode='ab', truncate=True)
simulation.operations.writers.append(restart_gsd)

nve = hoomd.md.methods.ConstantVolume(filter=mobile)
integrator.methods.append(nve)

simulation.operations.integrator = integrator

simulation.state.thermalize_particle_momenta(filter=mobile, kT=T)

print('Simulation Starting')

simulation.run(nsteps)

hoomd.write.GSD.write(state=simulation.state, filename=initialFname, mode='wb')

print('Simulation ended')
