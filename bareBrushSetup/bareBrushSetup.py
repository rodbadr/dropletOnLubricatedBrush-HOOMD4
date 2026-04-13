import math
import gsd
import gsd.hoomd
import hoomd
import numpy as np
import random
import sys
import time

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



##################################################
### Set parameters and read initial conditions ###
##################################################

time_start = time.time()

args = str(sys.argv[1]) # read command line argument

paramInd = int(args) # change it to integer

paramSpace = np.genfromtxt('ParameterSpaceBareBrushSetup.csv', delimiter=',') # read parameter file

params = paramSpace[paramInd-1,:] # select specific parameter set

fileroot_out = "bareBrush" # output file name root for the simulations

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
brushLen = int(params[i]) # number of monomers per brush polymer
i = i+1
Nbrush1 = int(params[i]) # number of polymers in each direction. Total number of polymers is Nbrush1*Nbrush2
i = i+1
Nbrush2 = int(params[i]); # number of polymers in each direction. Total number of polymers is Nbrush1*Nbrush2
i = i+1
brushDist = float(params[i])*rc; # distance between brush grafting sites

i = i+1
Amm = float(params[i]); # monomer-monomer interaction
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
Ntot = int(NmonBrush); # total number of particles in simulation box
Nbonds = int( (brushLen-1)*Nbrush1*Nbrush2 ) # total number of bonds

meas_period = int(nsteps/Nmeas) # period of dumping frames / making measurements

seed1 = random.randint(1,65535)

##################################
### Start HOOMD Initialization ###
##################################

Lx = brushDist*Nbrush1
Ly = brushDist*Nbrush2
Lz = brushLen + 4*rc


WallPos = -Lz/2 + sig_b # wall Position

brushHeight = rc*(brushLen - 1) # position of highest monomer in the initial condition for the substrate

zcoords = np.linspace(WallPos,WallPos + rc*brushHeight,brushLen) # initial z coordinates of brush monomers

tempx = -Lx/2 + brushDist/2


#auxiliar vector of zeros to initialize
positions = np.zeros((Ntot,3))
bond_pairs = np.zeros((Nbonds,2))
particles_id = np.zeros(Ntot)

for i in range(Nbrush1):
    tempy = -Ly/2 + brushDist/2
    for j in range(Nbrush2):
        ind = i*Nbrush2 + j ### index of the brush being initialized

        ### position initialization
        positions[ind*brushLen:(ind+1)*brushLen,0] = tempx
        positions[ind*brushLen:(ind+1)*brushLen,1] = tempy
        positions[ind*brushLen:(ind+1)*brushLen,2] = zcoords[:]

        ### type initialization (0 is grafted, 1 is normal)
        particles_id[ind*brushLen]=0
        particles_id[ind*brushLen+1:(ind+1)*brushLen]=1

        ### bond initialization
        if(brushLen > 1):
            bond_pairs[ind*(brushLen-1):(ind+1)*(brushLen-1),0] = np.linspace(ind*brushLen,(ind+1)*brushLen - 2,brushLen-1)
            bond_pairs[ind*(brushLen-1):(ind+1)*(brushLen-1),1] = np.linspace(ind*brushLen+1,(ind+1)*brushLen - 2 + 1,brushLen-1)

        tempy = tempy + brushDist

    tempx = tempx + brushDist

sout = '_' + strFormat.format(brushLen) + '_' + strFormat.format(Nbrush1) + '_' + strFormat.format(Nbrush2) + '_' + strFormat.format(brushDist) + '_' + strFormat.format(SimID)


#frame initialization
frame = gsd.hoomd.Frame()

#particle initialization
frame.particles.N = Ntot
frame.particles.types = ['A','B']
frame.particles.typeid = particles_id
frame.particles.position = positions
frame.particles.mass = np.ones(Ntot)*mPoly


#bond initialization
frame.bonds.N = Nbonds
frame.bonds.types = ['Polymer']
frame.bonds.typeid = [0]*Nbonds
frame.bonds.group = bond_pairs

frame.configuration.dimensions = 3
frame.configuration.box =  [Lx, Ly, Lz, 0, 0, 0]
frame.configuration.step = 0

#creating gsd file with the initialized frame


initialFname = "initialConditions/" + fileroot_out + sout + ".gsd"

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
dpd.params[('B', 'B')] = dict(A = Amm, gamma = gamma)

#many-body interactions
sqd = hoomd.md.many_body.SquareDensity(nl, default_r_cut = rd)
sqd.params[('A', 'A')] = dict(A=0, B=B/alpha)
sqd.params[('A', 'B')] = dict(A=0, B=B/alpha)
sqd.params[('B', 'B')] = dict(A=0, B=B/alpha)

#harmonic force between bonds
if(brushLen > 1 or oliLen > 1):
    harmonic = hoomd.md.bond.Harmonic()
    harmonic.params[bondType] = dict(k=k, r0=r0)

#wall with forces
walls = [hoomd.wall.Plane(origin=(0, 0, WallPos), normal=(0, 0, 1), open=True), hoomd.wall.Plane(origin=(0, 0, WallPos), normal=(0, 0, -1), open= True)]
lj = hoomd.md.external.wall.LJ(walls=walls)
lj.params['A'] = {"sigma": sig_b, "epsilon": 0.0, "r_cut": 0.0}
lj.params['B'] = {"sigma": sig_b, "epsilon": eps_b, "r_cut": sig_b*2.0**(1.0/6.0)}


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

print('Simulation Starting')

simulation.run(nsteps)

hoomd.write.GSD.write(state=simulation.state, filename=initialFname, mode='wb')

print('Simulation ended')
