import itertools
import math
import gsd
import gsd.hoomd
import hoomd
import numpy as np
import random
import sys
import time
import sys

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
oliRadius1 = float(params[i]); # radius of the sphere
i=i+1

oliLen2 = int(params[i]); # number of monomers per oligomer chain
i=i+1
oliRadius2 = float(params[i]); # radius of the sphere
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
i=i+1
Nmeas = int(params[i])
i=i+1
meas_period = int(nsteps/Nmeas)

animate = int(params[i]) # 0 - Do note print equilibration trajectory; 1 - print equilibration trajectory
i=i+1

SimID = int(params[i])

print(params)

seed1 = random.randint(1,65535)


sout = '_' + strFormat.format(oliLen1) + '_' + strFormat.format(oliRadius1) + '_' + strFormat.format(oliLen2) + '_' + strFormat.format(oliRadius2) + '_' + strFormat.format(-A11) + '_' + strFormat.format(-A22) + '_' + strFormat.format(-A12) + '_' + strFormat.format(B) + '_' + strFormat.format(SimID)

initialFname = "initialConditions/" + fileroot_out + sout + ".gsd"

#initializing the simulation with the gsd file
gpu = hoomd.device.GPU()
simulation = hoomd.Simulation(device=gpu, seed=seed1)
simulation.timestep = 0
simulation.create_state_from_gsd(filename=initialFname)


try:
    bondType = simulation.state.types['bond_types'][0];

except:
    print("No bonds");


snapshot = simulation.state.get_snapshot()
Lz = snapshot.configuration.box[2]

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
if( (oliLen1 > 1) or (oliLen2 > 1) ):
    harmonic = hoomd.md.bond.Harmonic()
    harmonic.params[bondType] = dict(k=k, r0=r0)


#set the integrator
integrator = hoomd.md.Integrator(dt=dt)


integrator.forces.append(dpd)
integrator.forces.append(sqd)

if( (oliLen1 > 1) or (oliLen2 > 1) ):
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
