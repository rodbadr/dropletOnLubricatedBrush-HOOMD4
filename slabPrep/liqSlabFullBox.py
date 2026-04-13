import hoomd
import hoomd.md
import gsd
import gsd.hoomd
import numpy as np
import time
import random
import math
import sys

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

args = str(sys.argv[1])

paramInd = int(args)

paramSpace = np.genfromtxt('ParameterSpaceliqSlab.csv', delimiter=',')

params = paramSpace[paramInd-1,:]

strFormat = '{:g}'

fileroot_out = 'liqSlab'


buf = 0.075


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

mPoly = 1 # mass of polymer monomers

i=0
Noli = int(params[i])

i=i+1
oliLen = int(params[i])

### particle types are: A-grafted monomer, B-brush/gel monomers, C-oligomer monomers, D-fluid particles
### mdpd interaction parameters in order: [AA,AB,AC,AD,BB,BC,BD,CC,CD,DD]
i=i+1
Amm = float(params[i]); # monomer-monomer interaction
i=i+1
B = float(params[i]); # density dependent repulsion strength. Needs to be the same for all species. (See no-go theorem in many-body dissipative particle dynamics)

i=i+1
Lxin = float(params[i])
i=i+1
Lyin = float(params[i])
i=i+1
dens = float(params[i])

i=i+1
nsteps = int(params[i]);
nsteps = int(2e5);
i=i+1
Nmeas = int(params[i]);

i=i+1
animate = int(params[i]); # 0 - Do note print equilibration trajectory; 1 - print equilibration trajectory

meas_period = int(nsteps/Nmeas)

NmonOli = int(oliLen*Noli) # total number of oligomer monomers
NbondOli = int((oliLen-1)*Noli) # number of oligomer bonds
Ntot = int(NmonOli) # total number of particles in simulation box

animate = 0; # 0 - Do note print equilibration trajectory; 1 - print equilibration trajectory


if Lxin < oliLen*r0 + 5*rc:
    Lx = np.rint( oliLen*r0 ) + 5*rc

else:
    Lx = Lxin

Lx_final = Lxin
Ly = Lyin
Lz = Ntot/Lx_final/Ly/dens


seed1 = random.randint(1,65535)


sout = '_' + strFormat.format(oliLen) + '_' + strFormat.format(-Amm) + '_' + strFormat.format(B) + '_' + strFormat.format(Lxin) + '_' + strFormat.format(Lyin);


################################
### Initialize particle data ###
################################

positions = np.zeros((Ntot,3))
if(oliLen > 1):

    bond_pairs = np.zeros((NbondOli,2))

    xcoords2 = np.linspace(0,(oliLen-1)*r0,oliLen) # initial x coordinates of oligomers

### set positions etc...

for ix in range(Noli):

    tempx = random.random()*(Lx-oliLen) - Lx/2
    tempy = random.random()*Ly - Ly/2
    tempz = random.random()*Lz - Lz/2

    ind1 = ix*oliLen
    ind2 = (ix+1)*oliLen
    indBond = ix*(oliLen-1)
    ### position initialization

    if(oliLen > 1):
        positions[ind1:ind2,0] = xcoords2[:] + tempx
    else:
        positions[ind1,0] = tempx

    positions[ind1:ind2,1] = tempy
    positions[ind1:ind2,2] = tempz


    ### bond initialization
    if(oliLen > 1):
        bond_pairs[indBond:indBond+(oliLen-1),0] = np.linspace(ind1,ind2 - 2,oliLen-1);
        bond_pairs[indBond:indBond+(oliLen-1),1] = np.linspace(ind1+1,ind2 - 2 + 1,oliLen-1);



##################################
### Start HOOMD Initialization ###
##################################


#frame initialization
frame = gsd.hoomd.Frame()

#particle initialization
frame.particles.N = Ntot
frame.particles.types = ['A']
frame.particles.typeid = np.zeros(Ntot)
frame.particles.typeid[:] = 0
frame.particles.position = np.zeros((Ntot, 3))
frame.particles.position[:] = positions
frame.particles.mass = np.zeros(Ntot)
frame.particles.mass[:] = mPoly


#bond initialization

if(oliLen > 1):

    frame.bonds.N = NbondOli
    frame.bonds.types = ['Polymer']
    frame.bonds.typeid = [0]*NbondOli
    frame.bonds.group = np.zeros((NbondOli, 2))
    frame.bonds.group[:] = bond_pairs

frame.configuration.dimensions = 3
frame.configuration.box =  [Lx, Ly, Lz, 0, 0, 0]
frame.configuration.step = 0

#creating gsd file with the initialized frame


initialFname = "initialConditions/" + fileroot_out + "Full" + sout + ".gsd"

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
dpd.params[('A', 'A')] = dict(A = Amm, gamma = gamma)


#many-body interactions
sqd = hoomd.md.many_body.SquareDensity(nl, default_r_cut = rd)
sqd.params[('A', 'A')] = dict(A=0, B=B/alpha)


#harmonic force between bonds
if(oliLen > 1):
    harmonic = hoomd.md.bond.Harmonic()
    harmonic.params[bondType] = dict(k=k, r0=r0)

#set the integrator
integrator = hoomd.md.Integrator(dt=dt)


integrator.forces.append(dpd)
integrator.forces.append(sqd)

if(oliLen > 1):
    integrator.forces.append(harmonic)



### box resize methods

if (Lx > Lx_final):

    startResize = 0 # step at which to start resizing
    resize_steps = int(nsteps/2)

    ramp = hoomd.variant.Ramp(A=0, B=1, t_start=startResize, t_ramp=resize_steps)

    initial_box = simulation.state.box
    final_box = hoomd.Box(Lx_final,Ly,Lz,0,0,0)

    box_resize_trigger = hoomd.trigger.Periodic(1)

    box_resize = hoomd.update.BoxResize(box1=initial_box, box2=final_box, variant=ramp, trigger=box_resize_trigger)

    simulation.operations.updaters.append(box_resize)

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

hoomd.write.GSD.write(state=simulation.state, filename=initialFname, mode='wb')

print('Simulation ended')
