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

paramSpace = np.genfromtxt('ParameterSpaceOliBrushSetup.csv', delimiter=',') # read parameter file

params = paramSpace[paramInd-1,:] # select specific parameter set

dirBrush = "../bareBrushSetup/" # directory of the bare brush initial condition
filerootBrush = "bareBrush" # file name root for the bare brush initial condition
dirMelt = "../slabPrep/" # directory of the source liquid slab
# filerootMelt  = "liqSlabSecond" # file name root for the source liquid slab
filerootMelt  = "liqSlabFull" # file name root for the source liquid slab

fileroot_out = "oliBrush" # output file name root for the simulations

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

i=i+1
Noli= int(params[i]) # number of oligomers/lubricant chains
i=i+1
oliLen = int(params[i]) # oligomer length

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
NmonOli = int(oliLen*Noli); # total number of oligomer monomers
NbondsBrush = int((brushLen-1)*Nbrush1*Nbrush2) # number of brush bonds
NbondsOli = int((oliLen-1)*Noli) # number of oligomer bonds

Ntot = int(NmonBrush + NmonOli) # total number of particles
NbondsTot = int(NbondsBrush + NbondsOli) # total number of bonds

meas_period = int(nsteps/Nmeas) # period of dumping frames / making measurements

seed1 = random.randint(1,65535) # seed for random number generation


########################################
### Start Reading Initial Conditions ###
########################################

Lx = brushDist*Nbrush1
Ly = brushDist*Nbrush2

### read and setup brush particles

sinBrush = '_' + strFormat.format(brushLen) + '_' + strFormat.format(Nbrush1) + '_' + strFormat.format(Nbrush2) + '_' + strFormat.format(brushDist) + '_' + strFormat.format(SimID)

inputFnameBrush = "initialConditions/" + filerootBrush + sinBrush + ".gsd"

tempSnap = gsd.hoomd.open(name=f"{dirBrush}{inputFnameBrush}",mode='r')
snapBrush = tempSnap[0]

LzBrush = snapBrush.configuration.box[2];

brushHeight =  snapBrush.particles.position[0:NmonBrush,2].max()-snapBrush.particles.position[0:NmonBrush,2].min()# brush thickness in the initial condition for the substrate

positionsBrush = snapBrush.particles.position
bondsBrush = snapBrush.bonds.group
typeID_Brush = snapBrush.particles.typeid


### read and setup melt/lubricant particles

LzMelt = 0

if(Noli>0):

    # in case of non square box in x-y, x is the long direction
    sinMelt = '_' + strFormat.format(oliLen) + '_' + strFormat.format(-Amm) + '_' + strFormat.format(B) + '_' + strFormat.format(Lx) + '_' + strFormat.format(Lx)

    inputFnameMelt = "initialConditions/" + filerootMelt + sinMelt + ".gsd"

    tempSnap = gsd.hoomd.open(name=f"{dirMelt}{inputFnameMelt}",mode='r')
    snapMelt = tempSnap[0]

    NtotMelt = snapMelt.particles.N
    NoliMelt = int(NtotMelt/oliLen)

    LyMelt = snapMelt.configuration.box[1]
    LzMelt = snapMelt.configuration.box[2]

    initPositions = snapMelt.particles.position[:,:]

    comZslabMelt = computeCOM(initPositions[:,2] + LzMelt/2, LzMelt) - LzMelt/2 # center of mass of the slab

    initPositions[:,2] = (initPositions[:,2] - comZslabMelt + LzMelt/2) % LzMelt - LzMelt/2 # center the slab in z-direction


    # adjust the chains that cross the periodic boundary in z-direction
    for chain in range(NoliMelt):

        shift = 0

        for mon in range(oliLen-1):

            dz = initPositions[chain*oliLen+mon,2] - initPositions[chain*oliLen+mon+1,2]

            if (abs(dz)>LzMelt/2):
                shift = 1

        if(shift == 1):

            tempz = np.zeros(oliLen)
            tempz[:] = initPositions[chain*oliLen:(chain+1)*oliLen,2]
            tempz[tempz>0] = tempz[tempz>0] - LzMelt
            initPositions[chain*oliLen:(chain+1)*oliLen,2] = tempz[:];


    if (Noli > NoliMelt):


        N_duplicate = int(np.ceil(Noli/NoliMelt)) # number of times to duplicate the slab

        print("WARNING: Not enough particles in the slab, duplicating the slab " + str(N_duplicate) + " times")

        initPositionsDuplicated = np.zeros((N_duplicate * NtotMelt, 3))

        for i in range(N_duplicate):
            initPositionsDuplicated[i * NtotMelt:(i + 1) * NtotMelt, :] = initPositions[:]
            initPositionsDuplicated[i * NtotMelt:(i + 1) * NtotMelt, 2] += i * LzMelt

        initPositions = initPositionsDuplicated

        LzMelt = LzMelt * N_duplicate # new length of the slab
        NoliMelt = NoliMelt * N_duplicate # new number of chains in the slab
        NtotMelt = NtotMelt * N_duplicate # new number of particles in the slab


        initPositions[:,2] = (initPositions[:,2] + LzMelt/2) % LzMelt - LzMelt/2 # center the slab in z-direction


    bottomMelt = -LzMelt/2


    ### compute centers of mass of the chains

    comyArr = np.zeros(NoliMelt)
    comzArr = np.zeros(NoliMelt)

    for chain in range(NoliMelt):

        ind1 = chain*oliLen
        ind2 = (chain+1)*oliLen

        comyArr[chain] = computeCOM(initPositions[ind1:ind2,1] + LyMelt/2, LyMelt) - LyMelt/2
        comzArr[chain] = computeCOM(initPositions[ind1:ind2,2] + LzMelt/2, LzMelt) - LzMelt/2


    mask1 = ( comzArr >  bottomMelt )
    mask2 = ( np.abs(comyArr) <  Ly/2 ) # if the melt box is larger than the brush box, select only the chains in the brush box

    mask = mask1 & mask2

    mask = np.repeat(mask, oliLen) # repeat the mask to apply it to monomers instead of chains


    NtotTemp = int(mask.sum()) # number of particles remaining
    NoliTemp = int(NtotTemp/oliLen) # number of chains remaining

    positionsTemp = np.zeros((NtotTemp,3))
    positionsTemp[:] = initPositions[mask] # positions of selected particles

    ### compute centers of mass of the chains

    comzArr = np.zeros(NoliTemp)

    for chain in range(NoliTemp):

        ind1 = chain*oliLen
        ind2 = (chain+1)*oliLen

        comzArr[chain] = computeCOM(positionsTemp[ind1:ind2,2] + LzMelt/2, LzMelt) - LzMelt/2


    sortInd = np.argsort(comzArr) # sort chain COM by position

    mask = np.zeros(NoliTemp, dtype=bool)
    mask[sortInd[:Noli]] = True # select the lowest Noli chains from the slab
    mask = np.repeat(mask, oliLen) # repeat the mask to apply it to monomers instead of chains

    if(mask.sum() != NmonOli):
        print("ERROR: number of lubricant monomers does not correspond to selected one, possibly not enough particles in the box")
        exit()

    positionsMelt = np.zeros((NmonOli,3))
    positionsMelt[:] = positionsTemp[mask] # positions of selected particles
    positionsMelt[:,1] = (positionsMelt[:,1] + Ly/2) % Ly  - Ly/2 # apply periodic boundary in y-direction

    # adjust the chains that cross the periodic boundary in z-direction
    for chain in range(Noli):

        shift = 0

        for mon in range(oliLen-1):

            dz = positionsMelt[chain*oliLen+mon,2] - positionsMelt[chain*oliLen+mon+1,2]

            if (abs(dz)>LzMelt/2):
                shift = 1

        if(shift == 1):

            tempz = np.zeros(oliLen)
            tempz[:] = positionsMelt[chain*oliLen:(chain+1)*oliLen,2]
            tempz[tempz>0] = tempz[tempz>0] - LzMelt
            positionsMelt[chain*oliLen:(chain+1)*oliLen,2] = tempz[:];

    LzMelt = positionsMelt[:,2].max() - positionsMelt[:,2].min()

    if(oliLen > 1): # assign bond pairs to array ([Nbonds,2])


        bondsMelt = np.zeros((NbondsOli,2))

        for chain in range(Noli):

            ind1 = chain*oliLen + NmonBrush # shifted by the number of brush monomers to have proper indexing
            ind2 = (chain+1)*oliLen + NmonBrush

            indBond1 = chain*(oliLen-1) # shifted by the number of brush bonds to have proper indexing
            indBond2 = indBond1+(oliLen-1)

            bondsMelt[indBond1:indBond2,0] = np.linspace(ind1,ind2 - 2,oliLen-1);
            bondsMelt[indBond1:indBond2,1] = np.linspace(ind1+1,ind2 - 2 + 1,oliLen-1);





Lz = math.ceil(LzBrush + LzMelt + sig_b + 30)

WallPos = -Lz/2 + sig_b # wall Position

# shifting z coordinates to desired positions

positionsBrush[:,2] = positionsBrush[:,2] - positionsBrush[:,2].min() + WallPos


positions = np.zeros((Ntot, 3))
positions[:NmonBrush] = positionsBrush[:]

if Noli>0:
    positionsMelt[:,2] = positionsMelt[:,2] - positionsMelt[:,2].min() + WallPos + brushHeight - 2*rc
    positions[NmonBrush:] = positionsMelt[:]


bond_pairs = np.zeros((NbondsTot,2))

bond_pairs[:NbondsBrush] = bondsBrush

if ( (NbondsOli>0) and (Noli>0) ):
    bond_pairs[NbondsBrush:] = bondsMelt


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
frame.particles.typeid[:NmonBrush] = typeID_Brush
frame.particles.typeid[NmonBrush:] = 1
frame.particles.mass = np.ones(Ntot)*mPoly

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





sout = '_' + strFormat.format(brushLen) + '_' + strFormat.format(Nbrush1) + '_' + strFormat.format(Nbrush2) + '_' + strFormat.format(oliLen) +'_' + strFormat.format(Noli) + '_' + strFormat.format(brushDist) + '_' + strFormat.format(-Amm) + '_' + strFormat.format(B) + '_' + strFormat.format(SimID);

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
