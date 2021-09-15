import ParticleClass as pc
import numpy as np

import os
from matplotlib import cm
from matplotlib.collections import EllipseCollection
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

np.random.seed(999)
Snapshot_output_dir = './SnapshotsDimers'
if not os.path.exists(Snapshot_output_dir): os.makedirs(Snapshot_output_dir)

Conf_output_dir = './ConfsDimers'
Path_ToConfiguration = Conf_output_dir+'/FinalDimerConf.p'
if os.path.isfile( Path_ToConfiguration ):
    '''Initialize system from existing file'''
    mols = pc.Dimers( FilePath = Path_ToConfiguration )
else:
    '''Initialize system with following parameters'''
    if not os.path.exists(Conf_output_dir): os.makedirs(Conf_output_dir)
    NumberOfMonomers, NumberOfDimers = 2, 1
    L_xMin, L_xMax = 0, 5
    L_yMin, L_yMax = 0, 5
    NumberMono_per_kind = np.array([ NumberOfMonomers ])
    Radiai_per_kind = np.array([ 0.5])
    Densities_per_kind = np.ones(len(NumberMono_per_kind))
    bond_length_scale = 1.5
    k_BT = 5
    # call constructor, which should initialize the configuration
    mols = pc.Dimers(NumberOfMonomers, NumberOfDimers, L_xMin, L_xMax, L_yMin, L_yMax, NumberMono_per_kind, Radiai_per_kind, Densities_per_kind, bond_length_scale, k_BT )
    
mols.snapshot( FileName = Snapshot_output_dir+'/InitialConf.png', Title = '$t = 0$')
#we could initialize next_event, but it's not necessary
#next_event = pc.CollisionEvent( Type = 'wall or other, to be determined', dt = 0, m_1 = 0, m_2 = 0, w_dir = 0)

'''define parameters for MD simulation'''
t = 0.0
dt = 0.02
NumberOfFrames = 100
next_event = mols.compute_next_event()
def MolecularDynamicsLoop( frame ):
    '''
    The MD loop including update of frame for animation.
    '''
    global t, mols, next_event
    
    mols.pos += mols.vel * dt
    t += dt
    #--> write your event-driven loop
    #--> we want a simulation in constant time steps dt
    
    # we can save additional snapshots for debugging -> slows down real-time animation
    #mols.snapshot( FileName = Snapshot_output_dir + '/Conf_t%.8f_0.png' % t, Title = '$t = %.8f$' % t)
    
    plt.title( '$t = %.4f$, remaining frames = %d' % (t, NumberOfFrames-(frame+1)) )
    collection.set_offsets( mols.pos )
    collection_m_i.set_offsets( mols.pos[mono_i] )
    collection_m_j.set_offsets( mols.pos[mono_j] )
    return collection, collection_m_i, collection_m_j


'''We define and initalize the plot for the animation'''
fig, ax = plt.subplots()
L_xMin, L_yMin = mols.BoxLimMin #not defined if initalized by file
L_xMax, L_yMax = mols.BoxLimMax #not defined if initalized by file
BorderGap = 0.1*(L_xMax - L_xMin)
ax.set_xlim(L_xMin-BorderGap, L_xMax+BorderGap)
ax.set_ylim(L_yMin-BorderGap, L_yMax+BorderGap)
ax.set_aspect('equal')

# confining hard walls plotted as dashed lines
rect = mpatches.Rectangle((L_xMin,L_yMin), L_xMax-L_xMin, L_yMax-L_yMin, linestyle='dashed', ec='gray', fc='None')
ax.add_patch(rect)


COLORS = np.linspace(0.2,0.95,mols.ND+1)
MonomerColors = np.ones(mols.NM)*COLORS[-1] #unique color for monomers
# recolor each dimer pair with individual color
MonomerColors[mols.dimer_pairs[:,0]] = COLORS[:len(mols.dimer_pairs)]
MonomerColors[mols.dimer_pairs[:,1]] = COLORS[:len(mols.dimer_pairs)]
# plotting all monomers as solid circles
Width, Hight, Angle = 2*mols.rad, 2*mols.rad, np.zeros(mols.NM)
collection = EllipseCollection(Width, Hight, Angle, units='x', offsets=mols.pos,
                       transOffset=ax.transData, cmap='nipy_spectral', edgecolor = 'k')
collection.set_array(MonomerColors)
collection.set_clim(0, 1) # <--- we set the limit for the color code
ax.add_collection(collection)

# plotting empty black circles indicating the maximum bond length of dimers
Width, Hight, Angle = mols.bond_length, mols.bond_length, np.zeros( mols.ND )
mono_i = mols.dimer_pairs[:,0]
mono_j = mols.dimer_pairs[:,1]
collection_m_i = EllipseCollection( Width, Hight, Angle, units='x', offsets=mols.pos[mono_i],
               transOffset=ax.transData, edgecolor = 'k', facecolor = 'None')
collection_m_j = EllipseCollection( Width, Hight, Angle, units='x', offsets=mols.pos[mono_j],
               transOffset=ax.transData, edgecolor = 'k', facecolor = 'None')
ax.add_collection(collection_m_i)
ax.add_collection(collection_m_j)

'''Create the animation, i.e. looping NumberOfFrames over the update function'''
Delay_in_ms = 33.3 #dely between images/frames for plt.show()
ani = FuncAnimation(fig, MolecularDynamicsLoop, frames=NumberOfFrames, interval=Delay_in_ms, blit=False, repeat=False)
plt.show()

'''Save the final configuration and make a snapshot.'''
#write the function to save the final configuration
mols.save_configuration(Path_ToConfiguration)
mols.snapshot( FileName = Snapshot_output_dir + '/FinalConf.png', Title = '$t = %.4f$' % t)
