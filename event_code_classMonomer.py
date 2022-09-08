import ParticleClass as pc
import numpy as np

import os
from matplotlib import cm
from matplotlib.collections import EllipseCollection
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

np.random.seed(999)
Snapshot_output_dir = './SnapshotsMonomers'
if not os.path.exists(Snapshot_output_dir): os.makedirs(Snapshot_output_dir)

Conf_output_dir = './ConfsMonomers'
Path_ToConfiguration = Conf_output_dir+'/FinalMonomerConf.p'
if os.path.isfile( Path_ToConfiguration ):
    '''Initialize system from existing file'''
    mols = pc.Monomers( FilePath = Path_ToConfiguration )
else:
    '''Initialize system with following parameters'''
    if not os.path.exists(Conf_output_dir): os.makedirs(Conf_output_dir)
    NumberOfMonomers = 5
    L_xMin, L_xMax = 0, 7
    L_yMin, L_yMax = 0, 7
    NumberMono_per_kind = np.array([ 2, 3])
    Radiai_per_kind = np.array([ 0.5, 0.7])
    Densities_per_kind = np.ones(len(NumberMono_per_kind))
    k_BT = 100

    mols = pc.Monomers(NumberOfMonomers, L_xMin, L_xMax, L_yMin, L_yMax, NumberMono_per_kind, Radiai_per_kind, Densities_per_kind, k_BT )

mols.snapshot( FileName = Snapshot_output_dir+'/InitialConf.png', Title = '$t = 0$')
#we could initialize next_event, but it's not necessary
#next_event = pc.CollisionEvent( Type = 'wall or other, to be determined', dt = 0, m_1 = 0, m_2 = 0, w_dir = 0)

'''define parameters for MD simulation'''
t = 0.0
dt = 0.01     # dt=0 corresponds to event-to-event animation
NumberOfFrames = 160
next_event = mols.compute_next_event()
def MolecularDynamicsLoop( frame ):
    '''
    The MD loop including update of frame for animation.
    '''
    global t, mols, next_event
    if dt: # dt=0 corresponds to event-to-event animation
        t_next_frame = t + dt
    else:
        t_next_frame = t + next_event.dt
    while t + next_event.dt <= t_next_frame:
        t += next_event.dt
        mols.pos += mols.vel * next_event.dt
        # we can save additional snapshots for debugging -> slows down real-time animation
        #mols.snapshot( FileName = Snapshot_output_dir + '/Conf_t%.8f_0.png' % t, Title = '$t = %.8f$' % t)
        mols.compute_new_velocities( next_event )
        #mols.snapshot( FileName = Snapshot_output_dir + '/Conf_t%.8f_1.png' % t, Title = '$t = %.8f$' % t)
        next_event = mols.compute_next_event()

    remain_t = t_next_frame - t #equals dt if no event between frames
    mols.pos += mols.vel * remain_t
    t += remain_t
    next_event.dt -= remain_t

    plt.title( '$t = %.4f$, remaining frames = %d' % (t, NumberOfFrames-(frame+1)) )
    collection.set_offsets( mols.pos )
    return collection



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


# plotting all monomers as solid circles of individual color
MonomerColors = np.linspace(0.2,0.95,mols.NM)
Width, Hight, Angle = 2*mols.rad, 2*mols.rad, np.zeros(mols.NM)
collection = EllipseCollection(Width, Hight, Angle, units='x', offsets=mols.pos,
                       transOffset=ax.transData, cmap='nipy_spectral', edgecolor = 'k')
collection.set_array(MonomerColors)
collection.set_clim(0, 1) # <--- we set the limit for the color code
ax.add_collection(collection)

'''Create the animation, i.e. looping NumberOfFrames over the update function'''
Delay_in_ms = 33.3 #dely between images/frames for plt.show()
ani = FuncAnimation(fig, MolecularDynamicsLoop, frames=NumberOfFrames, interval=Delay_in_ms, blit=False, repeat=False)
plt.show()
#Frames_Per_s=24
#ani.save('./Monomer3Simulation.gif', fps=Frames_Per_s)
#ani.save('basic_animation.mp4', fps=Frames_Per_s, extra_args=['-vcodec', 'libx264'])

'''Save the final configuration and make a snapshot.'''
mols.save_configuration(Path_ToConfiguration)
mols.snapshot( FileName = Snapshot_output_dir + '/FinalConf.png', Title = '$t = %.4f$' % t)
