import numpy as np
from matplotlib import cm
from matplotlib.collections import EllipseCollection
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pickle
            
class CollisionEvent:
    """
    Object contains all information about a collision event
    which are necessary to update the velocity after the collision.
    For MD of hard spheres (with hard bond-length dimer interactions)
    in a rectangular simulation box with hard walls, there are only
    two distinct collision types:
    1) wall collision of particle i with vertical or horizontal wall
    2) external (or dimer bond-length) collision between particle i and j
    """
    def __init__(self, Type = 'wall or other', dt = 0, m_1 = 0, m_2 = 0, w_dir = 1):
        """
        Type = 'wall' or other
        dt = remaining time until collision
        m_1 = index of monomer
        m_2 = if inter-particle collision, index of second monomer
        w_dir = if wall collision, direction of wall
        (   w_dir = 0 if wall in x direction, i.e. vertical walls
            w_dir = 1 if wall in y direction, i.e. horizontal walls   )
        """
        self.Type = Type
        self.dt = dt
        self.m_1 = m_1
        self.m_2 = m_2  # only importent for interparticle collisions
        self.w_dir = w_dir # only important for wall collisions
        
        
    def __str__(self):
        if self.Type == 'wall':
            return "Event type: {:s}, dt: {:.8f}, p1 = {:d}, dim = {:d}".format(self.Type, self.dt, self.m_1, self.w_dir)
        else:
            return "Event type: {:s}, dt: {:.8f}, p1 = {:d}, p2 = {:d}".format(self.Type, self.dt, self.m_1, self.m_2)
        
    def __iter__(self):
        #not used anywhere! Just here because: interesting!
        #python2 self.__dict__.iteritems()
        for attr, value in self.__dict__.items():
            yield attr, value

class Monomers:
    """
    Class for event-driven molecular dynamics simulation of hard spheres:
    -Object contains all information about a two-dimensional monomer system
    of hard spheres confined in a rectengular box of hard walls.
    -A configuration is fully described by the simulation box and
    the particles positions, velocities, radiai, and masses.
    -Initial configuration of $N$ monomers has random positions (without overlap)
    and velocities of random orientation and norms satisfying
    $E = \sum_i^N m_i / 2 (v_i)^2 = N d/2 k_B T$, with $d$ being the dimension,
    $k_B$ the Boltzmann constant, and $T$ the temperature.
    -Class contains all functions for an event-driven molecular dynamics (MD)
    simulation. Essentail for inter-particle collsions is the mono_pair array,
    which book-keeps all combinations without repetition of particle-index
    pairs for inter-particles collisions, e.g. for $N = 3$ particles
    indices = 0, 1, 2
    mono_pair = [[0,1], [0,2], [1,2]]
    -Monomers can be initialized with individual radiai and density = mass/volume.
    For example:
    NumberOfMonomers = 7
    NumberMono_per_kind = [ 2, 5]
    Radiai_per_kind = [ 0.2, 0.5]
    Densities_per_kind = [ 2.2, 5.5]
    then monomers m_0, m_1 have radius 0.2 and mass 2.2*pi*0.2^2
    and monomers m_2,...,m_6 have radius 0.5 and mass 5.5*pi*0.5^2
    -The dictionary of this class can be saved in a pickle file at any time of
    the MD simulation. If the filename of this dictionary is passed to
    __init__ the simulation can be continued at any point in the future.
    IMPORTANT! If system is initialized from file, then other parameters
    of __init__ are ignored!
    """
    def __init__(self, NumberOfMonomers = 4, L_xMin = 0, L_xMax = 1, L_yMin = 0, L_yMax = 1, NumberMono_per_kind = np.array([4]), Radiai_per_kind = 0.5*np.ones(1), Densities_per_kind = np.ones(1), k_BT = 1, FilePath = './Configuration.p'):
        try:
            self.__dict__ = pickle.load( open( FilePath, "rb" ) )
            print("IMPORTANT! System is initialized from file %s, i.e. other input parameters of __init__ are ignored!" % FilePath)
        except:
            assert ( NumberOfMonomers > 0 )
            assert ( (L_xMin < L_xMax) and (L_yMin < L_yMax) )
            self.NM = NumberOfMonomers
            self.DIM = 2 #dimension of system
            self.BoxLimMin = np.array([ L_xMin, L_yMin])
            self.BoxLimMax = np.array([ L_xMax, L_yMax])
            self.mass = -1*np.ones( self.NM ) # Masses, negative mass means not initialized
            self.rad = -1*np.ones( self.NM ) # Radiai, negative radiai means not initialized
            self.pos = np.empty( (self.NM, self.DIM) ) # Positions, not initalized but desired shape
            self.vel = np.empty( (self.NM, self.DIM) ) # Velocities, not initalized but desired shape
            self.mono_pairs = np.array( [ (k,l) for k in range(self.NM) for l in range( k+1,self.NM ) ] )
            self.next_wall_coll = CollisionEvent( 'wall', 0, 0, 0, 0)
            self.next_mono_coll = CollisionEvent( 'mono', 0, 0, 0, 0)
        
            self.__assignRadiaiMassesVelocities(NumberMono_per_kind, Radiai_per_kind, Densities_per_kind, k_BT )
            self.__assignRandomMonoPos( )
            
    def __getitem__(self, index):
        #not used anywhere! Just here because: interesting!
        #to use call mono[i], where mono = Monomers(...)
        return self.pos[index], self.vel[index], self.rad[index], self.mass[index]
    
    def save_configuration(self, FilePath = 'MonomerConfiguration.p'):
        '''Saves configuration. Callable at any time during simulation.'''
        #print( self.__dict__ )
        pickle.dump( self.__dict__, open( FilePath, 'wb' ) )
    
    def __assignRadiaiMassesVelocities(self, NumberMono_per_kind = np.array([4]), Radiai_per_kind = 0.5*np.ones(1), Densities_per_kind = np.ones(1), k_BT = 1 ):
        '''
        Private function -> cannot be called outside class definition.
        '''
        '''initialize radiai and masses'''
        assert( sum(NumberMono_per_kind) == self.NM )
        assert( isinstance(Radiai_per_kind,np.ndarray) and (Radiai_per_kind.ndim == 1) )
        assert( (Radiai_per_kind.shape == NumberMono_per_kind.shape) and (Radiai_per_kind.shape == Densities_per_kind.shape))
        Tot_init = 0
        for i, nm_per_r_i in enumerate(NumberMono_per_kind):
            self.rad[ Tot_init : Tot_init+nm_per_r_i] = Radiai_per_kind[i]
            self.mass[ Tot_init : Tot_init+nm_per_r_i] = Densities_per_kind[i]*np.pi*Radiai_per_kind[i]**2
            Tot_init += nm_per_r_i
        
        '''initialize velocities'''
        assert( k_BT > 0 )
        # E_kin = sum_i m_i /2 v_i^2 = N * dim/2 k_BT https://en.wikipedia.org/wiki/Ideal_gas_law#Energy_associated_with_a_gas
        mass_velSq = self.DIM * k_BT
        RandomOrientation = np.random.uniform( 0, 2.*np.pi, self.NM )
        NormedVelocity = np.vstack( (np.cos(RandomOrientation), np.sin(RandomOrientation)) ).T
        
        self.vel = np.sqrt( mass_velSq / self.mass[:,None] ) * NormedVelocity
        
    
    def __assignRandomMonoPos(self, start_index = 0 ):
        '''
        Private function -> cannot be called outside class definition.
        Initialize random positions without overlap between monomers and wall.
        '''
        assert ( min(self.rad) > 0 )#otherwise not initialized
        p_new, infiniteLoopTest = start_index, 0
        BoxLength = self.BoxLimMax - self.BoxLimMin
        while p_new < self.NM and infiniteLoopTest < 10**4:
            infiniteLoopTest += 1
            self.pos[p_new] = np.random.uniform( self.BoxLimMin+self.rad[p_new,None], self.BoxLimMax-self.rad[p_new,None])
            p_old, NoOverlap = 0, True
            while NoOverlap and p_old < p_new:
                dist = np.linalg.norm(self.pos[p_old]-self.pos[p_new])
                if dist < (self.rad[p_old]+self.rad[p_new]): NoOverlap = False
                p_old += 1
            if NoOverlap:
                p_new += 1
                infiniteLoopTest = 0
        if p_new != self.NM:
            print('Failed to initialize all particle positions.\nIncrease simulation box size!')
            exit()
    
    def __str__(self, index = 'all'):
        if index == 'all':
            return "\nMonomers with:\nposition = " + str(self.pos) + "\nvelocity = " + str(self.vel) + "\nradius = " + str(self.rad) + "\nmass = " + str(self.mass)
            #return "Particle with r = [{:.2f}, {:.2f}], v = [{:.2f}, {:.2f}]".format(self.pos[0], self.pos[1], self.vel[0], self.vel[1])
        else:
            return "\nMonomer at index = " + str(index) + " with:\nposition = " + str(self.pos[index]) + "\nvelocity = " + str(self.vel[index]) + "\nradius = " + str(self.rad[index]) + "\nmass = " + str(self.mass[index])
    
    def snapshot(self, FileName = './snapshot.png', Title = '$t = $?'):
        '''
        Function saves a snapshot of current configuration,
        i.e. particle positions as circles of corresponding radius,
        velocities as arrows on particles,
        blue dashed lines for the hard walls of the simulation box.
        '''
        fig, ax = plt.subplots( dpi=300 )
        L_xMin, L_xMax = self.BoxLimMin[0], self.BoxLimMax[0]
        L_yMin, L_yMax = self.BoxLimMin[1], self.BoxLimMax[1]
        BorderGap = 0.1*(L_xMax - L_xMin)
        ax.set_xlim(L_xMin-BorderGap, L_xMax+BorderGap)
        ax.set_ylim(L_yMin-BorderGap, L_yMax+BorderGap)

        #--->plot hard walls (rectangle)
        rect = mpatches.Rectangle((L_xMin,L_yMin), L_xMax-L_xMin, L_yMax-L_yMin, linestyle='dashed', ec='gray', fc='None')
        ax.add_patch(rect)
        ax.set_aspect('equal')
        ax.set_xlabel('$x$ position')
        ax.set_ylabel('$y$ position')
        
        #--->plot monomer positions as circles
        MonomerColors = np.linspace( 0.2, 0.95, self.NM)
        Width, Hight, Angle = 2*self.rad, 2*self.rad, np.zeros( self.NM )
        collection = EllipseCollection( Width, Hight, Angle, units='x', offsets=self.pos,
                       transOffset=ax.transData, cmap='nipy_spectral', edgecolor = 'k')
        collection.set_array(MonomerColors)
        collection.set_clim(0, 1) # <--- we set the limit for the color code
        ax.add_collection(collection)

        #--->plot velocities as arrows
        ax.quiver( self.pos[:,0], self.pos[:,1], self.vel[:,0], self.vel[:,1] , units = 'dots', scale_units = 'dots')
        
        plt.title(Title)
        plt.savefig(FileName)
        plt.close()
    
    def Wall_time(self):
        '''
        -Function computes list of remaining time dt until future
        wall collision in x and y direction for every particle.
        Then, it stores collision parameters of the event with
        the smallest dt in the object next_wall_coll.
        -Meaning of future:
        if v > 0: solve BoxLimMax - rad = x + v * dt
        else:     solve BoxLimMin + rad = x + v * dt
        '''
        # if velocity > 0 -> collision with Max, else collision with min
        coll_dist_list = (self.BoxLimMax - self.rad[:,None]) * (self.vel > 0) + (self.BoxLimMin + self.rad[:,None]) * (self.vel < 0)
        
        # collision time in x and y direction
        dt_list = (coll_dist_list - self.pos) / self.vel
        
        # find particle with min collision time and the corresponding direction
        collision_disk, wall_direction = divmod( dt_list.argmin(), self.DIM )
        minCollTime = dt_list[ collision_disk ][ wall_direction ]
        
        self.next_wall_coll.dt = minCollTime
        self.next_wall_coll.m_1 = collision_disk
        #self.next_wall_coll.m_2 = not necessary
        self.next_wall_coll.w_dir = wall_direction
        
        
    def Mono_pair_time(self):
        '''
        - Function computes list of remaining time dt until
        future external collition between all combinations of
        monomer pairs without repetition. Then, it stores
        collision parameters of the event with
        the smallest dt in the object next_mono_coll.
        - If particles move away from each other, i.e.
        scal >= 0 or Omega < 0, then remaining dt is infinity.
        '''
        P_i = self.mono_pairs[:,0] # List of collision partner 1
        P_j = self.mono_pairs[:,1] # List of collision partner 2
        CollisionDist_sq = (self.rad[P_i] + self.rad[P_j])**2 # distance squared at which collision partners touch
        
        del_VecPos = self.pos[P_i] - self.pos[P_j] # r_i - r_j
        del_VecVel = self.vel[P_i] - self.vel[P_j] # v_i - v_j
        
        del_VecPos_sq = (del_VecPos**2).sum(1) #|dr|^2
        del_VecVel_sq = (del_VecVel**2).sum(1) #|dv|^2
        
        # solve quadratic equation of collision time
        # only minus solution is relevant
        InitDist = del_VecPos_sq - CollisionDist_sq # initial distance
        scal = 2 * (del_VecPos * del_VecVel).sum(1) # 2( dr \cdot dv )
        Omega = scal**2 - 4.* del_VecVel_sq * InitDist

        RequiredCondition = ((scal < 0) & (Omega > 0))
        DismissCondition = np.logical_not( RequiredCondition )
        #DismissCondition = ((scal >= 0) | (Omega <= 0)) # alternative to two previous lines
        
        scal[DismissCondition] = -np.inf
        Omega[DismissCondition] = 0
        del_t = ( scal + np.sqrt(Omega) ) / (-2*del_VecVel_sq)
        
        # find index of min collision time
        # return corresponding time and collision partners
        indexMinTime = np.argmin( del_t )
        minCollTime = del_t[indexMinTime]
        collision_disk_1, collision_disk_2 = self.mono_pairs[indexMinTime]
        
        self.next_mono_coll.dt = minCollTime
        self.next_mono_coll.m_1 = collision_disk_1
        self.next_mono_coll.m_2 = collision_disk_2
        #self.next_mono_coll.w_dir = not necessary
        
    def compute_next_event(self):
        '''
        Function gets event information about:
        1) next possible wall event
        2) next possible pair event
        Function returns event info of event with
        minimal time, i.e. the clostest in future.
        '''
        self.Wall_time()
        self.Mono_pair_time()
        
        if self.next_wall_coll.dt < self.next_mono_coll.dt:
            return self.next_wall_coll
        else:
            return self.next_mono_coll
            
    def compute_new_velocities(self, next_event):
        '''
        Function updates the velocities of the monomer(s)
        involved in collision event.
        Update depends on event type.
        Ellastic wall collisions in x direction reverse vx.
        Ellastic pair collisions follow: https://en.wikipedia.org/wiki/Elastic_collision#Two-dimensional_collision_with_two_moving_objects
        '''
        
        if next_event.Type == 'wall':
            mono_index = next_event.m_1
            wall_direction = next_event.w_dir
            
            self.vel[mono_index][wall_direction] *= -1.0
        else:
            mono_1 = next_event.m_1
            mono_2 = next_event.m_2
            
            del_pos = self.pos[mono_2] - self.pos[mono_1]
            abs_pos = np.linalg.norm(del_pos)
            del_pos /= abs_pos
            
            del_vel = self.vel[mono_2] - self.vel[mono_1]
            scal = 2./(self.mass[mono_1] + self.mass[mono_2]) * np.inner( del_vel, del_pos)
            
            self.vel[mono_1] += del_pos * scal * self.mass[mono_2]
            self.vel[[mono_2]] -= del_pos * scal * self.mass[mono_1]
        
class Dimers(Monomers):
    """
    --> Class derived from Monomers.
    --> See also comments in Monomer class.
    --> Class for event-driven molecular dynamics simulation of hard-sphere
    system with DIMERS (and monomers). Two hard-sphere monomers form a dimer,
    and experience additional ellastic collisions at the maximum
    bond length of the dimer. The bond length is defined in units of the
    minimal distance of the monomers, i.e. the sum of their radiai.
    -Next to the monomer information, the maximum dimer bond length is needed
    to fully describe one configuration.
    -Initial configuration of $N$ monomers has random positions without overlap
    and separation of dimer pairs is smaller than the bond length.
    Velocities have random orientations and norms that satisfy
    $E = \sum_i^N m_i / 2 (v_i)^2 = N d/2 k_B T$, with $d$ being the dimension,
    $k_B$ the Boltzmann constant, and $T$ the temperature.
    -Class contains all functions for an event-driven molecular dynamics (MD)
    simulation. Essentail for all inter-particle collsions is the mono_pair array
    (explained in momonmer class). Essentail for the ellastic bond collision
    of the dimers is the dimer_pair array which book-keeps index pairs of
    monomers that form a dimer. For example, for a system of $N = 10$ monomers
    and $M = 2$ dimers:
    monomer indices = 0, 1, 2, 3, ..., 9
    dimer_pair = [[0,2], [1,3]]
    -Monomers can be initialized with individual radiai and density = mass/volume.
    For example:
    NumberOfMonomers = 10
    NumberOfDimers = 2
    bond_length_scale = 1.2
    NumberMono_per_kind = [ 2, 2, 6]
    Radiai_per_kind = [ 0.2, 0.5, 0.1]
    Densities_per_kind = [ 2.2, 5.5, 1.1]
    then monomers m_0, m_1 have radius 0.2 and mass 2.2*pi*0.2^2
    and monomers m_2, m_3 have radius 0.5 and mass 5.5*pi*0.5^2
    and monomers m_4,..., m_9 have radius 0.1 and mass 1.1*pi*0.1^2
    dimer pairs are: (m_0, m_2), (m_1, m_3) with bond length 1.2*(0.2+0.5)
    see bond_length_scale and radiai
    -The dictionary of this class can be saved in a pickle file at any time of
    the MD simulation. If the filename of this dictionary is passed to
    __init__ the simulation can be continued at any point in the future.
    IMPORTANT! If system is initialized from file, then other parameters
    of __init__ are ignored!
    """
    def __init__(self, NumberOfMonomers = 4, NumberOfDimers = 2, L_xMin = 0, L_xMax = 1, L_yMin = 0, L_yMax = 1, NumberMono_per_kind = np.array([4]), Radiai_per_kind = 0.5*np.ones(1), Densities_per_kind = np.ones(1), bond_length_scale = 1.2, k_BT = 1, FilePath = './Configuration.p'):
        #if __init__() defined in derived class -> child does NOT inherit parent's __init__()
        try:
            self.__dict__ = pickle.load( open( FilePath, "rb" ) )
            print("IMPORTANT! System is initialized from file %s, i.e. other input parameters of __init__ are ignored!" % FilePath)
        except:
            assert ( (NumberOfDimers > 0) and (NumberOfMonomers >= 2*NumberOfDimers) )
            assert ( bond_length_scale > 1. ) # is in units of minimal distance of respective monomer pair
            Monomers.__init__(self, NumberOfMonomers, L_xMin, L_xMax, L_yMin, L_yMax, NumberMono_per_kind, Radiai_per_kind, Densities_per_kind, k_BT )
            self.ND = NumberOfDimers
            self.dimer_pairs = np.array([[k,self.ND+k] for k in range(self.ND)])#choice 2 -> more practical than [2*k,2*k+1]
            mono_i, mono_j = self.dimer_pairs[:,0], self.dimer_pairs[:,1]
            self.bond_length = bond_length_scale * ( self.rad[mono_i] + self.rad[mono_j] )
            self.next_dimer_coll = CollisionEvent( 'dimer', 0, 0, 0, 0)
            
            '''
            Positions initialized as pure monomer system by monomer __init__.
            ---> Reinitalize all monomer positions, but place dimer pairs first
            while respecting the maximal distance given by the bond length!
            '''
            self.__assignRandomDimerPos()
            self._Monomers__assignRandomMonoPos( 2*NumberOfDimers )
    
    def __assignRandomDimerPos(self):
        '''
        this is a private function -> cannot be called outside class definition
        initialize random positions without overlap between monomers and wall
        '''
        dimer_new_index, infiniteLoopTest = 0, 0
        BoxLength = self.BoxLimMax - self.BoxLimMin
        while dimer_new_index < self.ND and infiniteLoopTest < 10**4:
            infiniteLoopTest += 1
            mono_i, mono_j = dimer_new = self.dimer_pairs[dimer_new_index]
            min_dimer_dist = self.rad[mono_i] + self.rad[mono_j]
            max_dimer_dist = self.bond_length[dimer_new_index]
            
            #place mono_i inside simulation box
            self.pos[mono_i] = np.random.uniform( self.BoxLimMin+self.rad[mono_i,None], self.BoxLimMax-self.rad[mono_i,None])
            #place mono_j in correct distance and inside box
            randomAngle = np.random.rand()*2*np.pi
            randomDist = np.random.uniform(min_dimer_dist, max_dimer_dist)
            self.pos[mono_j] = self.pos[mono_i] + randomDist * np.array([np.cos(randomAngle), np.sin(randomAngle)])
            while np.less(self.pos[mono_j], self.BoxLimMin+self.rad[ mono_j, None ]).sum() or np.less(self.BoxLimMax-self.rad[ mono_j, None ], self.pos[mono_j]).sum():
                #loop ensures, that m_j inside box
                randomAngle = np.random.rand()*2*np.pi
                randomDist = np.random.uniform(min_dimer_dist, max_dimer_dist)
                self.pos[mono_j] = self.pos[mono_i] + randomDist * np.array([np.cos(randomAngle), np.sin(randomAngle)])
            
            NoOverlap = True
            for dimer_old in self.dimer_pairs[:dimer_new_index]:
                if not NoOverlap: break
                MonoIndexPermutations = np.array( np.meshgrid(dimer_old,dimer_new) ).T.reshape(-1,2)
                mono_1, mono_2 = MonoIndexPermutations[:,0], MonoIndexPermutations[:,1]
                minDist = self.rad[mono_1]+self.rad[mono_2]
                dist = np.linalg.norm(self.pos[mono_1]-self.pos[mono_2], axis = 1)
                
                if min(dist-minDist) < 0: NoOverlap = False
                    
            if NoOverlap:
                dimer_new_index += 1
                infiniteLoopTest = 0
        if dimer_new_index != self.ND:
            print('Failed to initialize all dimer positions.\nIncrease simulation box size!')
            exit()
        
        
    def __str__(self, index = 'all'):
        if index == 'all':
            return Monomers.__str__(self) + "\ndimer pairs = " + str(self.dimer_pairs) + "\nwith max bond length = " + str(self.bond_length)
        else:
            return "\nDimer pair " + str(index) + " consists of monomers = " + str(self.dimer_pairs[index]) + "\nwith max bond length = " + str(self.bond_length[index]) + Monomers.__str__(self, self.dimer_pairs[index][0]) + Monomers.__str__(self, self.dimer_pairs[index][1])

    def Dimer_pair_time(self):
        '''
        Function computes list of remaining time dt until
        future dimer bond collition for all dimer pairs.
        Then, it stores collision parameters of the event with
        the smallest dt in the object next_dimer_coll.
        '''
        P_i = self.dimer_pairs[:,0] # List of collision partner 1
        P_j = self.dimer_pairs[:,1] # List of collision partner 2
        CollisionDist_sq = self.bond_length**2# distance squared at which collision partners touch
        
        del_VecPos = self.pos[P_i] - self.pos[P_j] # r_i - r_j
        del_VecVel = self.vel[P_i] - self.vel[P_j] # v_i - v_j
        
        del_VecPos_sq = (del_VecPos**2).sum(1) #|dr|^2
        del_VecVel_sq = (del_VecVel**2).sum(1) #|dv|^2
        
        # solve quadratic equation of collision time
        # only minus solution is relevant
        InitDist = np.minimum( del_VecPos_sq - CollisionDist_sq, 0 )# initial distance
        #should be <= 0, but can be of order 10**-15 due to precission errors
        #np.minimum to ensure Omega >= 0
        scal = 2 * (del_VecPos * del_VecVel).sum(1) # 2( dr \cdot dv )
        Omega = scal**2 - 4.* del_VecVel_sq * InitDist # > 0, if (InitDist <= 0)
        
        del_t = ( -scal + np.sqrt(Omega) ) / (2*del_VecVel_sq)
        
        # find index of min collision time
        # return corresponding time and collision partners
        indexMinTime = np.argmin( del_t )
        minCollTime = del_t[indexMinTime]
        collision_disk_1, collision_disk_2 = self.dimer_pairs[indexMinTime]
        
        self.next_dimer_coll.dt = minCollTime
        self.next_dimer_coll.m_1 = collision_disk_1
        self.next_dimer_coll.m_2 = collision_disk_2
        #self.next_dimer_coll.w_dir = not necessary
        
        #return minCollTime, collision_disk_1, collision_disk_2
    
    def snapshot(self, FileName = './snapshot.png', Title = ''):
        '''
        ---> Overwriting snapshot(...) of Monomers class!
        Function saves a snapshot of current configuration,
        i.e. monomer positions as circles of corresponding radius,
        dimer bond length as back empty circles (on top of monomers)
        velocities as arrows on monomers,
        blue dashed lines for the hard walls of the simulation box.
        '''
        fig, ax = plt.subplots( dpi=300 )
        L_xMin, L_xMax = self.BoxLimMin[0], self.BoxLimMax[0]
        L_yMin, L_yMax = self.BoxLimMin[1], self.BoxLimMax[1]
        BorderGap = 0.1*(L_xMax - L_xMin)
        ax.set_xlim(L_xMin-BorderGap, L_xMax+BorderGap)
        ax.set_ylim(L_yMin-BorderGap, L_yMax+BorderGap)

        #--->plot hard walls (rectangle)
        rect = mpatches.Rectangle((L_xMin,L_yMin), L_xMax-L_xMin, L_yMax-L_yMin, linestyle='dashed', ec='gray', fc='None')
        ax.add_patch(rect)
        ax.set_aspect('equal')
        ax.set_xlabel('$x$ position')
        ax.set_ylabel('$y$ position')
        
        #--->plot monomer positions as circles
        COLORS = np.linspace(0.2,0.95,self.ND+1)
        MonomerColors = np.ones(self.NM)*COLORS[-1] #unique color for monomers
        # recolor each monomer pair with individual color
        MonomerColors[self.dimer_pairs[:,0]] = COLORS[:len(self.dimer_pairs)]
        MonomerColors[self.dimer_pairs[:,1]] = COLORS[:len(self.dimer_pairs)]

        #plot solid monomers
        Width, Hight, Angle = 2*self.rad, 2*self.rad, np.zeros( self.NM )
        collection = EllipseCollection( Width, Hight, Angle, units='x', offsets=self.pos,
                       transOffset=ax.transData, cmap='nipy_spectral', edgecolor = 'k')
        collection.set_array(MonomerColors)
        collection.set_clim(0, 1) # <--- we set the limit for the color code
        ax.add_collection(collection)
        
        #plot bond length of dimers as black cicles
        Width, Hight, Angle = self.bond_length, self.bond_length, np.zeros( self.ND )
        mono_i = self.dimer_pairs[:,0]
        mono_j = self.dimer_pairs[:,1]
        collection_m_i = EllipseCollection( Width, Hight, Angle, units='x', offsets=self.pos[mono_i],
                       transOffset=ax.transData, edgecolor = 'k', facecolor = 'None')
        collection_m_j = EllipseCollection( Width, Hight, Angle, units='x', offsets=self.pos[mono_j],
                       transOffset=ax.transData, edgecolor = 'k', facecolor = 'None')
        ax.add_collection(collection_m_i)
        ax.add_collection(collection_m_j)

        #--->plot velocities as arrows
        ax.quiver( self.pos[:,0], self.pos[:,1], self.vel[:,0], self.vel[:,1] , units = 'dots', scale_units = 'dots')
        
        plt.title(Title)
        plt.savefig( FileName)
        plt.close()

    
    def compute_next_event(self):
        '''
        function gets event info about:
        1) next possible wall event
        2) next possible monomer pair event
        3) next possible dimer event
        function returns event info of event with minimal time, i.e. the clostest in future
        '''
        self.Wall_time()
        self.Mono_pair_time()
        self.Dimer_pair_time()
        
        if self.next_wall_coll.dt < min(self.next_mono_coll.dt, self.next_dimer_coll.dt):
            return self.next_wall_coll
        elif (self.next_mono_coll.dt < self.next_dimer_coll.dt):
            return self.next_mono_coll
        else:
            return self.next_dimer_coll
        
