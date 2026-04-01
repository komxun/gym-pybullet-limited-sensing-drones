import pybullet as p
class SceneCreator:
    def __init__(self,
                 aviary, 
                 scene_number: int=1):
        self.SCENE = scene_number
        self.aviary  = aviary
    
    def create(self):
        if self.SCENE == 0:
            pass

        elif self.SCENE == 1:
            self.aviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf",
                        [0, 2, .5],
                        p.getQuaternionFromEuler([0,0,0]),
                        useFixedBase = True, 
                        physicsClientId=self.aviary.CLIENT
                        ))
            
            
            self.aviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [0.5, 3, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.aviary.CLIENT))
            
            self.aviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [-0.2, 5, 0+1], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.aviary.CLIENT))
            
            self.aviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [0.8, 7, 0], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.aviary.CLIENT))    
            
            self.aviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [1, 4, 0], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.aviary.CLIENT)) 
            
            self.aviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [2.5, 4, 1.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.aviary.CLIENT)) 
    
            self.aviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [3.5, 6, 2.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.aviary.CLIENT)) 
            self.aviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [4.5, 7, 1.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.aviary.CLIENT)) 
            self.aviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [3, 6, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.aviary.CLIENT)) 
            
            
            self.aviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [1.5, 8.5, 2.8], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.aviary.CLIENT)) 
            
            self.aviary.OBSTACLE_IDS.add(
                p.loadURDF("sphere2.urdf", 
                        [1.5, 9.5, 1], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.aviary.CLIENT)) 
            
            self.aviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [-0.5, 8, 1], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.aviary.CLIENT)) 
            
        elif self.SCENE == 2:
            
            # Moving obstacles
            id = p.loadURDF("sphere2.urdf",
                        [0, 2, .5],
                        p.getQuaternionFromEuler([0,0,0]),
                        useFixedBase = False, 
                        globalScaling = 1.2, 
                        physicsClientId=self.aviary.CLIENT
                        )
            self.aviary.OBSTACLE_IDS.add(id)
            p.changeDynamics(id, -1, mass=1, linearDamping=2)
            
            id = p.loadURDF("sphere2.urdf", 
                        [0, 5, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        globalScaling = 0.8,
                        useFixedBase = False, 
                        physicsClientId=self.aviary.CLIENT)
            self.aviary.OBSTACLE_IDS.add(id)
            p.changeDynamics(id, -1, mass=1, linearDamping=2)
            
            id = p.loadURDF("sphere2.urdf", 
                        [-0.5, 3.5, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        globalScaling = 1,
                        useFixedBase = False,
                        physicsClientId=self.aviary.CLIENT)
            self.aviary.OBSTACLE_IDS.add(id) 
            p.changeDynamics(id, -1, linearDamping=2, mass=1)
            
            # Static obstacle
            self.aviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [2, 5, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        globalScaling = 1.2, 
                        physicsClientId=self.aviary.CLIENT,
                        )) 
            self.aviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [-2, 5, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        globalScaling = 1.2, 
                        physicsClientId=self.aviary.CLIENT)) 
            
            self.aviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [2, 5, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        globalScaling = 1.2, 
                        physicsClientId=self.aviary.CLIENT)) 
            
            self.aviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [-3, 2, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        globalScaling = 1.2, 
                        physicsClientId=self.aviary.CLIENT)) 
            self.aviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [2, 2, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.aviary.CLIENT))
            self.aviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [-2.5, 3.5, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.aviary.CLIENT))   
            
            self.aviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [4.5, 7, 1.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.aviary.CLIENT)) 
            self.aviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [3, 6, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.aviary.CLIENT)) 
            
            
            self.aviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [2, 8.5, 2.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.aviary.CLIENT)) 
            
            
            self.aviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [0.8, 7, 0], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.aviary.CLIENT))    
            
            self.aviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [2, 3.5, 0], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.aviary.CLIENT)) 
            
            self.aviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [2.5, 4, 1.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.aviary.CLIENT)) 
            self.aviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [3.5, 4, 1.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.aviary.CLIENT)) 
        elif scene == 3:
            
            # Moving obstacles
            id1 = p.loadURDF("sphere2.urdf",
                        [0, 2, .5],
                        p.getQuaternionFromEuler([0,0,0]),
                        useFixedBase = False, 
                        globalScaling = 0.9, 
                        physicsClientId=self.aviary.CLIENT
                        )
            self.aviary.OBSTACLE_IDS.add(id1)
            p.changeDynamics(id1, -1, mass=1, linearDamping=2)
            
            id2 = p.loadURDF("sphere2.urdf", 
                        [0, 10, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        globalScaling = 0.8,
                        useFixedBase = False, 
                        physicsClientId=self.aviary.CLIENT)
            self.aviary.OBSTACLE_IDS.add(id2)
            p.changeDynamics(id2, -1, mass=1, linearDamping=2)
            
            id3 = p.loadURDF("sphere2.urdf", 
                        [-0.5, 7, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        globalScaling = 1,
                        useFixedBase = False,
                        physicsClientId=self.aviary.CLIENT)
            self.aviary.OBSTACLE_IDS.add(id3) 
            p.changeDynamics(id3, -1, linearDamping=2, mass=1)

            self.aviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [-3, 2, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        globalScaling = 1.2, 
                        physicsClientId=self.aviary.CLIENT)) 
            self.aviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [2, 2, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.aviary.CLIENT))
            
            # Static obstacle
            self.aviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [2, 10, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        globalScaling = 1.2, 
                        physicsClientId=self.aviary.CLIENT,
                        )) 
            self.aviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [-2, 10, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        globalScaling = 1.2, 
                        physicsClientId=self.aviary.CLIENT)) 
            
            self.aviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [2, 10, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        globalScaling = 1.2, 
                        physicsClientId=self.aviary.CLIENT)) 
            
            
            self.aviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [-2.5, 7, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.aviary.CLIENT))  
            
            # self.aviary.OBSTACLE_IDS.add(
            #     p.loadURDF("cube.urdf", 
            #             [0.5, 3, 0], 
            #             p.getQuaternionFromEuler([0,0,0]), 
            #             useFixedBase = True, 
            #             physicsClientId=self.aviary.CLIENT))
            
            self.aviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [2.8, 7, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.aviary.CLIENT))    
            
            self.aviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [4.5, 10, 1.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.aviary.CLIENT)) 
            self.aviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [3, 6, 0.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.aviary.CLIENT)) 
            
            
            self.aviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [2, 10.5, 2.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.aviary.CLIENT)) 
            
            
            self.aviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [0.8, 10, 0], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.aviary.CLIENT))    
            
            self.aviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [2, 3.5, 0], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.aviary.CLIENT)) 
            
            self.aviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [2.5, 7, 1.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.aviary.CLIENT)) 
            self.aviary.OBSTACLE_IDS.add(
                p.loadURDF("cube.urdf", 
                        [3.5, 7, 1.5], 
                        p.getQuaternionFromEuler([0,0,0]), 
                        useFixedBase = True, 
                        physicsClientId=self.aviary.CLIENT))
            
        # self._getObstaclesData()