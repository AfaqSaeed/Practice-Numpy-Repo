import matplotlib.pyplot as plt
import numpy as np 
class Checker:
    def __init__(self, resolution: int , tile_size: int  ):
        self.output = None
    
        self.resolution = resolution if resolution%(2*tile_size)==0 else int(resolution/(2*tile_size)*tile_size)
        self.tile_size = tile_size
    def draw(self):
        zeros = np.zeros((self.tile_size,self.tile_size),dtype=np.int32)
        ones  = np.ones((self.tile_size,self.tile_size),dtype=np.int32)
        # print(zeros)
        onezero = np.concatenate([ones,zeros])
        zeroone = np.concatenate([zeros,ones])

        tile = np.concatenate([zeroone,onezero],1)
        
        self.output = np.tile(tile,(int(self.resolution/(2*self.tile_size)),int(self.resolution/(2*self.tile_size)))).copy()
        return self.output

    def show(self):
        plt.imshow(self.output,cmap="gray")
        plt.show()
class Circle:
    output = None
    def __init__(self,resolution,radius,position):
        self.output = None
        self.resoltion = resolution
        self.radius = radius
        self.position = position
    def draw(self):
        x , y = self.position
        np.linspace(radius)
        np.meshgrid()
    def show(self):
        plt.imshow(self.output,cmap="gray")
        plt.show()