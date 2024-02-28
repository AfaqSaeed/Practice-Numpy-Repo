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
        
        onezero = np.concatenate([ones,zeros])
        zeroone = np.concatenate([zeros,ones])

        tile = np.concatenate([zeroone,onezero],1)
        
        self.output = np.tile(tile,(int(self.resolution/(2*self.tile_size)),int(self.resolution/(2*self.tile_size))))
        return self.output.copy()

    def show(self):
        plt.imshow(self.output,cmap="gray")
        plt.show()
class Circle:
    def __init__(self,resolution,radius,position):
        self.output = None
        self.resolution = resolution
        self.radius = radius
        self.position = position
    def draw(self):
        x , y = self.position
        
        xx = np.arange(0,self.resolution,1)
        yy = np.arange(0,self.resolution,1)
        
        xv,yv = np.meshgrid(xx,yy)
        self.output = (((xv-x)**2 +(yv-y)**2) <= (self.radius)**2).astype(np.bool_)
        return self.output.copy()
    def show(self):
        plt.imshow(self.output,cmap="gray")
        plt.show()
class Spectrum:
    def __init__(self,resolution):
        self.output = None
        self.resolution = resolution
 
    def draw(self):
        red1 = np.expand_dims(np.linspace(0,1.0,self.resolution),axis=0)
        red2 = np.expand_dims(np.ones((self.resolution)),axis=1)
        red = np.matmul(red2,red1)
        
        blue1=np.expand_dims(np.linspace(1,0,self.resolution),axis=0)
        blue2=np.expand_dims(np.ones((self.resolution)),axis=1)
        blue = np.dot(blue2,blue1)
        
        green1 = np.expand_dims(np.ones((self.resolution)),axis=0)
        green2 = np.expand_dims(np.linspace(0,1,self.resolution),axis=1)
        green  = np.dot(green2,green1)
        
        rgb = np.zeros((self.resolution,self.resolution,3))
        rgb[:,:,0] = red
        rgb[:,:,1] = green
        rgb[:,:,2] = blue
        self.output = rgb.copy()

        return self.output.copy()
    
    def show(self):
        plt.imshow(self.output,cmap="gray")
        plt.show()
