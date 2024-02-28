import os.path
import json
import scipy.misc
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import random 
# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path:str, label_path:str, batch_size:int, image_size:list, rotation:bool=False, mirroring:bool=False, shuffle:bool=False):
        # Define all members of your generator class object as global members here.

        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        #TODO: implement constructor
        self.current_batch_no = 0
        self.current_epoch_no = 0
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle 
        labeljson = open( self.label_path, "rb" )
        self.labels=json.load(labeljson)
        labeljson.close()
        image_files = os.listdir(self.file_path)
        self.image_names = [image_file[:-4] for image_file in image_files]
        if self.shuffle:
            random.shuffle(self.image_names)
    
    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases

        self.current_image_no = self.current_batch_no*self.batch_size
        
        previmg = None
        prevlabel = None

        ### for last batch change last image num so we do not overflow
        # 
        
        if len(self.image_names) >= (self.current_image_no + self.batch_size): 
            
            self.last_img_num = (self.current_image_no + self.batch_size)
        else:
        
            self.current_epoch_no = self.current_epoch_no + 1
            
            if self.shuffle:
                random.shuffle(self.image_names)
            self.current_batch_no = -1
            self.last_img_num = len(self.image_names)

        self.image_batch_names = self.image_names[self.current_image_no:self.last_img_num]
        
        #### if batch is smaller than batch_size add images from the front to it   ###
        
        if len(self.image_batch_names) < self.batch_size:
           
            num_of_less_images = self.batch_size - len(self.image_batch_names)
            self.image_batch_names =  self.image_batch_names+self.image_names[0:num_of_less_images]   
        
        for i,image_name in enumerate(self.image_batch_names):
        
            image = np.load(self.file_path+image_name+".npy")   
            image = self.augment(image)
            image = cv2.resize(image,tuple(self.image_size[:-1])) 
            label=self.labels[image_name]
            
            if i==0:

                previmg = image
                prevlabel = label

            elif i == 1:
                
                images = np.stack([previmg,image])
                labels =  np.stack([prevlabel,label])
            
            else:
            
                images = np.concatenate([images,np.expand_dims(image,axis=0)],axis=0)    
                labels = np.concatenate([labels,np.expand_dims(label,axis=0)],axis=0)
        
        self.current_batch_no = self.current_batch_no + 1 

        return images, labels
    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        if self.mirroring and random.random()>0.5:
            img=np.flip(img,axis=1)
        
        if self.rotation and random.random()>0.5:
            img = np.rot90(img)
        if self.rotation and random.random()>0.5:
            img = np.rot90(img)
        if self.rotation and random.random()>0.5:
            img = np.rot90(img)
                
        
        return img

    def current_epoch(self):
        # return the current epoch number
        return self.current_epoch_no

    def class_name(self, x):
        # This function returns the class name for a specific input
        
        return self.class_dict[x]
    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        images,labels = self.next()
        batch_sqrt = np.ceil(np.sqrt(self.batch_size))
        counter = 0 
        for image,label in zip(images,labels):
            counter = counter + 1  
            plt.subplot(batch_sqrt,batch_sqrt,counter)
            plt.title(self.class_name(label))
            plt.imshow(image)     
        plt.show()
if __name__ == "__main__":
    gen = ImageGenerator("./exercise_data/","./Labels.json", 60, [32, 32, 3], rotation=False, mirroring=False,shuffle=False)

    gen.show()