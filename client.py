import numpy as np
import gc
import tracemalloc
import copy
import tensorflow as tf 
from tensorflow.data import AUTOTUNE
from cgan import CGAN_mnist
from cgan import CGAN_femnist
from cgan import CGAN_cifar10
import tensorflow.keras.backend as K
from models.initialize_model import create
from sklearn.model_selection import train_test_split

class Client:    

    def __init__(self,id_client,train_partition,test_partition,classes,dataset,model,loss,metrics,lr,
                                                    image_shape,latent_dim,number_classes,batch_size):   
        n='client'
        self.name=f'{n}_{id_client+1}'
        self.x=train_partition
        self.len=train_partition.cardinality()
        self.y=test_partition
        self.train=train_partition.shuffle(train_partition.cardinality()).batch(batch_size,drop_remainder=True)
        self.train=self.train.prefetch(buffer_size=AUTOTUNE)
        self.test=test_partition.batch(32)                
        self.classes=classes
        self.classifier=create(dataset,model,loss,metrics,lr,image_shape,number_classes) # includes build and compile     
        if dataset=="mnist":
            self.dis_model=CGAN_mnist(image_shape,latent_dim,number_classes).dis_model
        elif dataset=="femnist":
            self.dis_model=CGAN_femnist(image_shape,latent_dim,number_classes).dis_model
        else:
            self.dis_model=CGAN_cifar10(image_shape,latent_dim,number_classes).dis_model
        self.generated_data=None  
        self.test_acc=[]

    def num_batch(self,batch_size):
        if self.generated_data:
            l=self.train.cardinality()+int(self.generated_data.cardinality()/batch_size) # بچ بچ شده پس تعداد بچ رو می ده  train 
        else:
            l=self.train.cardinality()
        return l 

    def local_model_train(self,epochs,batch_size,verbose):     
        if self.generated_data:
            merge_data=self.train.unbatch()    
            merge_data=merge_data.concatenate(self.generated_data).batch(batch_size) 
            self.classifier.fit(merge_data,epochs=epochs,verbose=verbose)   
            del merge_data
            gc.collect()
        else:
            self.classifier.fit(self.train,epochs=epochs,verbose=verbose)
        K.clear_session()    
    
    def get_data(self,batch_size):
        if self.generated_data:
            merge_data=self.train.unbatch()    
            merge_data=merge_data.concatenate(self.generated_data).shuffle(self.generated_data.cardinality()).batch(
                                                          batch_size,drop_remainder=True)
            return merge_data   
        else:
            return self.train
        
    def train_disc(self,X_real,Y_real,X_fake,Y_fake,batch_size):       
        dis_loss_real,_=self.dis_model.train_on_batch([X_real,Y_real], tf.ones((batch_size,1)))
        dis_loss_fake,_=self.dis_model.train_on_batch([X_fake,Y_fake], tf.zeros((batch_size,1)))
        dis_loss=(dis_loss_fake+dis_loss_real)*0.5

    def send_to_edgeserver(self,edgeserver):                 
        edgeserver.models[edgeserver.cnames.index(self.name)].dis_model.set_weights(self.dis_model.get_weights())

    def refresh_client(self):                                               
        self.generated_data=None
        
    def m_compile(self,loss,optimizer,metrics):
        self.classifier.compile(loss=loss,optimizer=optimizer,metrics=metrics)

    def compute_test(self):      
        _,acc=self.classifier.evaluate(self.test)
        self.test_acc.append(np.round(acc,2))
