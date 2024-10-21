from tensorflow.keras.datasets.mnist import load_data
import numpy as np 
import os
from matplotlib import pyplot
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.utils import plot_model,to_categorical
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Multiply
from tensorflow.keras.models import Model


class CGAN_mnist:
    def __init__(self,image_shape,latent_dim,number_classes):         
        self.dis_model=self.define_discriminator(image_shape,number_classes)
        self.gen_model=self.define_generator(latent_dim,number_classes)
        self.cgan_model=self.define_cgan()
    
    def define_discriminator(self,image_shape,number_classes ):   
        input_label=Input(shape=(number_classes,))      
        l=Dense(image_shape[0]*image_shape[1])(input_label)
        l=Reshape((image_shape[0],image_shape[1],image_shape[2]))(l)
        input_image=Input(shape=image_shape)
        merge=concatenate([input_image,l])
        i=Conv2D(64, (3,3), strides=(2,2), padding='same')(merge)
        i=LeakyReLU(alpha=0.2)(i)
        i=Dropout(0.4)(i)
        i=Conv2D(64, (3,3), strides=(2,2), padding='same')(i)
        i=LeakyReLU(alpha=0.2)(i)
        i=Dropout(0.4)(i)
        i=Flatten()(i)
        out_layer=Dense(1, activation='sigmoid')(i)
        model=Model([input_image,input_label], out_layer)
        opt=Adam(learning_rate=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    def define_generator(self,latent_dim,number_classes):
        input_label=Input(shape=(number_classes,))
        input_latent=Input(shape=(latent_dim,))
        merge=concatenate([input_latent,input_label])
        d=Dense(128 *7*7)(merge)
        d=LeakyReLU(alpha=0.2)(d)
        d=Reshape((7,7,128))(d)
        d=Conv2DTranspose(128, (4,4),strides=(2,2),padding='same')(d)
        d=LeakyReLU(alpha=0.2)(d)
        d=Conv2DTranspose(128, (4,4),strides=(2,2),padding='same')(d)
        d=LeakyReLU(alpha=0.2)(d)
        out_layer=Conv2D(1, (7,7),activation='sigmoid',padding='same')(d)    
        model=Model([input_latent, input_label],out_layer)
        return model

    def define_cgan(self):
        self.dis_model.trainable=False
        gen_noise,gen_label=self.gen_model.input
        gen_output=self.gen_model.output
        output=self.dis_model([gen_output,gen_label])
        model=Model([gen_noise,gen_label],output)
        opt=Adam(learning_rate=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model

class CGAN_femnist:
    def __init__(self,image_shape,latent_dim,number_classes):       
        self.dis_model=self.define_discriminator(image_shape,number_classes)
        self.gen_model=self.define_generator(latent_dim,number_classes)
        self.cgan_model=self.define_cgan()
    
    def define_discriminator(self,image_shape,number_classes ):   
        input_label=Input(shape=(number_classes,))      
        l=Dense(image_shape[0]*image_shape[1])(input_label)
        l=Reshape((image_shape[0],image_shape[1],image_shape[2]))(l)
        input_image=Input(shape=image_shape)
        merge=concatenate([input_image,l])
        i=Conv2D(64, (3,3), strides=(2,2), padding='same')(merge)
        i=LeakyReLU(alpha=0.2)(i)
        i=Dropout(0.4)(i)
        i=Conv2D(64, (3,3), strides=(2,2), padding='same')(i)
        i=LeakyReLU(alpha=0.2)(i)
        i=Dropout(0.4)(i)
        i=Flatten()(i)
        i=Dense(500)(i)
        i=Dense(30)(i)
        out_layer=Dense(1, activation='sigmoid')(i)
        model=Model([input_image,input_label], out_layer)
        opt=Adam(learning_rate=0.0005, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    def define_generator(self,latent_dim,number_classes):
        input_label=Input(shape=(number_classes,))
        input_latent=Input(shape=(latent_dim,))
        merge=concatenate([input_latent,input_label])
        d=Dense(128 *7*7)(merge)
        d=LeakyReLU(alpha=0.2)(d)
        d=Reshape((7,7,128))(d)
        d=Conv2DTranspose(128, (4,4),strides=(2,2),padding='same')(d)
        d=LeakyReLU(alpha=0.2)(d)
        d=Conv2DTranspose(128, (4,4),strides=(2,2),padding='same')(d)
        d=LeakyReLU(alpha=0.2)(d)
        out_layer=Conv2D(1, (7,7),activation='sigmoid',padding='same')(d)    
        model=Model([input_latent, input_label],out_layer)
        return model

    def define_cgan(self):
        self.dis_model.trainable=False
        gen_noise,gen_label=self.gen_model.input
        gen_output=self.gen_model.output
        output=self.dis_model([gen_output,gen_label])
        model=Model([gen_noise,gen_label],output)
        opt=Adam(learning_rate=0.0005, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model
    
class CGAN_cifar10:
    def __init__(self,image_shape,latent_dim,number_classes):       
        self.dis_model=self.define_discriminator(image_shape,number_classes)
        self.gen_model=self.define_generator(latent_dim,number_classes)
        self.cgan_model=self.define_cgan()
    
    def define_discriminator(self,image_shape,number_classes ):   
        input_label=Input(shape=(number_classes,))      
        l=Dense(image_shape[0]*image_shape[1]*image_shape[2])(input_label)
        l=Reshape((image_shape[0],image_shape[1],image_shape[2]))(l)
        input_image=Input(shape=image_shape)
        merge=concatenate([input_image,l])
        i=Conv2D(64, (3,3), strides=(2,2), padding='same')(merge)
        i=LeakyReLU(alpha=0.2)(i)
        i=Conv2D(128, (3,3), strides=(2,2), padding='same')(i)
        i=LeakyReLU(alpha=0.2)(i)
        i=Conv2D(256, (3,3), strides=(2,2), padding='same')(i)
        i=LeakyReLU(alpha=0.2)(i)
        i=Flatten()(i)
        i=Dropout(0.4)(i)
        out_layer=Dense(1, activation='sigmoid')(i)
        model=Model([input_image,input_label], out_layer)
        opt=Adam(learning_rate=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    def define_generator(self,latent_dim,number_classes):
        input_label=Input(shape=(number_classes,))
        input_latent=Input(shape=(latent_dim,))
        merge=concatenate([input_latent,input_label])
        d=Dense(256 *4*4)(merge)
        d=LeakyReLU(alpha=0.2)(d)
        d=Reshape((4,4,256))(d)
        # upsample to 8x8
        d=Conv2DTranspose(128, (4,4),strides=(2,2),padding='same')(d)
        d=LeakyReLU(alpha=0.2)(d)
        # upsample to 16x16
        d=Conv2DTranspose(128, (4,4),strides=(2,2),padding='same')(d)
        d=LeakyReLU(alpha=0.2)(d)
        # upsample to 32x32
        d=Conv2DTranspose(128, (4,4),strides=(2,2),padding='same')(d)
        d=LeakyReLU(alpha=0.2)(d)
        out_layer=Conv2D(3, (3,3),activation='sigmoid',padding='same')(d)     
        model=Model([input_latent, input_label],out_layer)
        return model

    def define_cgan(self):
        self.dis_model.trainable=False
        gen_noise,gen_label=self.gen_model.input
        gen_output=self.gen_model.output
        output=self.dis_model([gen_output,gen_label])
        model=Model([gen_noise,gen_label],output)
        opt=Adam(learning_rate=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model
