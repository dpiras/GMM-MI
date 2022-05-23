#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 21:22:06 2021

@author: lillian

Quantifying alignment of latent space with data generation factors.
Producing distributions of cosine similarity between data generation factors and latent variables.
"""
import time
from datetime import timedelta
import numpy as np
import tensorflow as tf

def get_samples(factor, model, dataset, latent_dim, repeats_per_image_batch=100):    
    label = 'label_' + factor
    factors_dict = shapes3d_factors_dict()
    
    mean_list = []
    logvar_list = []
    latents_list = []
    
    to_iterate = range(factors_dict[factor])
    
    for class_fixed in to_iterate:  
        sub_ds = dataset.shuffle(buffer_size=1000)
        filtered_ds = sub_ds.filter(lambda x: x[label] == class_fixed).map(preprocess).batch(1000)
        image_batch = next(iter(filtered_ds))
        
        mean, logvar = model.encode(image_batch)
        mean_list.append(mean)
        logvar_list.append(logvar)        
            
        latents = []
            
        for i in range(repeats_per_image_batch):
            z_sample = model.reparameterize(mean, logvar)
            latents.append(z_sample.numpy())
        
        latents_sampled = np.stack(latents)
        latents_sampled = latents_sampled.reshape((-1, latent_dim))
        latents_list.append(latents_sampled)
        
    latents_list_arr = np.stack(latents_list)
    latents_list_arr = np.split(latents_list_arr, latent_dim, axis=-1)
    
    return mean_list, logvar_list, latents_list_arr  

def shapes3d_factors_dict():
    dict_ = {'floor_hue': 10,
             'object_hue': 10,
             'orientation': 15,
             'scale': 8,
             'shape': 4,
             'wall_hue': 10}
    return dict_

def preprocess(features):
    image = tf.image.convert_image_dtype(features['image'], tf.float32)
    return image