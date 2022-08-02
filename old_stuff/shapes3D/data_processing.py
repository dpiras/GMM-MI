#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 23:34:58 2021

@author: lillian
"""
import tensorflow as tf
import tensorflow_datasets as tfds



class DataPreparation:
    def __init__(self, dataset_name='shapes3d', data_dir='/home/nguo/Interpretability_Methods/data/', 
                 split_method='train_val_test', split_at=[60, 80], batch_size=32, seed=330):
        self.seed = seed
        self.batch_size = batch_size
        
        if split_method == 'train_val_test':
            train_ds, val_ds, test_ds, self.metadata =  self.train_val_test_split(dataset_name, data_dir, split_at=split_at)
            self.train_ds = self.prepare_single_dataset(train_ds)
            self.val_ds = self.prepare_single_dataset(val_ds)
            self.test_ds = self.prepare_single_dataset(test_ds)
        
        elif split_method == 'cross_validation':
            train_ds_list, val_ds_list, test_ds_list, self.holdout_ds = self.cross_validation(dataset_name, data_dir)
            self.train_ds = self.prepare_datasets(train_ds_list)
            self.val_ds = self.prepare_datasets(val_ds_list)
            self.test_ds = self.prepare_datasets(test_ds_list)
            print('5-fold split on :90% training data. holdout_ds consists of 90%: of data.')                     
    
    def cross_validation(self, dataset_name, data_dir):
        tests_ds = tfds.load(dataset_name, data_dir=data_dir,
                             download=False,
                             split=[
                                 f'train[{k}%:{k+18}%]' for k in range(0, 90, 18)    
                                 ], shuffle_files=False)
        vals_ds = tfds.load(dataset_name, data_dir=data_dir,
                            download=False,
                            split=[
                                f'train[{k+18}%:{k+36}%]' for k in [0, 18, 36, 54, -18]
                                ], shuffle_files=False)
        trains_ds = tfds.load(dataset_name, data_dir=data_dir,
                              download=False,
                              split=[
                                  f'train[:{k}%]+train[{k+36}%:90%]' for k in [0, 18, 36, 54]] + ['train[18%:72%]'], 
                              shuffle_files=False)
        holdout_ds = tfds.load(dataset_name, data_dir=data_dir,
                               download=False, split=['train[90%:]'], shuffle_files=False)
        return tests_ds, vals_ds, trains_ds, holdout_ds
    
    def prepare_single_dataset(self, dataset):
        AUTOTUNE = tf.data.experimental.AUTOTUNE #tune No.elements to prefetch during run
        dataset = dataset.map(self.preprocess, num_parallel_calls=AUTOTUNE)        
        dataset = self.configure_for_performance(dataset)
        return dataset
    
    def prepare_datasets(self, dataset_list):
        for i, ds in enumerate(dataset_list):
            dataset_list[i] = self.prepare_single_dataset(ds)
        return dataset_list
        
    def train_val_test_split(self, dataset_name, data_dir, split_at=[60, 80]):
        (train_ds, val_ds, test_ds), metadata = tfds.load(
            dataset_name,
            split=[f'train[:{split_at[0]}%]', f'train[{split_at[0]}%:{split_at[1]}%]', f'train[{split_at[1]}%:]'],
            data_dir=data_dir,
            download=False,
            with_info=True,
            as_supervised=False,
            shuffle_files=False
        )
        return train_ds, val_ds, test_ds, metadata

    def preprocess(self, features):
        image = tf.image.convert_image_dtype(features['image'], tf.float32)
        return image
    
    def configure_for_performance(self, ds):
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        ds = ds.shuffle(buffer_size=1000, seed=self.seed)
        ds = ds.batch(batch_size=self.batch_size)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds





