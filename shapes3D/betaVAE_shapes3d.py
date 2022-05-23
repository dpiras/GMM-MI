#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 10:48:52 2021

@author: lillian
"""
import time
from datetime import timedelta, datetime
from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


class betaVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, latent_dim, beta, batch_size=32, seed=330, load_model=False, initialiser=None, mse=True):
        super(betaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.batch_size = batch_size
        self.seed = seed
        self.mse = mse

        print(f'using seed {self.seed}, latent_dim = {latent_dim}')

        if initialiser is None:
            initialiser = tf.keras.initializers.GlorotUniform(seed=self.seed)

        if isinstance(load_model, str):
            self.encoder = tf.keras.models.load_model(load_model + '/encoder')
            self.decoder = tf.keras.models.load_model(load_model + '/decoder')

        else:
            self.latent_dim = latent_dim
            self.beta = beta
            self.encoder = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(input_shape=(64, 64, 3)),
                    tf.keras.layers.Conv2D(
                        filters=32, kernel_size=4, strides=(2, 2), activation='relu', kernel_initializer=initialiser),  # add a layer
                    tf.keras.layers.Conv2D(
                        filters=32, kernel_size=4, strides=(2, 2), activation='relu', kernel_initializer=initialiser),
                    tf.keras.layers.Conv2D(
                        filters=64, kernel_size=4, strides=(2, 2), activation='relu', kernel_initializer=initialiser),
                    tf.keras.layers.Conv2D(
                        filters=64, kernel_size=4, strides=(2, 2), activation='relu', kernel_initializer=initialiser),
                    tf.keras.layers.Flatten(),
                    # No activation
                    tf.keras.layers.Dense(256, kernel_initializer=initialiser),
                    tf.keras.layers.Dense(latent_dim + latent_dim, kernel_initializer=initialiser),  # mean + log_var
                ]
            )

            self.decoder = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                    tf.keras.layers.Dense(units=256, activation=tf.nn.relu, kernel_initializer=initialiser),
                    tf.keras.layers.Dense(units=4*4*64, activation=tf.nn.relu, kernel_initializer=initialiser),
                    tf.keras.layers.Reshape(target_shape=(4, 4, 64)),
                    tf.keras.layers.Conv2DTranspose(
                        filters=64, kernel_size=4, strides=2, padding='same',
                        activation='relu', kernel_initializer=initialiser),
                    tf.keras.layers.Conv2DTranspose(
                        filters=32, kernel_size=4, strides=2, padding='same',
                        activation='relu', kernel_initializer=initialiser),
                    tf.keras.layers.Conv2DTranspose(
                        filters=32, kernel_size=4, strides=2, padding='same',
                        activation='relu', kernel_initializer=initialiser),
                    # No activation
                    tf.keras.layers.Conv2DTranspose(
                        filters=3, kernel_size=4, strides=2, padding='same', kernel_initializer=initialiser),
                ]
            )
        print('encoder summary')
        self.encoder.summary()
        print('decoder summary')
        self.decoder.summary()

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property  # https://keras.io/examples/generative/vae/
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs):
        mean, logvar = self.encode(inputs)
        z = self.reparameterize(mean, logvar)  # sample a z
        x_logit = self.decode(z)

        total_loss, reconstruction_loss, kl_loss = self.compute_loss(
            inputs, x_logit, mean, logvar, beta=self.beta, mse=self.mse)

        self.add_metric(total_loss, name='total_loss')
        self.add_metric(reconstruction_loss, name='reconstruction_loss')
        self.add_metric(kl_loss, name='kl_loss')
        return x_logit

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(1, self.latent_dim), seed=self.seed)  # what does 100 do
        return self.decode(eps, apply_sigmoid=False)

    @tf.function
    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    @tf.function
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=(self.batch_size, self.latent_dim), seed=self.seed)
        return eps * tf.exp(logvar * .5) + mean

    @tf.function
    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    @tf.function
    def compute_loss(self, x, x_logit, mean, logvar, beta, mse=True):
        print('mse shape ', tf.shape(tf.keras.losses.mse(x, x_logit)))
        if mse:
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.mse(x, x_logit), axis=(1, 2)
                )
            )
        else:
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(x, x_logit), axis=(1, 2)
                )
            )

        # modelling latent & approx. posterior as Gaussian
        kl_loss = -0.5 * (1 + logvar - tf.square(mean) - tf.exp(logvar))
        print('kl_loss shape ', tf.shape(kl_loss))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + beta * kl_loss
        return total_loss, reconstruction_loss, kl_loss

    @tf.function
    def train_step(self, x):
        """Executes one training step and returns the loss.

        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """

        with tf.GradientTape() as tape:
            mean, logvar = self.encode(x)
            z = self.reparameterize(mean, logvar)  # sample a z
            x_logit = self.decode(z)
            total_loss, reconstruction_loss, kl_loss = self.compute_loss(
                x, x_logit, mean, logvar, beta=self.beta, mse=self.mse)

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


class RunBetaVAE:
    def __init__(self, vae, latent_dim, beta, epochs, train_ds, val_ds, profile_model=True, early_stopping=False, checkpoint=True, optimiser=None):
        K.clear_session()

        self.latent_dim = latent_dim
        self.beta = beta
        self.num_epochs = epochs

        self.profile_model = profile_model
        self.early_stopping = early_stopping
        self.checkpoint = checkpoint

        if optimiser is None:
            self.optimiser = tf.keras.optimizers.Adam(1e-4)
        else:
            self.optimiser = optimiser

        self.model_savepath = self.create_model_savepath(beta, latent_dim)

        self.history, self.vae_model = self.compile_and_run(vae, self.optimiser, train_ds, val_ds)

        self.save_relevant_data(self.history)

        self.val_loss_arr = np.array(self.history.history['val_total_loss'])

    def create_model_savepath(self, beta, latent_dim):
        if beta == 1:
            vaename = 'Basic'
        else:
            vaename = 'Beta'

        dated = datetime.now().strftime("%Y.%m.%d-%H%M%S")
        model_savepath = f'models/{vaename}VAE_shapes3d-beta={beta}-zdim={latent_dim}-{dated}'
        Path(model_savepath).mkdir(parents=True, exist_ok=True)
        return model_savepath

    def tensorboard_callback(self, log_dir_, profile_batches='500,520'):
        logs = log_dir_  # + datetime.now().strftime("%Y%m%d-%H%M%S")
        tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs,
                                                         histogram_freq=1,
                                                         profile_batch=profile_batches)
        return tboard_callback

    def get_callbacks(self):
        callbacks_list = []
        if self.profile_model:
            callbacks_list.append(self.tensorboard_callback(f'{self.model_savepath}/logs/'))
        if self.early_stopping:
            estop = tf.keras.callbacks.EarlyStopping(monitor="val_total_loss", patience=3, verbose=1,
                                                     restore_best_weights=True)
            callbacks_list.append(estop)
        if self.checkpoint:
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.model_savepath + "/weights-training-ep{epoch:02d}.h5",
                                                             save_weights_only=True,
                                                             verbose=1)
            callbacks_list.append(cp_callback)
        return callbacks_list

    def compile_and_run(self, vae, optimiser, train_ds, val_ds):
        t0 = time.time()
        vae.compile(optimizer=optimiser)
        callbacks_list = self.get_callbacks()
        history = vae.fit(train_ds,
                          epochs=self.num_epochs,
                          validation_data=val_ds,
                          callbacks=callbacks_list
                          )
        print(f'time taken = {timedelta(seconds=(time.time()-t0))}')
        return history, vae

    def save_relevant_data(self, history):
        np.save(f'{self.model_savepath}/total_training_loss.npy', np.array(history.history['loss']))
        np.save(f'{self.model_savepath}/total_val_loss.npy', np.array(history.history['val_total_loss']))
