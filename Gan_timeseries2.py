import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class TimeseriesGenerator(Model):
    def __init__(self, input_dim, hidden_dim, num_features, sequence_length):
        super(TimeseriesGenerator, self).__init__()
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.num_features = num_features
        
        self.initial_dense = layers.Dense(hidden_dim, activation='relu')
        
        self.lstm_1 = layers.LSTM(
            hidden_dim,
            return_sequences=True,
            stateful=False,
            recurrent_dropout=0.2
        )
        
        self.lstm_2 = layers.LSTM(
            hidden_dim,
            return_sequences=True,
            stateful=False,
            recurrent_dropout=0.2
        )
        
        self.output_layer = layers.Dense(num_features)
        
    def call(self, inputs):
        # Process noise vector
        x = self.initial_dense(inputs)
        
        # Reshape for LSTM input: (batch_size, sequence_length, hidden_dim)
        x = tf.expand_dims(x, 1)
        x = tf.repeat(x, self.sequence_length, axis=1)
        
        # Pass through LSTM layers
        x = self.lstm_1(x)
        x = self.lstm_2(x)
        
        # Generate time series data
        outputs = self.output_layer(x)
        
        return outputs

class TimeseriesDiscriminator(Model):
    def __init__(self, num_features, hidden_dim):
        super(TimeseriesDiscriminator, self).__init__()
        
        self.lstm_1 = layers.LSTM(
            hidden_dim,
            return_sequences=True,
            stateful=False,
            recurrent_dropout=0.2
        )
        
        self.lstm_2 = layers.LSTM(
            hidden_dim,
            return_sequences=False,  # We only need the last output
            stateful=False,
            recurrent_dropout=0.2
        )
        
        self.dense_1 = layers.Dense(hidden_dim // 2, activation='relu')
        self.dropout = layers.Dropout(0.3)
        self.dense_2 = layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs):
        x = self.lstm_1(inputs)
        x = self.lstm_2(x)
        x = self.dense_1(x)
        x = self.dropout(x)
        outputs = self.dense_2(x)
        return outputs

class TimeseriesGAN:
    def __init__(self, input_dim, hidden_dim, num_features, sequence_length):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_features = num_features
        self.sequence_length = sequence_length
        
        # Initialize generator and discriminator
        self.generator = TimeseriesGenerator(input_dim, hidden_dim, num_features, sequence_length)
        self.discriminator = TimeseriesDiscriminator(num_features, hidden_dim)
        
        # Define optimizers
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        
        # Define loss
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
    
    @tf.function
    def train_step(self, real_data):
        batch_size = tf.shape(real_data)[0]
        
        # Generate random noise
        noise = tf.random.normal([batch_size, self.input_dim])
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate fake data
            generated_data = self.generator(noise, training=True)
            
            # Get discriminator outputs
            real_output = self.discriminator(real_data, training=True)
            fake_output = self.discriminator(generated_data, training=True)
            
            # Calculate losses
            gen_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)
            
            real_loss = self.loss_fn(tf.ones_like(real_output), real_output)
            fake_loss = self.loss_fn(tf.zeros_like(fake_output), fake_output)
            disc_loss = real_loss + fake_loss
            
        # Calculate gradients
        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        
        # Apply gradients
        self.g_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))
        
        return gen_loss, disc_loss
    
    def train(self, dataset, epochs):
        for epoch in range(epochs):
            for batch in dataset:
                g_loss, d_loss = self.train_step(batch)
                
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Gen Loss: {g_loss:.4f}, Disc Loss: {d_loss:.4f}')
    
    def generate(self, num_samples):
        noise = tf.random.normal([num_samples, self.input_dim])
        return self.generator(noise, training=False)

# Example usage
def create_and_train_gan(sequence_length=24, num_features=3, hidden_dim=64, input_dim=64,
                        batch_size=32, epochs=100):
    # Create the GAN
    gan = TimeseriesGAN(input_dim, hidden_dim, num_features, sequence_length)
    
    # Prepare your dataset
    # Assuming your_data is a numpy array of shape (num_samples, sequence_length, num_features)
    # dataset = tf.data.Dataset.from_tensor_slices(your_data).batch(batch_size)
    
    # Train the GAN
    # gan.train(dataset, epochs)
    
    return gan
