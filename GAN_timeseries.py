import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd

class TimeSeriesGAN:
    def __init__(self, sequence_length, num_features, latent_dim=100):
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.latent_dim = latent_dim
        
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
        # Define optimizers
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        
        # Define loss
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def build_generator(self):
        noise_input = layers.Input(shape=(self.latent_dim,))
        
        # First dense layer to get enough units for LSTM
        x = layers.Dense(self.sequence_length * 64)(noise_input)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Reshape((self.sequence_length, 64))(x)
        
        # LSTM layers
        x = layers.LSTM(128, return_sequences=True)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer.  The final shape will be (batch, sequenceLength, num_features)
        output = layers.Dense(self.num_features, activation='tanh')(x)
        
        return models.Model(noise_input, output, name='generator')

    def build_discriminator(self):
        sequence_input = layers.Input(shape=(self.sequence_length, self.num_features))
        
        x = layers.LSTM(64, return_sequences=True)(sequence_input)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.LSTM(32, return_sequences=False)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(16)(x)
        x = layers.LeakyReLU(0.2)(x)
        
        output = layers.Dense(1)(x)
        
        return models.Model(sequence_input, output, name='discriminator')

    @tf.function
    def train_step(self, real_sequences):
        batch_size = tf.shape(real_sequences)[0]
        noise = tf.random.normal([batch_size, self.latent_dim])
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate fake sequences
            generated_sequences = self.generator(noise, training=True)
            
            # Get discriminator decisions
            real_output = self.discriminator(real_sequences, training=True)
            fake_output = self.discriminator(generated_sequences, training=True)
            
            # Calculate losses
            gen_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)
            real_loss = self.loss_fn(tf.ones_like(real_output), real_output)
            fake_loss = self.loss_fn(tf.zeros_like(fake_output), fake_output)
            disc_loss = real_loss + fake_loss
        
        # Calculate gradients
        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        
        # Apply gradients
        self.generator_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))
        
        return gen_loss, disc_loss

    def generate_sequences(self, num_sequences):
        """Generate new time series sequences"""
        noise = tf.random.normal([num_sequences, self.latent_dim])
        generated_sequences = self.generator(noise, training=False)
        return generated_sequences.numpy()

# Example usage with sample data preparation
def prepare_sequences(data, sequence_length):
    """Prepare sequences from DataFrame for training"""
    sequences = []
    for i in range(len(data) - sequence_length):
        sequences.append(data.iloc[i:i + sequence_length].values)
    return np.array(sequences)

#======================================================================================

# Sample data creation (replace this with your actual data)
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=1000, freq='D')
data = pd.DataFrame({
    'value1': np.sin(np.linspace(0, 10*np.pi, 1000)) + np.random.normal(0, 0.1, 1000),
    'value2': np.cos(np.linspace(0, 8*np.pi, 1000)) + np.random.normal(0, 0.1, 1000),
    'value3': np.random.normal(0, 1, 1000)
    }, index=dates)

# Normalize data
scaler = lambda x: (x - x.mean()) / x.std()
normalized_data = data.apply(scaler)

# Prepare sequences
sequence_length = 30  # 30 time steps per sequence
sequences = prepare_sequences(normalized_data, sequence_length)

# Create and train GAN
gan = TimeSeriesGAN(
    sequence_length=sequence_length,
    num_features=normalized_data.shape[1],
    latent_dim=100
    )

# Training parameters
EPOCHS = 100
BATCH_SIZE = 32

# Create dataset
dataset = tf.data.Dataset.from_tensor_slices(sequences).shuffle(1000).batch(BATCH_SIZE)

# Training loop
for epoch in range(EPOCHS):
    gen_losses = []
    disc_losses = []
    
    for sequence_batch in dataset:
        gen_loss, disc_loss = gan.train_step(sequence_batch)
        gen_losses.append(gen_loss)
        disc_losses.append(disc_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}, Gen Loss: {np.mean(gen_losses):.4f}, '
                f'Disc Loss: {np.mean(disc_losses):.4f}')

# Generate new data.  It will produce data with shape (5, 30, 3).
generated = gan.generate_sequences(num_sequences=5)

# Convert generated sequences back to DataFrame format
generated_df = pd.DataFrame(
    generated[0],  # Taking first sequence as example
    columns=data.columns,
    index=pd.date_range(start='2024-01-01', periods=sequence_length, freq='D')
)

print("\nGenerated sequence sample:")
print(generated_df.head())