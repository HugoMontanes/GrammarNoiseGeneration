import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, callbacks
import os
from PIL import Image
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Check GPU availability
physical_devices = tf.config.list_physical_devices('GPU')
print(f"TensorFlow version: {tf.__version__}")
print(f"Available GPUs: {physical_devices}")

def create_discriminator(output_dims=3):
    """
    Creates a discriminator/classifier model for texture analysis.
    
    Args:
        output_dims (int): Number of parameters to predict
        
    Returns:
        tf.keras.Model: The compiled discriminator model
    """
    model = models.Sequential([
        # First Conv Block
        layers.Input(shape=(256, 256, 1)),  # Size of the input image
        layers.Conv2D(64, 4, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        
        # Second
        layers.Conv2D(128, 4, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        
        # Third
        layers.Conv2D(256, 4, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        
        # Fourth (added depth)
        layers.Conv2D(512, 4, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        
        # Flatten
        layers.Flatten(),
        layers.Dense(512),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),  # Prevent overfitting
        layers.Dense(output_dims, activation='linear')  # Linear activation for regression
    ])
    return model

def create_generator(latent_dim=100):
    """
    Creates a generator model for texture synthesis.
    
    Args:
        latent_dim (int): Size of the input noise vector
        
    Returns:
        tf.keras.Model: The generator model
    """
    noise_input = layers.Input(shape=(latent_dim + 3,))  # Noise + conditional parameters
    
    x = layers.Dense(16 * 16 * 256)(noise_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Reshape((16, 16, 256))(x)
    
    # Upsampling blocks
    x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    x = layers.Conv2DTranspose(128, 4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    x = layers.Conv2DTranspose(64, 4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    x = layers.Conv2DTranspose(32, 4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    # Output layer
    output = layers.Conv2D(1, 3, padding='same', activation='tanh')(x)
    
    # Create model
    model = models.Model(noise_input, output)
    return model

def load_and_preprocess_images(image_directory, metadata_file):
    """
    Load and preprocess images and their associated parameters.
    
    Args:
        image_directory (str): Directory containing texture images
        metadata_file (str): JSON file with texture parameters
        
    Returns:
        tuple: (images, parameters) as numpy arrays
    """
    # Load tags information
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print(f"Error: Metadata file {metadata_file} not found.")
        return None, None
    except json.JSONDecodeError:
        print(f"Error: Metadata file {metadata_file} is not valid JSON.")
        return None, None
    
    images = []
    parameters = []
    valid_files = 0
    skipped_files = 0

    # Process each image
    for filename in os.listdir(image_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            if filename not in metadata:
                print(f"Warning: No metadata found for {filename}")
                skipped_files += 1
                continue  # Skip image without metadata

            img_path = os.path.join(image_directory, filename)

            try:
                # Process image
                image = Image.open(img_path).convert('L')
                image = image.resize((256, 256))
                image_array = np.array(image) / 255.0 * 2.0 - 1.0  # Scale to [-1, 1]

                # Get parameters
                param = metadata[filename]['parameters']
                param_vector = [
                    param['frequency'],
                    param['amplitude'],
                    param['octaves']
                ]

                images.append(image_array)
                parameters.append(param_vector)
                valid_files += 1
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                skipped_files += 1

    print(f"Processed {valid_files} valid files, skipped {skipped_files} files")
    return np.array(images)[..., np.newaxis], np.array(parameters)

def create_gan(generator, discriminator):
    """
    Create a GAN model that connects generator and discriminator.
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
        
    Returns:
        tf.keras.Model: The GAN model
    """
    # Freeze discriminator weights for generator training
    discriminator.trainable = False
    
    # GAN input (noise vector + parameters)
    gan_noise_input = layers.Input(shape=(100,))
    gan_param_input = layers.Input(shape=(3,))
    
    # Concatenate inputs for generator
    combined_input = layers.Concatenate()([gan_noise_input, gan_param_input])
    
    # Generate image
    generated_img = generator(combined_input)
    
    # Discriminator determines parameters from generated image
    gan_output = discriminator(generated_img)
    
    # Create and compile GAN
    gan = models.Model([gan_noise_input, gan_param_input], gan_output)
    gan.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                loss='mse')
    
    return gan

def train_texture_gan(image_directory, metadata_file, epochs=100, batch_size=32):
    """
    Train a GAN for texture analysis and generation.
    
    Args:
        image_directory (str): Directory containing texture images
        metadata_file (str): JSON file with texture parameters
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        
    Returns:
        tuple: (discriminator, generator, history) - Trained models and training history
    """
    # Load dataset
    images, parameters = load_and_preprocess_images(image_directory, metadata_file)
    if images is None or len(images) == 0:
        print("No valid images found. Exiting.")
        return None, None, None
    
    # Split dataset
    x_train, x_val, y_train, y_val = train_test_split(
        images, parameters, test_size=0.2, random_state=42
    )
    
    # Normalize parameters for model training
    param_min = np.min(parameters, axis=0)
    param_max = np.max(parameters, axis=0)
    param_range = param_max - param_min
    
    y_train_norm = (y_train - param_min) / param_range
    y_val_norm = (y_val - param_min) / param_range
    
    # Save normalization parameters for later use
    norm_params = {
        'min': param_min.tolist(),
        'max': param_max.tolist()
    }
    with open('parameter_normalization.json', 'w') as f:
        json.dump(norm_params, f)
    
    # Create models
    discriminator = create_discriminator(output_dims=3)
    discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                          loss='mse',
                          metrics=['mae'])
    
    generator = create_generator(latent_dim=100)
    gan = create_gan(generator, discriminator)
    
    # Create callbacks
    checkpoint = callbacks.ModelCheckpoint(
        'texture_discriminator.h5',
        save_best_only=True,
        monitor='val_loss'
    )
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train discriminator first on real data
    discriminator_history = discriminator.fit(
        x_train, y_train_norm,
        validation_data=(x_val, y_val_norm),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stopping]
    )
    
    # Visualize training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(discriminator_history.history['loss'])
    plt.plot(discriminator_history.history['val_loss'])
    plt.title('Discriminator Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(discriminator_history.history['mae'])
    plt.plot(discriminator_history.history['val_mae'])
    plt.title('Mean Absolute Error')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    
    # Save models
    discriminator.save('texture_discriminator_final.h5')
    generator.save('texture_generator.h5')
    
    # Implement GAN training (adversarial)
    # Set up training parameters
    gan_epochs = 100
    steps_per_epoch = len(x_train) // batch_size
    
    # Lists to store losses
    d_losses = []
    g_losses = []
    
    # Adversarial training loop
    for epoch in range(gan_epochs):
        d_loss_epoch = 0
        g_loss_epoch = 0
        
        for step in range(steps_per_epoch):
            # Select a random batch of images
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            real_images = x_train[idx]
            real_params = y_train_norm[idx]
            
            # Generate random noise
            noise = np.random.normal(0, 1, (batch_size, 100))
            
            # Train discriminator
            # First unfreeze discriminator
            discriminator.trainable = True
            
            # Train on real images
            d_loss_real = discriminator.train_on_batch(real_images, real_params)[0]
            
            # Generate fake images
            gen_input = np.concatenate([noise, real_params], axis=1)
            fake_images = generator.predict(gen_input)
            
            # Train on fake images
            d_loss_fake = discriminator.train_on_batch(fake_images, real_params)[0]
            
            # Calculate total discriminator loss
            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            d_loss_epoch += d_loss
            
            # Train generator
            # Freeze discriminator when training generator
            discriminator.trainable = False
            
            # Generate new noise
            noise = np.random.normal(0, 1, (batch_size, 100))
            
            # Train generator to fool discriminator
            g_loss = gan.train_on_batch([noise, real_params], real_params)
            g_loss_epoch += g_loss
        
        # Calculate average losses for epoch
        d_loss_epoch /= steps_per_epoch
        g_loss_epoch /= steps_per_epoch
        
        # Store losses
        d_losses.append(d_loss_epoch)
        g_losses.append(g_loss_epoch)
        
        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{gan_epochs}, D Loss: {d_loss_epoch:.4f}, G Loss: {g_loss_epoch:.4f}")
        
        # Generate and save sample images
        if epoch % 20 == 0:
            # Generate sample images
            sample_noise = np.random.normal(0, 1, (3, 100))
            # Use the first 3 parameter sets from validation data
            sample_params = y_val_norm[:3]
            sample_gen_input = np.concatenate([sample_noise, sample_params], axis=1)
            sample_images = generator.predict(sample_gen_input)
            
            # Save images
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for i in range(3):
                # Convert from [-1, 1] to [0, 1] range
                img = (sample_images[i] + 1) / 2.0
                axes[i].imshow(img.squeeze(), cmap='gray')
                axes[i].axis('off')
                # Denormalize parameters for display
                params_real = sample_params[i] * param_range + param_min
                axes[i].set_title(f"F:{params_real[0]:.2f}, A:{params_real[1]:.2f}, O:{int(params_real[2])}")
            
            plt.tight_layout()
            plt.savefig(f'gan_samples_epoch_{epoch}.png')
            plt.close()
    
    # Plot GAN training history
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.title('GAN Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('gan_training_history.png')
    
    # Save final models
    discriminator.save('texture_discriminator_gan.h5')
    generator.save('texture_generator_gan.h5')
    
    return discriminator, generator, discriminator_history

def analyze_new_texture(model, image_path):
    """
    Analyze a new texture image to predict its parameters.
    
    Args:
        model: Trained discriminator model
        image_path (str): Path to the image file
        
    Returns:
        dict: Predicted parameters
    """
    try:
        # Load normalization parameters
        with open('parameter_normalization.json', 'r') as f:
            norm_params = json.load(f)
        
        param_min = np.array(norm_params['min'])
        param_max = np.array(norm_params['max'])
        param_range = param_max - param_min
        
        # Load and process new image
        image = Image.open(image_path).convert('L')
        image = image.resize((256, 256))
        image_array = np.array(image) / 255.0 * 2.0 - 1.0  # Scale to [-1, 1]
        image_array = image_array[np.newaxis, ..., np.newaxis]
        
        # Get normalized predictions
        predictions_norm = model.predict(image_array)[0]
        
        # Denormalize
        predictions = predictions_norm * param_range + param_min
        
        # Show results
        print("\nPerlin Noise Parameter Predictions:")
        print(f"Frequency: {predictions[0]:.4f}")
        print(f"Amplitude: {predictions[1]:.4f}")
        print(f"Octaves: {round(predictions[2])}")  # Round octaves to nearest integer
        
        return {
            "frequency": predictions[0],
            "amplitude": predictions[1],
            "octaves": round(predictions[2])
        }
    except Exception as e:
        print(f"Error analyzing texture: {str(e)}")
        return None

def find_closest_match(model, new_image_path, image_directory, metadata_file):
    """
    Find the closest matching texture in the database.
    
    Args:
        model: Trained discriminator model
        new_image_path (str): Path to the query image
        image_directory (str): Directory of reference images
        metadata_file (str): JSON file with texture parameters
        
    Returns:
        str: Filename of the closest match
    """
    try:
        # Get predicted parameters for new image
        new_params = analyze_new_texture(model, new_image_path)
        if new_params is None:
            return None
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Find closest match
        best_match = None
        best_score = float('inf')
        matches = []
        
        for filename, data in metadata.items():
            params = data['parameters']
            
            # Calculate distance between parameters (weighted)
            freq_diff = (new_params['frequency'] - params['frequency'])**2
            amp_diff = (new_params['amplitude'] - params['amplitude'])**2
            oct_diff = (new_params['octaves'] - params['octaves'])**2 * 0.5  # Lower weight for octaves
            
            score = freq_diff + amp_diff + oct_diff
            
            # Store all matches and scores
            matches.append((filename, score))
            
            if score < best_score:
                best_score = score
                best_match = filename
        
        # Sort matches by score
        matches.sort(key=lambda x: x[1])
        
        # Display top 3 matches
        print("\nTop matches:")
        for i, (name, score) in enumerate(matches[:3]):
            similarity = 1 / (1 + score)
            print(f"{i+1}. {name} - Similarity: {similarity:.4f}")
        
        print(f"\nBest matching texture: {best_match}")
        print(f"Similarity score: {1/(1+best_score):.4f}")  # Convert to 0-1 scale
        
        return best_match
    except Exception as e:
        print(f"Error finding closest match: {str(e)}")
        return None

def generate_texture(generator, parameters, noise_dim=100):
    """
    Generate a texture with specific parameters.
    
    Args:
        generator: Trained generator model
        parameters (dict): Parameters for the texture
        noise_dim (int): Dimension of the noise vector
        
    Returns:
        numpy.ndarray: Generated texture image
    """
    # Create noise vector
    noise = np.random.normal(0, 1, (1, noise_dim))
    
    # Create parameter vector
    param_vector = np.array([[
        parameters['frequency'],
        parameters['amplitude'],
        parameters['octaves']
    ]])
    
    # Load normalization parameters
    with open('parameter_normalization.json', 'r') as f:
        norm_params = json.load(f)
    
    param_min = np.array(norm_params['min'])
    param_max = np.array(norm_params['max'])
    param_range = param_max - param_min
    
    # Normalize parameters
    param_vector_norm = (param_vector - param_min) / param_range
    
    # Generate image
    generated_image = generator.predict([noise, param_vector_norm])[0]
    
    # Convert from [-1, 1] to [0, 1] range
    generated_image = (generated_image + 1) / 2.0
    
    return generated_image

# Main execution
if __name__ == "__main__":
    import os
    # Get the absolute path to the database_images directory
    image_dir = os.path.abspath("C:/Users/hugom/Documents/GitHub/GrammarNoiseGeneration/assets/database_images")

    if not os.path.exists(image_dir):
        print(f"Error: Directory {image_dir} does not exist")
        exit(1)

    metadata_file = os.path.abspath("C:/Users/hugom/Documents/GitHub/GrammarNoiseGeneration/assets/database_images/tags.json")

    if not os.path.exists(metadata_file):
        print(f"Error: Directory {metadata_file} does not exist")
        exit(1)
    
    # Train model
    discriminator, generator, history = train_texture_gan(image_dir, metadata_file, epochs=50)
    
    # Example: Analyze a new texture
    if discriminator is not None:
        result = analyze_new_texture(discriminator, "C:/Users/hugom/Documents/GitHub/GrammarNoiseGeneration/assets/database_images/image_1.png")
        
        # Find closest match
        closest = find_closest_match(discriminator, "C:/Users/hugom/Documents/GitHub/GrammarNoiseGeneration/assets/database_images/image_1.png", image_dir, metadata_file)
        
        # Generate a similar texture
        if generator is not None and result is not None:
            generated = generate_texture(generator, result)
            
            # Save generated image
            generated_img = (generated * 255).astype(np.uint8).squeeze()
            Image.fromarray(generated_img).save("generated_texture.png")