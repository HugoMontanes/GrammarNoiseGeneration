import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, callbacks # type: ignore
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
                    float(param['frequency']),  # Ensure float conversion
                    float(param['amplitude']),
                    float(param['octaves'])
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

def train_discriminator_only(image_directory, metadata_file, epochs=100, batch_size=32):
    """
    Train only the discriminator for texture analysis.
    
    Args:
        image_directory (str): Directory containing texture images
        metadata_file (str): JSON file with texture parameters
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        
    Returns:
        tuple: (discriminator, history) - Trained model and training history
    """
    # Load dataset
    images, parameters = load_and_preprocess_images(image_directory, metadata_file)
    if images is None or len(images) == 0:
        print("No valid images found. Exiting.")
        return None, None
    
    # Print parameter ranges before normalization
    print("Parameter ranges before normalization:")
    print(f"Min: {np.min(parameters, axis=0)}")
    print(f"Max: {np.max(parameters, axis=0)}")
    
    # Split dataset
    x_train, x_val, y_train, y_val = train_test_split(
        images, parameters, test_size=0.2, random_state=42
    )
    
    # Normalize parameters for model training - WITH SAFETY CHECKS
    param_min = np.min(parameters, axis=0)
    param_max = np.max(parameters, axis=0)
    param_range = param_max - param_min
    
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    param_range = np.where(param_range < epsilon, epsilon, param_range)
    
    y_train_norm = (y_train - param_min) / param_range
    y_val_norm = (y_val - param_min) / param_range
    
    # Check for NaN values after normalization
    if np.isnan(y_train_norm).any() or np.isnan(y_val_norm).any():
        print("Warning: NaN values detected after normalization!")
        print("Non-finite values will be replaced with zeros.")
        y_train_norm = np.nan_to_num(y_train_norm)
        y_val_norm = np.nan_to_num(y_val_norm)
    
    # Save normalization parameters for later use
    norm_params = {
        'min': param_min.tolist(),
        'max': param_max.tolist(),
        'range': param_range.tolist()  # Added range for safety
    }
    with open('parameter_normalization.json', 'w') as f:
        json.dump(norm_params, f)
    
    # Create discriminator model
    discriminator = create_discriminator(output_dims=3)
    discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                          loss='mse',
                          metrics=['mae'])
    
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
    
    # Train discriminator on real data
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
    plt.savefig('discriminator_training_history.png')

    save_dir = "C:/Users/hugom/Documents/GitHub/GrammarNoiseGeneration/assets/models"
    
    # Save the discriminator model in both TensorFlow's format and for ONNX conversion
    discriminator.save(os.path.join(save_dir,'texture_discriminator_for_onnx.h5'))
    
    # Also save model in TensorFlow's SavedModel format which can be better for ONNX conversion
    tf.saved_model.save(discriminator, os.path.join(save_dir, 'texture_discriminator_savedmodel'))
    
    return discriminator, discriminator_history

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
    discriminator, history = train_discriminator_only(image_dir, metadata_file, epochs=50)
    
    # Example: Analyze a new texture
    if discriminator is not None:
        result = analyze_new_texture(discriminator, "C:/Users/hugom/Documents/GitHub/GrammarNoiseGeneration/assets/generated_images/image_1.png")
        
        # Find closest match
        closest = find_closest_match(discriminator, "C:/Users/hugom/Documents/GitHub/GrammarNoiseGeneration/assets/database_images/image_1.png", image_dir, metadata_file)