
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models # type: ignore
import os
from PIL import Image
import json

tf.config.list_physical_devices('GPU')
tf.__version__

def create_discriminator():
    model = models.Sequential([
        #First Conv Block
        layers.Input(shape = (256, 256, 1)), #Size of the input image
        layers.Conv2D(64, 4, strides = 2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        #Second
        layers.Conv2D(128, 4, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        #Third
        layers.Conv2D(256, 4, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        #Flatten
        layers.Flatten(),
        layers.Dense(512),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(3, activation='sigmoid') #Depending of the number of categories it will be one or another
        
    ])
    return model

discriminator = create_discriminator(4)
discriminator.summary()

def load_and_preprocess_images(image_directory, metadata_file):
    
    #Load tags information
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    images = []
    parameters = []

    #Proess each image
    for filename in os.listdir(image_directory):
        if filename.lower().endwith(('.png', '.jpg', '.jpeg')):
            if filename not in metadata:
                continue #Skip image without metadata

            img_path = os.path.join(image_directory, filename)

            #Process image
            image = Image.open(img_path).convert('L')
            image = image.resize((256, 256))
            image_array = np.array(image)/255.0

            #Get parameters
            param = metadata[filename]['parameters']
            param_vector=[
                param['frequency'],
                param['amplitude'],
                param['octaves']
            ]

            images.append(image_array)
            parameters.append(param_vector)

    return np.array(images)[..., np.newaxis], np.array(parameters)

    


def train_texture_classifier(image_directory, metadata_file, epochs=100):

    #Load dataset
    images, parameters = load_and_preprocess_images(image_directory, metadata_file)

    #Create and train model
    model = create_discriminator()
    model.compile(optimize=tf.keras.optimizers.Adam(learning_rate=0.0002),
                loss = 'mse') #Mean squared error for numerical prediction
    #Train
    history = model.fit(images, parameters, epochs=epochs, batch_size = 32)

    #Save model
    tf.saved_model.save(model, "perlin_model")
    
    return model

def analyze_new_texture(model, image_path):

    #Load and process new image
    image = Image.open(image_path).convert('L')
    image = image.resize((256, 256))
    image_array = np.array(image)/255.0
    image_array = image_array[np.newaxis, ..., np.newaxis]

    #Get predictions
    predictions = model.predict(image_array)[0]

    #Show results
    print("Perlin NOise Parameter Predictions: ")
    print(f"Frequency: {predictions[0]:.4f}")
    print(f"Amplitude: {predictions[1]:.4f}")
    print(f"Octaves: {round(predictions[2])}") #Round octaves to nearest integer

    return{
        "frequency": predictions[0],
        "aplitude": predictions[1],
        "octaves": round(predictions[2])
    }

def find_closest_match(model, new_image_path, image_directory, metadata_file):

    #Get predicted parameters for new image
    new_params = analyze_new_texture(model, new_image_path)

    #Load metadata
    with open(metadata_file,'r') as f:
        metadata = json.load(f)

    #find closest match
    best_match = None
    best_score = float('inf')

    for filename, data in metadata.items():
        params = data['parameters']

        #Calculate distance between parameters (normalized)
        freq_diff = (new_params['frequency'] - params['frequency'])**2
        amp_diff = (new_params['amplitude'] - params['amplitude'])**2
        oct_diff = (new_params['octaves'] - params['octaves'])**2/10 #Lower weight for octaves

        score = freq_diff + amp_diff + oct_diff

        if score < best_score:
            best_score = score
            best_match = filename

    print(f"\nClosest matching texture: {best_match}")
    print(f"Similarity score: {1/(1+best_score):.4f}") #Convert to 0-1 scale

    return best_match

model = train_texture_classifier("database_images", "")

