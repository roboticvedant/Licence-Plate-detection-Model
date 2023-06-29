import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load and preprocess the dataset
# This code assumes you have a dataset of images with bounding box annotations for the license plates
# in Pascal VOC format. If your dataset is in a different format, you will need to modify this code.

data_dir = '/path/to/your/dataset'

datagen = ImageDataGenerator(rescale=1./255,
                             rotation_range=20,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest',
                             validation_split=0.2)  # reserve 20% of images for validation

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training') 

validation_generator = datagen.flow_from_directory(
    data_dir, 
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation')

# Load a pre-trained model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model
base_model.trainable = False

# Add your head on top
x = Flatten()(base_model.output)
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)  

model = Model(base_model.input, x)

# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch = train_generator.samples // 32,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // 32,
    epochs = 20)

# Save the trained model
model.save('license_plate_model.h5')
