from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model_builder import build_model
import os

# Set your dataset path
data_dir = os.path.join('Data', 'Training')  # G:/DL_project/Brain Tumor Detection using CNN/Data/Training

# Initialize ImageDataGenerator with rescaling and validation split
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # 80% train, 20% validation
)

# Training data
train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Validation data
val_data = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# Build and train model
model = build_model(num_classes=train_data.num_classes)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# Save the model
model.save('brain_tumor_model.h5')
