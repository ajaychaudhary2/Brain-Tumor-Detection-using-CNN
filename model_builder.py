
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout


def build_model(input_shape=(224,224,3),num_class=4):
    
    base_model=ResNet50(weights='imagenet',include_top=False,input_shape=input_shape)
    
    for layer in base_model.layers:
        layer.trainable=False
        
        
    x=base_model.output
    x=Flatten()(x)
    x=Dense(128,activation='relu')(x)
    x=Dropout(0.5)(x)
    
    output = Dense(num_class,activation='softmax')(x)
    
    model = Model(input=base_model.input,outputs=output)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model