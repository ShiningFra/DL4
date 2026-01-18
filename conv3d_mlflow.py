import mlflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def simple_conv3d_block(input_shape=(32, 32, 32, 1)):
    """Bloc de convolution 3D pour données volumétriques"""
    inputs = keras.Input(input_shape)
    x = layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPool3D((2, 2, 2))(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return keras.Model(inputs, outputs)

# Tracking avec MLflow
mlflow.set_experiment("3D_Volumetric_Analysis")
with mlflow.start_run(run_name="Conv3D_Medical_Baseline"):
    model_3d = simple_conv3d_block()
    
    # Log des paramètres techniques et architecture
    mlflow.log_dict({"config": model_3d.to_json()}, "architecture.json")
    mlflow.log_param("filters_initial", 16)
    mlflow.log_param("optimizer", "adam")
    
    # Simulation de log de métrique finale
    mlflow.log_metric("final_dice", 0.84)
    print("Tracking MLflow terminé avec succès.")