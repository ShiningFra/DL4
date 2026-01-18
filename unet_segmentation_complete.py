import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend as K
import mlflow
import numpy as np

# --- PARTIE 1: METRIQUES SEGMENTATION --- [cite: 122-157]
def dice_coeff(y_true, y_pred, smooth=1.):
    """Coefficient de Dice : 2 * Intersection / (Somme des Aires)"""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou_metric(y_true, y_pred):
    """Intersection over Union (Jaccard Index)"""
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true, axis=-1) + K.sum(y_pred, axis=-1) - intersection
    return K.mean((intersection + 1e-7) / (union + 1e-7))

# --- PARTIE 2: ARCHITECTURE U-NET --- [cite: 59-121]
def conv_block(input_tensor, num_filters):
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def build_unet(input_shape=(128, 128, 1)):
    inputs = layers.Input(input_shape)

    # Encodeur (Contracting Path)
    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    c3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Bridge (Bottleneck)
    b = conv_block(p3, 256)

    # Décodeur (Expansive Path) + Skip Connections
    u1 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(b)
    u1 = layers.Concatenate()([u1, c3]) # Skip connection [cite: 66, 105]
    d1 = conv_block(u1, 128)

    u2 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(d1)
    u2 = layers.Concatenate()([u2, c2])
    d2 = conv_block(u2, 64)

    u3 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(d2)
    u3 = layers.Concatenate()([u3, c1])
    d3 = conv_block(u3, 32)

    # Sortie binaire
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(d3)

    model = keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coeff, iou_metric])
    return model

# --- PARTIE 3: TRACKING MLFLOW (Exemple 3D) --- [cite: 172-224]
def run_mlflow_experiment():
    mlflow.set_experiment("Medical_Segmentation_TP4")
    with mlflow.start_run(run_name="U-Net_Standard_Adam"):
        model = build_unet()
        
        # Logging des paramètres
        mlflow.log_param("architecture", "U-Net")
        mlflow.log_param("filters_base", 32)
        mlflow.log_param("optimizer", "adam")
        
        # Simulation d'entraînement
        mlflow.log_metric("final_dice", 0.88)
        mlflow.log_metric("final_iou", 0.79)
        
        print("Expérience loggée dans MLflow.")

if __name__ == "__main__":
    run_mlflow_experiment()