import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend as K

# 1. Métriques personnalisées pour la segmentation
def dice_coeff(y_true, y_pred, smooth=1.):
    """Calcule le coefficient de Dice (F1-Score pour segmentation)"""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou_metric(y_true, y_pred):
    """Calcule l'Intersection over Union (IoU)"""
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true, axis=-1) + K.sum(y_pred, axis=-1) - intersection
    return K.mean((intersection + 1e-7) / (union + 1e-7))

# 2. Construction de l'architecture U-Net
def conv_block(input_tensor, num_filters):
    """Bloc de base : Conv -> BatchNorm -> ReLU (x2)"""
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def build_unet(input_shape=(128, 128, 1)):
    inputs = layers.Input(input_shape)

    # ENCODEUR (Contracting Path)
    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    c3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # BRIDGE / BOTTLENECK
    b = conv_block(p3, 256)

    # DÉCODEUR (Expansive Path) + Skip Connections
    u1 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(b)
    u1 = layers.Concatenate()([u1, c3]) # Connexion de saut
    d1 = conv_block(u1, 128)

    u2 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(d1)
    u2 = layers.Concatenate()([u2, c2])
    d2 = conv_block(u2, 64)

    u3 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(d2)
    u3 = layers.Concatenate()([u3, c1])
    d3 = conv_block(u3, 32)

    # SORTIE (Segmentation Binaire)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(d3)

    model = keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coeff, iou_metric])
    return model

if __name__ == "__main__":
    model = build_unet()
    model.summary()