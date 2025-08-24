import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

tf.get_logger().setLevel('ERROR')

# --- Konfiguracja ---
data_dir = 'datasetv2'
class_names = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
               'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']

IMG_SIZE = (224, 224)  # Zwiększ rozmiar dla lepszej dokładności
BATCH_SIZE = 32
NUM_CLASSES = len(class_names)
EPOCHS = 100

# --- Analiza i przygotowanie danych ---
print("==== Analiza danych ====")


# Sprawdź rozkład klas
def check_class_distribution(directory):
    class_counts = {}
    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        if os.path.exists(class_dir):
            class_counts[class_name] = len(os.listdir(class_dir))
        else:
            class_counts[class_name] = 0
    return class_counts


train_class_counts = check_class_distribution(data_dir)
print("Rozkład klas w danych treningowych:")
for cls, count in train_class_counts.items():
    print(f"{cls}: {count} obrazów")

# --- Ładowanie danych z poprawionym podziałem ---
print("\n==== Wczytywanie danych ====")

# Użyj image_data_generator dla lepszej augmentacji
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    validation_split=0.2,
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=123
)

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=123
)

# Oblicz wagi klas dla niezbalansowanych danych
class_weights = compute_class_weight(
    'balanced',
    classes=np.arange(NUM_CLASSES),
    y=train_generator.classes
)
class_weights_dict = dict(enumerate(class_weights))
print("Wagi klas:", class_weights_dict)


# --- Poprawiona architektura modelu ---
def create_improved_model():
    # Użyj bardziej zaawansowanej bazy
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )

    base_model.trainable = False

    inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )

    return model


# --- Callbacks ---
callbacks_list = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
    ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=1),
    TensorBoard(log_dir='./logs', histogram_freq=1)
]

# --- Trening ---
print("==== Krok 1: Trening z zamrożonymi warstwami ====")
model = create_improved_model()
model.summary()

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks_list,
    class_weight=class_weights_dict,
    verbose=1
)

print("\n==== Krok 2: Fine-tuning ====")
# Odmroź więcej warstw
base_model = model.layers[1]
base_model.trainable = True

# Zamroź pierwsze 100 warstw, odmróź resztę
for layer in base_model.layers[:100]:
    layer.trainable = False
for layer in base_model.layers[100:]:
    layer.trainable = True

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy', 'top_k_categorical_accuracy']
)

model.summary()

history_fine_tune = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks_list,
    class_weight=class_weights_dict,
    verbose=1
)

# --- Ocena ---
print("\n==== Ocena modelu ====")
# Stwórz generator testowy bez augmentacji
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_loss, test_acc, test_top5 = model.evaluate(test_generator)
print(f'Dokładność testowa: {test_acc:.4f}')
print(f'Top-5 dokładność: {test_top5:.4f}')

# --- Wizualizacja ---
full_history = {
    'accuracy': history.history['accuracy'] + history_fine_tune.history['accuracy'],
    'val_accuracy': history.history['val_accuracy'] + history_fine_tune.history['val_accuracy'],
    'loss': history.history['loss'] + history_fine_tune.history['loss'],
    'val_loss': history.history['val_loss'] + history_fine_tune.history['val_loss']
}

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(full_history['accuracy'], label='Trening')
plt.plot(full_history['val_accuracy'], label='Walidacja')
plt.title('Dokładność modelu')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(full_history['loss'], label='Trening')
plt.plot(full_history['val_loss'], label='Walidacja')
plt.title('Strata modelu')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.legend()

# Macierz pomyłek
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

test_generator.reset()
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

plt.subplot(1, 3, 3)
cm = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Macierz pomyłek')
plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()

print("\nRaport klasyfikacji:")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

# Zapisz finalny model
model.save('final_waste_classification_model.h5')
print("Model zapisany jako 'final_waste_classification_model.h5'")