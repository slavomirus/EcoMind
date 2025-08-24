import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import json
import pickle

tf.get_logger().setLevel('ERROR')

# --- Konfiguracja ---
data_dir = 'EcoMind/datasetv2'
class_names = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
               'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = len(class_names)
EPOCHS = 100


# --- Funkcje do zapisywania i wczytywania historii ---
def save_training_history(history, filename='training_history.pkl'):
    """Zapisz historię treningu do pliku"""
    with open(filename, 'wb') as f:
        pickle.dump(history, f)
    print(f"Historia treningu zapisana jako {filename}")


def load_training_history(filename='training_history.pkl'):
    """Wczytaj historię treningu z pliku"""
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                history = pickle.load(f)
            print(f"Historia treningu wczytana z {filename}")
            return history
        except Exception as e:
            print(f"Błąd podczas wczytywania historii: {e}")
            return None
    else:
        print("Nie znaleziono pliku z historią treningu")
        return None


def save_combined_history(existing_history, new_history, filename='combined_training_history.pkl'):
    """Połącz i zapisz historię treningu"""
    if existing_history and new_history:
        combined_history = {}
        for key in existing_history.keys():
            if key in new_history:
                combined_history[key] = existing_history[key] + new_history[key]
            else:
                combined_history[key] = existing_history[key]
        save_training_history(combined_history, filename)
        return combined_history
    elif new_history:
        save_training_history(new_history, filename)
        return new_history
    else:
        return existing_history


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


# --- Sprawdź czy istnieje zapisany model i historia ---
def load_existing_model_and_history():
    model_path = 'EcoMind/best_model.keras'
    history_path = 'EcoMind/training_history.pkl'

    model = None
    existing_history = None
    initial_epoch = 0

    # Wczytaj model jeśli istnieje
    if os.path.exists(model_path):
        print(f"Znaleziono istniejący model: {model_path}")
        try:
            model = load_model(model_path)
            print("Model pomyślnie wczytany!")

            # Wczytaj historię jeśli istnieje
            existing_history = load_training_history(history_path)
            if existing_history:
                # Oblicz początkową epokę na podstawie historii
                if 'accuracy' in existing_history:
                    initial_epoch = len(existing_history['accuracy'])
                    print(f"Kontynuuję trening od epoki {initial_epoch}")
        except Exception as e:
            print(f"Błąd podczas wczytywania modelu: {e}")
            print("Tworzę nowy model...")
            model = None
            existing_history = None
    else:
        print("Nie znaleziono istniejącego modelu. Tworzę nowy model...")

    return model, existing_history, initial_epoch


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

# --- Sprawdź i wczytaj istniejący model i historię ---
print("==== Sprawdzanie istniejącego modelu i historii ====")
model, existing_history, initial_epoch = load_existing_model_and_history()

if model is None:
    print("==== Tworzenie nowego modelu ====")
    model = create_improved_model()

model.summary()

# --- Trening ---
print(f"\n==== Rozpoczynanie treningu od epoki {initial_epoch} ====")

print("=== Etap 1: Trening z zamrożonymi warstwami ===")
history_stage1 = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks_list,
    class_weight=class_weights_dict,
    verbose=1,
    initial_epoch=initial_epoch
)

# Zapisz historię po pierwszym etapie
current_history = history_stage1.history
if existing_history:
    combined_history = save_combined_history(existing_history, current_history, 'training_history_stage1.pkl')
else:
    save_training_history(current_history, 'training_history_stage1.pkl')
    combined_history = current_history

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

print("=== Etap 2: Fine-tuning ===")
history_stage2 = model.fit(
    train_generator,
    epochs=EPOCHS + 50,
    validation_data=val_generator,
    callbacks=callbacks_list,
    class_weight=class_weights_dict,
    verbose=1,
    initial_epoch=EPOCHS
)

# Połącz i zapisz pełną historię
full_history = {}
for key in history_stage1.history.keys():
    if key in history_stage2.history:
        full_history[key] = history_stage1.history[key] + history_stage2.history[key]

if combined_history:
    final_history = save_combined_history(combined_history, full_history, 'training_history_full.pkl')
else:
    save_training_history(full_history, 'training_history_full.pkl')
    final_history = full_history

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
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
if final_history and 'accuracy' in final_history:
    plt.plot(final_history['accuracy'], label='Trening')
    plt.plot(final_history['val_accuracy'], label='Walidacja')
    plt.title('Dokładność modelu')
    plt.xlabel('Epoka')
    plt.ylabel('Dokładność')
    plt.legend()
else:
    plt.text(0.5, 0.5, 'Brak danych historii', ha='center', va='center')
    plt.title('Brak danych historii')

plt.subplot(1, 3, 2)
if final_history and 'loss' in final_history:
    plt.plot(final_history['loss'], label='Trening')
    plt.plot(final_history['val_loss'], label='Walidacja')
    plt.title('Strata modelu')
    plt.xlabel('Epoka')
    plt.ylabel('Strata')
    plt.legend()
else:
    plt.text(0.5, 0.5, 'Brak danych historii', ha='center', va='center')
    plt.title('Brak danych historii')

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

# Zapisz również w formacie .keras dla przyszłej kontynuacji
model.save('best_model.keras')
print("Model zapisany również jako 'best_model.keras' dla przyszłej kontynuacji treningu")

# Zapisz ostatnią historię
save_training_history(final_history, 'training_history.pkl')
print("Ostatnia historia treningu zapisana")

# Zapisz informacje o treningu do pliku tekstowego
training_info = {
    'total_epochs': len(final_history['accuracy']) if final_history and 'accuracy' in final_history else 0,
    'final_accuracy': test_acc,
    'final_top5_accuracy': test_top5,
    'class_distribution': train_class_counts,
    'timestamp': str(np.datetime64('now'))
}

with open('training_info.json', 'w') as f:
    json.dump(training_info, f, indent=4)
print("Informacje o treningu zapisane do 'training_info.json'")