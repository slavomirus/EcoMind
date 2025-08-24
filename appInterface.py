import os
import kivy
import numpy as np
import threading
import queue
import cv2
import tensorflow as tf
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from ultralytics import YOLO

kivy.require('2.0.0')

# Konfiguracja Twojego modelu
IMG_SIZE = (128, 128)
CLASS_NAMES = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
               'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']


class WasteClassificationApp(App):
    def build(self):
        Window.clearcolor = (0.1, 0.1, 0.1, 1)

        root = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Górny panel z kontrolkami
        control_panel = BoxLayout(size_hint_y=0.1, spacing=10)

        self.toggle_btn = ToggleButton(
            text='Uruchom kamerę',
            on_press=self.toggle_camera,
            size_hint_x=0.3
        )

        self.info_label = Label(
            text="Naciśnij przycisk aby uruchomić kamerę",
            size_hint_x=0.7
        )

        control_panel.add_widget(self.toggle_btn)
        control_panel.add_widget(self.info_label)
        root.add_widget(control_panel)

        # Podgląd kamery
        self.camera_image = Image(size_hint=(1, 0.6))
        root.add_widget(self.camera_image)

        # Panel wyników
        results_panel = BoxLayout(orientation='vertical', size_hint_y=0.3, spacing=5)

        self.prediction_label = Label(
            text="Wynik klasyfikacji:",
            font_size='18sp',
            color=(0, 1, 0, 1),
            size_hint_y=0.4
        )

        self.confidence_label = Label(
            text="Pewność: 0.00%",
            font_size='16sp',
            color=(1, 1, 1, 1),
            size_hint_y=0.3
        )

        self.detection_label = Label(
            text="Wykryte obiekty: 0",
            font_size='16sp',
            color=(1, 1, 1, 1),
            size_hint_y=0.3
        )

        results_panel.add_widget(self.prediction_label)
        results_panel.add_widget(self.confidence_label)
        results_panel.add_widget(self.detection_label)
        root.add_widget(results_panel)

        self.result_queue = queue.Queue()
        self.is_running = False
        self.camera_active = False
        self.model = None
        self.yolo_model = None

        return root

    def toggle_camera(self, instance):
        if instance.state == 'down':
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        try:
            self.info_label.text = "Wczytuję modele..."
            self.toggle_btn.text = 'Stop'
            self.toggle_btn.disabled = True

            # Wczytaj modele w tle
            threading.Thread(target=self.load_models_thread, daemon=True).start()

        except Exception as e:
            self.info_label.text = f"Błąd: {str(e)}"
            self.toggle_btn.state = 'normal'

    def load_models_thread(self):
        try:
            # Wczytaj YOLO do wykrywania obiektów
            self.info_label.text = "Wczytuję YOLO..."
            self.yolo_model = YOLO("yolov8n.pt")

            # Wczytaj Twój model do klasyfikacji
            self.info_label.text = "Wczytuję best_model.keras..."
            self.model = tf.keras.models.load_model("best_model.keras")

            Clock.schedule_once(lambda dt: self.update_info("Modele załadowane. Otwieram kamerę..."), 0)

            # Inicjalizacja kamery
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise IOError("Nie można otworzyć kamery.")

            # Ustawienie parametrów kamery
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            self.camera_active = True
            self.is_running = True

            # Uruchom wątek przetwarzania
            self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
            self.processing_thread.start()

            Clock.schedule_once(lambda dt: self.update_info("Kamera uruchomiona. Wykrywam obiekty..."), 0)
            Clock.schedule_interval(self.update_ui, 1.0 / 30.0)

            # Odblokuj przycisk
            Clock.schedule_once(lambda dt: setattr(self.toggle_btn, 'disabled', False), 0)

        except Exception as e:
            Clock.schedule_once(lambda dt: self.show_error(f"Błąd ładowania: {str(e)}"), 0)
            Clock.schedule_once(lambda dt: setattr(self.toggle_btn, 'disabled', False), 0)

    def stop_camera(self):
        self.is_running = False
        self.camera_active = False
        self.toggle_btn.text = 'Uruchom kamerę'
        self.info_label.text = "Kamera zatrzymana"

        if hasattr(self, 'cap'):
            self.cap.release()

        # Wyczyść podgląd
        self.camera_image.texture = None
        self.prediction_label.text = "Wynik klasyfikacji:"
        self.confidence_label.text = "Pewność: 0.00%"
        self.detection_label.text = "Wykryte obiekty: 0"

    def update_info(self, message):
        self.info_label.text = message

    def show_error(self, message):
        self.info_label.text = message
        self.toggle_btn.state = 'normal'
        self.toggle_btn.text = 'Uruchom kamerę'

    def preprocess_frame(self, frame):
        """Przygotuj klatkę do predykcji Twojego modelu"""
        # Zmniejsz do wymaganego rozmiaru 128x128
        resized = cv2.resize(frame, IMG_SIZE)
        # Konwersja BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalizacja do [0, 1]
        normalized = rgb.astype('float32') / 255.0
        # Dodaj batch dimension
        return np.expand_dims(normalized, axis=0)

    def processing_loop(self):
        while self.is_running and self.camera_active:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                # ODBICIĘ LUSTRZANE - aby wyglądało naturalnie
                frame = cv2.flip(frame, 1)  # 1 = flip horizontal (lustrzane odbicie)

                # Wykryj obiekty używając YOLO
                results = self.yolo_model(frame, verbose=False, conf=0.5)
                display_frame = frame.copy()
                detected_objects = []

                if results and results[0].boxes:
                    for i, box in enumerate(results[0].boxes):
                        x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0].tolist()]
                        yolo_conf = float(box.conf[0])
                        yolo_cls = int(box.cls[0])
                        yolo_name = self.yolo_model.names[yolo_cls]

                        # Wytnij wykryty obiekt
                        roi = frame[y1:y2, x1:x2]
                        if roi.size > 0 and roi.shape[0] > 20 and roi.shape[1] > 20:
                            try:
                                # Klasyfikuj swoim modelem
                                processed = self.preprocess_frame(roi)
                                predictions = self.model.predict(processed, verbose=0)

                                predicted_class = np.argmax(predictions[0])
                                confidence = float(np.max(predictions[0]))
                                class_name = CLASS_NAMES[predicted_class]

                                # Zapisz informacje o obiekcie
                                detected_objects.append({
                                    'bbox': (x1, y1, x2, y2),
                                    'class_name': class_name,
                                    'confidence': confidence,
                                    'yolo_name': yolo_name,
                                    'yolo_confidence': yolo_conf
                                })

                                # Narysuj bounding box
                                color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255) if confidence > 0.4 else (0,
                                                                                                                     0,
                                                                                                                     255)
                                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

                                # Dodaj etykietę z wynikiem klasyfikacji
                                label = f"{class_name} ({confidence:.2f})"
                                cv2.putText(display_frame, label, (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                            except Exception as e:
                                print(f"Błąd klasyfikacji: {e}")
                                # Narysuj bbox tylko z nazwą YOLO
                                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                                cv2.putText(display_frame, f"{yolo_name} ({yolo_conf:.2f})",
                                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Przygotuj dane dla UI
                result_data = {
                    'frame': display_frame,
                    'detected_objects': detected_objects,
                    'total_detections': len(detected_objects)
                }

                self.result_queue.put(result_data)

            except Exception as e:
                print(f"Błąd przetwarzania: {e}")
                continue

    def update_ui(self, dt):
        try:
            if self.result_queue.empty():
                return

            result_data = self.result_queue.get_nowait()

            # Konwersja klatki do tekstury Kivy
            frame = result_data['frame']

            # Dla Kivy musimy odwrócić obraz wertykalnie (bo Kivy ma odwrócony układ Y)
            frame_flipped = cv2.flip(frame, 0)

            # Konwersja BGR to RGB
            frame_rgb = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2RGB)
            buf = frame_rgb.tobytes()

            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
            texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
            self.camera_image.texture = texture

            # Aktualizuj wyniki
            if result_data['detected_objects']:
                # Znajdź obiekt z najwyższą pewnością
                best_object = max(result_data['detected_objects'], key=lambda x: x['confidence'])

                self.prediction_label.text = f"Główny obiekt: {best_object['class_name']}"
                self.confidence_label.text = f"Pewność: {best_object['confidence']:.2%}"
                self.detection_label.text = f"Wykryte obiekty: {result_data['total_detections']}"

                # Kolor w zależności od pewności
                if best_object['confidence'] > 0.8:
                    self.prediction_label.color = (0, 1, 0, 1)  # Zielony
                elif best_object['confidence'] > 0.5:
                    self.prediction_label.color = (1, 1, 0, 1)  # Żółty
                else:
                    self.prediction_label.color = (1, 0, 0, 1)  # Czerwony
            else:
                self.prediction_label.text = "Brak wykrytych obiektów"
                self.confidence_label.text = "Pewność: 0.00%"
                self.detection_label.text = f"Wykryte obiekty: 0"
                self.prediction_label.color = (1, 1, 1, 1)

        except queue.Empty:
            pass
        except Exception as e:
            print(f"Błąd aktualizacji UI: {e}")

    def on_stop(self):
        self.is_running = False
        self.camera_active = False
        if hasattr(self, 'cap'):
            self.cap.release()
        print("Aplikacja zamknięta")


if __name__ == '__main__':
    WasteClassificationApp().run()