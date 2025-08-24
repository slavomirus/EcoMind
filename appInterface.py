import os
import kivy
import numpy as np
import threading
import queue
import cv2
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.graphics.texture import Texture
from kivy.graphics import Color, Line
from kivy.clock import Clock
from ultralytics import YOLO

kivy.require('2.0.0')

YOLO_IMG_SIZE = (640, 480)
CLASS_NAMES = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass', 'metal', 'paper',
               'plastic', 'shoes', 'trash', 'white-glass']


class CameraApp(App):
    def build(self):
        root = BoxLayout(orientation='vertical')

        self.camera_image = Image(size_hint=(1, 0.9))
        root.add_widget(self.camera_image)

        self.info_label = Label(
            text="Wczytuję model...",
            size_hint_y=0.1,
            font_size='24sp',
            bold=True,
            color=(1, 1, 1, 1)
        )
        root.add_widget(self.info_label)

        self.result_queue = queue.Queue()
        self.is_running = True

        Clock.schedule_once(self.load_model, 0.5)
        return root

    def load_model(self, dt):
        try:
            self.yolo_model = YOLO("yolov8n.pt")
            self.info_label.text = "Model załadowany. Detekcja w toku..."

            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise IOError("Nie można otworzyć kamery.")

            # Ustawienie rozdzielczości kamery
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            self.prediction_thread = threading.Thread(target=self.start_prediction_thread, daemon=True)
            self.prediction_thread.start()

            Clock.schedule_interval(self.update_ui, 1.0 / 30.0)
        except Exception as e:
            self.info_label.text = f"BŁĄD: Nie można załadować modelu lub kamery.\n{e}"
            print(f"Błąd ładowania modelu/kamery: {e}")

    def start_prediction_thread(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                print("Nie można odebrać klatki z kamery.")
                continue

            # Konwersja z BGR do RGB dla YOLO
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                # Predykcja
                results = self.yolo_model(frame_rgb, verbose=False)

                # Rysowanie bounding boxów bezpośrednio na klatce
                frame_with_boxes = frame.copy()
                processed_boxes = []
                detected_text = "Nie wykryto obiektów."

                if results and results[0].boxes:
                    detected_text = "Wykryto: "
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0].tolist()]
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])

                        # Rysowanie bounding box na klatce
                        cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{self.yolo_model.names[cls]}: {conf:.2f}"
                        cv2.putText(frame_with_boxes, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        processed_boxes.append({
                            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                            'cls': cls, 'conf': conf,
                            'name': self.yolo_model.names[cls]
                        })

                        detected_text += f"{self.yolo_model.names[cls].capitalize()} ({conf:.2f}), "

                    detected_text = detected_text.rstrip(', ')

                # Konwersja do formatu odpowiedniego dla Kivy
                frame_kivy = np.flip(frame_with_boxes, 0).tobytes()

                # Umieszczamy gotową teksturę i dane w kolejce
                self.result_queue.put({
                    'texture_data': frame_kivy,
                    'boxes': processed_boxes,
                    'text': detected_text,
                    'width': frame_with_boxes.shape[1],
                    'height': frame_with_boxes.shape[0]
                })

            except Exception as e:
                print(f"Błąd w wątku predykcji: {e}")

    def update_ui(self, dt):
        try:
            result_data = self.result_queue.get_nowait()

            # Tworzenie tekstury z danych klatki
            width = result_data['width']
            height = result_data['height']
            texture = Texture.create(size=(width, height), colorfmt='bgr')
            texture.blit_buffer(result_data['texture_data'], colorfmt='bgr', bufferfmt='ubyte')

            # Aktualizacja widżetu Image
            self.camera_image.texture = texture
            self.info_label.text = result_data['text']

        except queue.Empty:
            pass

    def on_stop(self):
        self.is_running = False
        if hasattr(self, 'prediction_thread'):
            self.prediction_thread.join(timeout=1.0)
        if hasattr(self, 'cap'):
            self.cap.release()


if __name__ == '__main__':
    CameraApp().run()