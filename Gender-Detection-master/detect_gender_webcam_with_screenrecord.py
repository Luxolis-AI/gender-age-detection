import pyautogui
import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import threading
import time
import queue
from datetime import datetime
from io import BytesIO
import cairosvg
from tensorflow.keras.models import load_model
import tensorflow as tf
import cvlib as cv
from tensorflow.keras.preprocessing.image import img_to_array

class VideoUploader(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Luxolis Human Detection")
        self.geometry("1920x1080")
        self.configure(bg="#000A1E")

        self.load_logo('/home/mushtariy/Desktop/KT/logo.svg')

        button_width = 60
        button_height = 60

        self.record_img = ImageTk.PhotoImage(Image.open("/home/mushtariy/Desktop/KT/Gender-Detection-master/record_button.png").resize((button_width, button_height), Image.LANCZOS))
        self.start_img = ImageTk.PhotoImage(Image.open("/home/mushtariy/Desktop/KT/Gender-Detection-master/play_button1.png").resize((button_width, button_height), Image.LANCZOS))
        self.stop_img = ImageTk.PhotoImage(Image.open("/home/mushtariy/Desktop/KT/Gender-Detection-master/stop_button.png").resize((button_width, button_height), Image.LANCZOS))
        self.download_img = ImageTk.PhotoImage(Image.open("/home/mushtariy/Desktop/KT/Gender-Detection-master/download_button.png").resize((button_width, button_height), Image.LANCZOS))
        self.table_img = ImageTk.PhotoImage(Image.open("/home/mushtariy/Desktop/KT/Gender-Detection-master/report.png").resize((button_width, button_height), Image.LANCZOS))
        self.graph_img = ImageTk.PhotoImage(Image.open("/home/mushtariy/Desktop/KT/Gender-Detection-master/graph_button.png").resize((button_width, button_height), Image.LANCZOS))

        self.start_button = tk.Button(self, image=self.start_img, command=self.start_webcams, bg="#000000", fg="#FFFFFF", width=button_width, height=button_height)
        self.start_button.place(x=40, y=930)

        self.download_button = tk.Button(self, image=self.download_img, command=self.download_csv, bg="#FFFFFF", fg="#000000", width=button_width, height=button_height)
        self.download_button.place(x=120, y=930)

        self.record_button = tk.Button(self, image=self.record_img, command=self.toggle_recording, bg="#FFFFFF", fg="#000000", width=button_width, height=button_height)
        self.record_button.place(x=200, y=930)

        self.stop_button = tk.Button(self, image=self.stop_img, command=self.stop_current_videos, bg="#000000", fg="#FFFFFF", width=button_width, height=button_height)
        self.stop_button.place(x=280, y=930)

        self.graph_button = tk.Button(self, image=self.graph_img, command=self.show_graph, bg="#000000", fg="#FFFFFF", width=button_width, height=button_height)
        self.graph_button.place(x=360, y=930)

        self.canvas_width = 600
        self.canvas_height = 400
        self.canvas1 = self.create_canvas(20, 110)
        self.canvas2 = self.create_canvas(620, 110)
        self.canvas3 = self.create_canvas(1220, 110)
        self.canvas4 = self.create_canvas(20, 510)
        self.canvas5 = self.create_canvas(620, 510)
        self.canvas6 = self.create_canvas(1220, 510)

        self.gender_model = load_model('gender_detection.keras')

        self.age_model = tf.lite.Interpreter(model_path="/home/mushtariy/Desktop/KT/Gender-Detection-master/AgeClass_best_06_02-16-02.tflite")
        self.age_model.allocate_tensors()
        self.input_details_age = self.age_model.get_input_details()
        self.output_details_age = self.age_model.get_output_details()

        self.caps = [None, None]
        self.is_playing = False
        self.is_recording = False
        self.detections = []  # List to hold detection data
        self.person_id = 1
        self.detected_faces = []
        self.video_writer = None
        self.recording_index = 1
        self.recording_thread = None
        self.frame_queue = queue.Queue(maxsize=10)

    def create_canvas(self, x, y):
        canvas = tk.Canvas(self, width=self.canvas_width, height=self.canvas_height, bg="#000A1E")
        canvas.place(x=x, y=y)
        return canvas

    def start_webcams(self):
        print("Starting webcams...")
        self.caps[0] = cv2.VideoCapture(0)
        self.caps[1] = cv2.VideoCapture(1)
        for i in range(2):
            if not self.caps[i].isOpened():
                self.caps[i].open(i)
        if not any(cap.isOpened() for cap in self.caps):
            messagebox.showerror("Webcam Error", "Failed to open webcams.")
        else:
            self.is_playing = True
            for i in range(2):
                threading.Thread(target=self.update_frame, args=(i,)).start()

    def update_frame(self, cam_idx):
        while self.is_playing:
            ret, frame = self.caps[cam_idx].read()
            if not ret:
                break
            self.process_frame(frame, cam_idx)

    def process_frame(self, frame, cam_idx):
        faces, confidence = cv.detect_face(frame)
        for idx, f in enumerate(faces):
            (startX, startY, endX, endY) = f[0], f[1], f[2], f[3]
            face_crop = np.copy(frame[startY:endY, startX:endX])
            if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                continue

            face_crop_gender = cv2.resize(face_crop, (96, 96))
            face_crop_gender = face_crop_gender.astype("float") / 255.0
            face_crop_gender = img_to_array(face_crop_gender)
            face_crop_gender = np.expand_dims(face_crop_gender, axis=0)

            face_crop_age = cv2.resize(face_crop, (224, 224))
            face_crop_age = face_crop_age.astype("float") / 255.0
            face_crop_age = img_to_array(face_crop_age)
            face_crop_age = np.expand_dims(face_crop_age, axis=0)

            gender_conf = self.gender_model.predict(face_crop_gender)[0]
            gender_idx = np.argmax(gender_conf)
            gender_label = 'woman' if gender_idx == 1 else 'man'

            self.age_model.set_tensor(self.input_details_age[0]['index'], face_crop_age.astype('float32'))
            self.age_model.invoke()
            age_output = self.age_model.get_tensor(self.output_details_age[0]['index'])
            age_idx = np.argmax(age_output)
            age_label = ['04 - 06', '07 - 08', '09 - 11', '12 - 19', '20 - 27', '28 - 35', '36 - 45', '46 - 60', '61 - 75'][age_idx]

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if self.is_new_face(f):
                self.detected_faces.append(f)
                self.detections.append([timestamp, self.person_id, gender_label, age_label])
                self.person_id += 1

            label = f"{gender_label}, {age_label}"
            Y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        
        frame = self.resize_to_canvas(frame)
        self.schedule_image_update(frame, cam_idx)

    def is_new_face(self, face):
        threshold = 50
        for detected_face in self.detected_faces:
            if (abs(face[0] - detected_face[0]) < threshold and
                abs(face[1] - detected_face[1]) < threshold and
                abs(face[2] - detected_face[2]) < threshold and
                abs(face[3] - detected_face[3]) < threshold):
                return False
        return True

    def schedule_image_update(self, frame, cam_idx):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.after(0, self.update_canvas, imgtk, cam_idx)
        if self.is_recording and not self.frame_queue.full():
            self.frame_queue.put(frame)

    def update_canvas(self, imgtk, cam_idx):
        canvas = [self.canvas1, self.canvas2, self.canvas3, self.canvas4, self.canvas5, self.canvas6][cam_idx]
        canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        canvas.image = imgtk

    def stop_current_videos(self):
        self.is_playing = False
        self.is_recording = False
        for cap in self.caps:
            if cap:
                cap.release()
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join()

    def load_logo(self, logo_path):
        output = BytesIO()
        cairosvg.svg2png(url=logo_path, write_to=output)
        logo_image = Image.open(output)
        logo_image = logo_image.resize((280, 80), Image.LANCZOS)
        logo_tk = ImageTk.PhotoImage(logo_image)
        logo_label = tk.Label(self, image=logo_tk, bg="#000A1E")
        logo_label.image = logo_tk
        logo_label.place(x=25, y=15)

    def center_canvas(self, event=None):
        window_width = self.winfo_width()
        window_height = self.winfo_height()
        x_offset = (window_width - 3 * self.canvas_width) // 4
        y_offset = (window_height - 2 * self.canvas_height) // 3 + 100
        self.canvas1.place(x=x_offset, y=y_offset - 50)
        self.canvas2.place(x=x_offset * 2 + self.canvas_width, y=y_offset - 50)
        self.canvas3.place(x=x_offset * 3 + self.canvas_width * 2, y=y_offset - 50)
        self.canvas4.place(x=x_offset, y=y_offset * 2 + self.canvas_height - 50)
        self.canvas5.place(x=x_offset * 2 + self.canvas_width, y=y_offset * 2 + self.canvas_height - 50)
        self.canvas6.place(x=x_offset * 3 + self.canvas_width * 2, y=y_offset * 2 + self.canvas_height - 50)

    def resize_to_canvas(self, frame):
        frame_height, frame_width = frame.shape[:2]
        aspect_ratio_frame = frame_width / frame_height
        aspect_ratio_canvas = self.canvas_width / self.canvas_height
        if aspect_ratio_frame > aspect_ratio_canvas:
            new_width = self.canvas_width
            new_height = int(new_width / aspect_ratio_frame)
        else:
            new_height = self.canvas_height
            new_width = int(new_height * aspect_ratio_frame)
        return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

    def toggle_recording(self):
        if self.is_recording:
            self.is_recording = False
            if self.recording_thread and self.recording_thread.is_alive():
                self.recording_thread.join()
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.recording_index += 1
            messagebox.showinfo("Info", "Recording stopped and saved.")
        else:
            self.is_recording = True
            filename = f"Recording{self.recording_index}.avi"
            fourcc = cv2.VideoWriter_fourcc(*'avi')
            fps = 20.0
            frame_size = (self.winfo_width(), self.winfo_height())
            self.video_writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
            if not self.video_writer.isOpened():
                print("Failed to open video writer")
                return
            print(f"Started recording to {filename}")
            self.recording_thread = threading.Thread(target=self.record_screen)
            self.recording_thread.start()
            messagebox.showinfo("Info", "Recording started.")

    def record_screen(self):
        while self.is_recording:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                self.video_writer.write(frame)
            time.sleep(1 / 20)  # Control frame rate

    def download_csv(self):
        if not self.detections:
            messagebox.showinfo("Info", "No data to export")
            return

        filepath = filedialog.asksaveasfilename(defaultextension=".csv",
                                                filetypes=[("CSV files", "*.csv")],
                                                title="Save file")
        if not filepath:
            return

        with open(filepath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Person ID", "Gender", "Age"])
            writer.writerows(self.detections)
        messagebox.showinfo("Success", "Data exported successfully")

    def show_table(self):
        table_window = tk.Toplevel(self)
        table_window.title("Detection Table")
        table_window.geometry("800x600")
        table_window.configure(bg="#000A1E")

        table = tk.Text(table_window, bg="#000A1E", fg="#FFFFFF", font=("Arial", 12))
        table.pack(fill=tk.BOTH, expand=True)

        table.insert(tk.END, f"{'Timestamp':<25} {'Person ID':<10} {'Gender':<10} {'Age':<10}\n")
        table.insert(tk.END, "-" * 60 + "\n")

        for detection in self.detections:
            table.insert(tk.END, f"{detection[0]:<25} {detection[1]:<10} {detection[2]:<10} {detection[3]:<10}\n")
        table.configure(state='disabled')

    # Update show_graph to call the function from graphs.py
    def show_graph(self):
        show_graph(self.detections)

if __name__ == "__main__":
    app = VideoUploader()
    app.mainloop()



