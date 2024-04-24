import cv2
import os

def convertir_video_a_imagenes(video_path, output_folder, nombre_base, cada_n_fotogramas=2, resize_width=1280, resize_height=720):
   
    video_capture = cv2.VideoCapture(video_path)
    
    if not video_capture.isOpened():
        print("Error al abrir el video.")
        return
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    count = 0
    frame_count = 0
    
    while True:
        ret, frame = video_capture.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count % cada_n_fotogramas == 0:
            
            resized_frame = cv2.resize(frame, (resize_width, resize_height))
            
            output_path = os.path.join(output_folder, f"{nombre_base}_{count:04d}.jpg")
            cv2.imwrite(output_path, resized_frame)
            
            count += 1
    
    video_capture.release()
    print("Conversion completa.")

video_path = r"/home/alan/gui_sam/Camera Roll/rubik_cube.mp4"
output_folder = r"/home/alan/gui_sam/objects/rubik_cube"
nombre_base = "imagen"
convertir_video_a_imagenes(video_path, output_folder, nombre_base, cada_n_fotogramas=4)

