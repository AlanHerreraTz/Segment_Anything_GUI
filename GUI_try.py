import os
import torch
import cv2
from tkinter import Tk, Label, Button, Frame, filedialog, Toplevel, Entry, OptionMenu, StringVar, messagebox,BooleanVar,END
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from PIL import Image, ImageTk
from shutil import copyfile
import numpy as np
import random
import yaml
from ultralytics import YOLO
import datetime
import string
#import keyboard
#import time


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = '/home/alan/gui_sam/sam_vit_l_0b3195.pth'
MODEL_TYPE = 'vit_l'


#UNCOMMENT IF THERE IS A GPU AVAILABLE,
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
#mask_generator = SamAutomaticMaskGenerator(sam)
mask_generator = SamAutomaticMaskGenerator(
   model=sam,
   points_per_side=32,
   pred_iou_thresh=0.98,
   stability_score_thresh=0.98,
   crop_n_layers=1,
   crop_n_points_downscale_factor=2,
   min_mask_region_area=600,  # Requires open-cv to run post-processing
)
mask_predictor = SamPredictor(sam)



class PhotoViewer:
   def __init__(self, master):
       self.master = master
       self.folder = ""
       self.folder_2 = ""
       self.current_index = 0
       self.total_photos = 0
       self.labels_folder = ""
       self.classes = []
       self.drawing = False
       self.ix, self.iy = -1,-1
       self.rect_end = None
       self.seggmented_image = None
       self.saved_masks = []
       self.class_selectors = []
       self.saved_masks_and_classes = []
       self.selected_classes = []
       self.yolo_annotations_general = []
       master.bind("<Right>", self.on_right_arrow)
       master.bind("<Left>", self.on_left_arrow)
       master.bind("<Control_L>", self.ctrl_key)
       master.bind("<Up>", self.up_arrow)


       self.cache_folder = os.path.join(os.getcwd(), "cache")
       os.makedirs(self.cache_folder, exist_ok=True)


       #This is the main frame
       self.image_frame = Frame(master)
       self.image_frame.pack()


       #This is the foto counter
       self.photos_info_label = Label(self.image_frame, text="Images: {}/{}".format(self.current_index + 1, self.total_photos))
       self.photos_info_label.pack(side='top')
      
       #replace when functionality available//left band frame
       self.left_band = Frame(self.image_frame)
       self.left_band.pack(side = 'left')
      
       self.label_left = Label(self.left_band, text ='Label Left', bg = 'green') #example label
       self.label_left.pack(side = 'top', fill = 'both', expand = True)


       #right side frame for start training function


       self.right_band = Frame(self.image_frame)
       self.right_band.pack(side = "right")


       #main image viewer
       self.image_label = Label(self.image_frame)
       self.image_label.pack(side = 'left')
     
       #frame for the class selector
       self.class_frame = Frame(master)
       self.class_frame.pack()


       self.auto_segment_button = Button(self.class_frame, text = "Create Annotations", command = self.save_annotations)
       self.auto_segment_button.pack(side = "left", fill= "both", expand = True)


       self.class_label = Label(self.class_frame, text="Class:")
       self.class_label.pack(side="left", fill= "both", expand= True)


       self.class_selector_var = StringVar(master)
       self.class_selector_var.set("")  # Default value


       self.start_train_button = Button(self.class_frame, text = "Start Training", command = self.start_training)
       self.start_train_button.pack(side = "right", fill = "both", expand = True)


       self.class_selector = OptionMenu(self.class_frame, self.class_selector_var, "")
       self.class_selector.pack(side="left")


       #frame for the buttons of the image viewer
       self.button_frame = Frame(master)
       self.button_frame.pack(side="bottom", fill="x", padx=10, pady=10)


       self.prev_button = Button(self.button_frame, text="Previous", command=self.show_prev, width=15)
       self.prev_button.pack(side="left", expand=True, fill='both')


       self.segment_button = Button(self.button_frame, text="Save Mask", command=self.save_mask)
       self.segment_button.pack(side="left", expand=True, fill='both')


       self.next_button = Button(self.button_frame, text="Next", command=self.show_next, width=15)
       self.next_button.pack(side="right", expand=True, fill='both')


       self.choose_folder()
       self.create_class()


   def choose_folder(self):
      
       self.folder = filedialog.askdirectory()
       if self.folder:
           parent_folder = os.path.dirname(self.folder)  # Obtener el directorio padre
           self.cache_folder = os.path.join(parent_folder, "cache")  # Crear la carpeta cache en el directorio padre
           self.data_folder = os.path.join(parent_folder, "data")  # Crear la carpeta data en el directorio padre
           self.images_folder = os.path.join(self.data_folder, "images")  # Crear la subcarpeta images en la carpeta data
           self.labels_folder = os.path.join(self.data_folder, "labels")  # Crear la subcarpeta labels en la carpeta data
           os.makedirs(self.cache_folder, exist_ok=True)  # Crear la carpeta cache si no existe
           os.makedirs(self.data_folder, exist_ok=True)  # Crear la carpeta data si no existe
           os.makedirs(self.images_folder, exist_ok=True)  # Crear la subcarpeta images si no existe
           os.makedirs(self.labels_folder, exist_ok=True)  # Crear la subcarpeta labels si no existe
           self.total_photos = len(os.listdir(self.folder))
           self.update_photos_info()
           self.show_image()
           #self.create_class()
           self.create_config_file()
           #self.next_2()


   def show_image(self):
       files = sorted(os.listdir(self.folder))
       if files:
           file_path = os.path.join(self.folder, files[self.current_index])
           img = cv2.imread(file_path)
           img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
           img_pil = Image.fromarray(img_rgb)
           self.img_tk = ImageTk.PhotoImage(img_pil)
           self.image_label.config(image=self.img_tk)
           self.image_label.bind("<Button-1>", self.start_rect)
           self.image_label.bind("<B1-Motion>", self.draw_rect)
           self.image_label.bind("<ButtonRelease-1>", lambda event, rgb=img_rgb: self.end_rect(event, rgb))
           self.image_label.bind("<Button-3>", lambda event: self.segment_point(event, event.x, event.y))
           self.update_photos_info()


           #image_save_path = os.path.join(self.images_folder, f"{self.current_index + 1}_.jpg")
           #copyfile(file_path, image_save_path)
           #label_file_path = os.path.join(self.labels_folder, f"{self.current_index + 1}_.txt")
           #open(label_file_path, 'a').close()
           #self.save_image_and_label()


   def show_next(self):
       if self.current_index < self.total_photos - 1:
           self.current_index += 1
           self.show_image()
           self.update_saved_masks_display()
           self.next_2()
       else:
           self.clear_left_label()

   def on_right_arrow(self, event):
       self.show_next()
       self.on_right_arrow_pressed = True

   def show_next_autosegment(self):
       if self.current_index < self.total_photos - 1:
           self.current_index += 1
           #self.show_image()
           #self.update_saved_masks_display()
           #self.next_2()
       else:
           self.clear_left_label()


   def show_prev(self):
       if self.current_index > 0:
           self.current_index -= 1
           self.show_image()
           self.update_saved_masks_display()
       else:
           self.clear_left_label()

   def on_left_arrow(self, master):
       self.show_prev()           

   def clear_left_label(self):
       # Limpiar left_label
       for widget in self.label_left.winfo_children():
           widget.destroy()


   def save_image_and_label(self):
       length = 8
       letters = string.ascii_lowercase
       random_variable = ''.join(random.choice(letters) for _ in range(length))

       timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
       image_save_path = os.path.join(self.images_folder, "{}_{}.jpg".format(random_variable, timestamp))
       original_image_path = os.path.join(self.folder, sorted(os.listdir(self.folder))[self.current_index])
       copyfile(original_image_path, image_save_path)


       label_file_path = os.path.join(self.labels_folder, "{}_{}.txt".format(random_variable, timestamp))
       open(label_file_path, 'a').close()

       return label_file_path


   def create_class(self):
       self.class_creator_window = Toplevel(self.master)
       self.class_creator_window.title("Class Creator")


       self.class_name_label = Label(self.class_creator_window, text="Enter class name:")
       self.class_name_label.pack()


       self.class_name_entry = Entry(self.class_creator_window)
       self.class_name_entry.pack()


       self.confirm_button = Button(self.class_creator_window, text="Confirm", command=self.confirm_class)
       self.confirm_button.pack(side='right')


       self.end_button = Button(self.class_creator_window, text="End", command=self.class_creator_window.destroy)
       self.end_button.pack(side='left')


   def confirm_class(self):
       class_name = self.class_name_entry.get()
       if class_name:
           self.classes.append(class_name)
           self.update_class_selector()
           self.class_name_entry.delete(0, END)


   def update_class_selector(self):
       self.class_selector['menu'].delete(0, 'end')
       for cls in self.classes:
           self.class_selector['menu'].add_command(label=cls, command=lambda value=cls: self.class_selector_var.set(value))


   def update_photos_info(self):
       self.photos_info_label.config(text="Images: {}/{}".format(self.current_index + 1, self.total_photos))


   def start_rect (self, event):
       self.drawing = True
       self.ix, self.iy = event.x, event.y
      
   def draw_rect(self, event):
       if self.drawing:
           img_copy = cv2.cvtColor(cv2.imread(os.path.join(self.folder, sorted(os.listdir(self.folder))[self.current_index])), cv2.COLOR_BGR2RGB)
           cv2.rectangle(img_copy, (self.ix, self.iy), (event.x, event.y), (255,0,0),2)
           self.img_tk = ImageTk.PhotoImage(image = Image.fromarray(img_copy))
           self.image_label.config(image = self.img_tk)

   def next_2 (self):
       current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
       
       image_save_path = os.path.join(self.images_folder, "{}_{}.jpg".format(self.current_index, current_time))
       original_image_path = os.path.join(self.folder, sorted(os.listdir(self.folder))[self.current_index])
       copyfile(original_image_path, image_save_path)

          
   def end_rect(self, event, imagen_rgb):
       self.drawing = False
       self.rect_end = (event.x, event.y)

       # Obtener las coordenadas del rectángulo dibujado
       x1, y1 = self.ix, self.iy
       x2, y2 = self.rect_end

       # Asegurar que x1 sea menor que x2 y y1 sea menor que y2
       x1, x2 = min(x1, x2), max(x1, x2)
       y1, y2 = min(y1, y2), max(y1, y2)

       # Obtener el tamaño de la imagen
       imagen_size = imagen_rgb.shape[:2]
       img_height, img_width = imagen_size

       # Crear el input_box con las coordenadas normalizadas
       input_box = np.array([x1, y1, x2, y2])

       self.input_box = input_box

       self.segmentar_objeto(input_box)
       

      
   def generate_yolo_annotation(self, class_index):
       
       mask_binary = self.mask_binary


       contornos, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


       coordenadas_poligonales = []


       for contorno in contornos:
           poligono = cv2.approxPolyDP(contorno, 0.01 * cv2.arcLength(contorno, True), True)


           for punto in poligono:
               x, y = punto[0]
               coordenadas_norm = (x / mask_binary.shape[1], y / mask_binary.shape[0])
               coordenadas_poligonales.append(coordenadas_norm)


       # Construir la anotación YOLO en una sola cadena
       yolo_annotation = f"{class_index}"
       for coordenada in coordenadas_poligonales:
           yolo_annotation += f" {coordenada[0]:.6f} {coordenada[1]:.6f}"


       print(yolo_annotation)  # Esto imprime la anotación una vez construida correctamente


       return yolo_annotation


   def segmentar_objeto(self, input_box): #(Box segmentation)
        if self.folder:
            files = sorted(os.listdir(self.folder))
            if files:
                file_path = os.path.join(self.folder, files[self.current_index])
                imagen = cv2.imread(file_path)
                #imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

                mask_predictor.set_image(imagen)

                predictions = mask_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box,
                    multimask_output=True,
                )

                masks = predictions[0]
                scores = predictions[1]

                max_score_idx = np.argmax(scores)
                self.max_score_mask = masks[max_score_idx]

                print(input_box)
                #print(self.max_score_mask)

                max_score_mask_int = self.max_score_mask.astype(int)

                mask_binary = (max_score_mask_int * 255).astype(np.uint8)

                _, mask_binary = cv2.threshold(mask_binary, 127, 255, cv2.THRESH_BINARY)

                imagen_pil = Image.fromarray(mask_binary.astype('uint8'))
                imagen_segmentada_tk = ImageTk.PhotoImage(imagen_pil)

                self.segmented_image = imagen_pil
                self.image_label.config(image=imagen_segmentada_tk)
                self.image_label.image = imagen_segmentada_tk

                self.mask_binary = mask_binary

   def segment_point(self, event,x, y):
       
       if self.folder: 
           files = sorted(os.listdir(self.folder))

           if files:
               file_path = os.path.join(self.folder, files[self.current_index])
               imagen = cv2.imread(file_path)
                   
           if event.num == 3:  # Verifica el clic derecho del ratón
               relative_x = x 
               relative_y = y 

               input_point = np.array([[relative_x, relative_y]])

               print(f"Selected point: ({relative_x},{relative_y})")

               mask_predictor.set_image(imagen)

               input_label = np.array([1])

               predictions = mask_predictor.predict(
                   point_coords=input_point,
                   point_labels=input_label,
                   box=None,
                   multimask_output=True,
                   )
               
               masks = predictions[0]
               scores = predictions[1]

               max_score_idx = np.argmax(scores)
               self.max_score_mask = masks[max_score_idx]

               max_score_mask_int = self.max_score_mask.astype(int)

               mask_binary = (max_score_mask_int * 255).astype(np.uint8)

               _, mask_binary = cv2.threshold(mask_binary, 127, 255, cv2.THRESH_BINARY)

               imagen_pil = Image.fromarray(mask_binary.astype('uint8'))

               imagen_segmentada_tk = ImageTk.PhotoImage(imagen_pil)

               self.segmented_image = imagen_pil

               self.image_label.config(image = imagen_segmentada_tk)

               self.image_label.image = imagen_segmentada_tk

               self.mask_binary = mask_binary


   def save_mask(self):
       
       if hasattr(self, 'max_score_mask') and self.mask_binary.any():
           selected_class = self.class_selector_var.get()
           if not selected_class:
               selected_class = "None"


           image_folder = os.path.join(self.cache_folder, f"image_{self.current_index}")
           os.makedirs(image_folder, exist_ok=True)


           mask_save_path = os.path.join(image_folder, f"{selected_class}.png")
           self.segmented_image.save(mask_save_path)


           if selected_class in self.classes:
               # Llamar a generate_yolo_annotation para obtener las anotaciones YOLO
               yolo_annotation = self.generate_yolo_annotation(self.classes.index(selected_class))
               if yolo_annotation:
                   txt_file_path = os.path.join(self.labels_folder, f"{self.current_index}.txt")
                   with open(txt_file_path, 'a') as txt_file:
                       txt_file.write(yolo_annotation + '\n')
               else:
                   print("No object detected.")
           else:
               print("Selected class not found in classes list.")


       self.update_saved_masks_display()
       self.show_image()
 
   def ctrl_key (self, master):
       self.save_mask()
       
   def update_saved_masks_display(self):
       for widget in self.label_left.winfo_children():
           widget.destroy()


       image_folder = os.path.join(self.cache_folder, f"image_{self.current_index}")


       if os.path.exists(image_folder):
           for mask_file in os.listdir(image_folder):
               mask_path = os.path.join(image_folder, mask_file)
               class_name = mask_file.split(".")[0]  # Obtener el nombre de la clase desde el nombre del archivo
               mask_img = Image.open(mask_path)
               mask_img.thumbnail((100, 100))
               mask_tk = ImageTk.PhotoImage(mask_img)
               mask_label = Label(self.label_left, image=mask_tk)
               mask_label.image = mask_tk
               mask_label.pack(side='top')


               class_label = Label(self.label_left, text=f"Class: {class_name}")
               class_label.pack(side='top')


               delete_button = Button(self.label_left, text="Delete", command=lambda mask=mask_path: self.delete_mask(mask))
               delete_button.pack(side='top')
      
   def delete_mask(self, mask_path):
       os.remove(mask_path)
       self.update_saved_masks_display()


   def create_config_file(self):


       config_data = {
           'path': self.data_folder,
           'train': self.images_folder,
           'val': self.images_folder,
       }


       config_file_path = os.path.join(self.cache_folder, "config.yaml")
       with open(config_file_path, 'w') as config_file:
           yaml.dump(config_data, config_file)
           config_file.write("names:\n")
           for i, cls in enumerate(self.classes):
               config_file.write(f"  {i}: {cls}\n")


       return config_file_path
  
   def adjust_contrast(self,image,contrast_factor):
       
       hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
       h, s, v = cv2.split(hsv_image)

       # Ajustar el contraste multiplicando el componente de luminancia (v) por el factor de contraste
       v = cv2.multiply(v, contrast_factor)

       # Asegurarse de que los valores de v estén en el rango permitido (0-255)
       v = np.clip(v, 0, 255)

       adjusted_hsv_image = cv2.merge([h, s, v])

       adjusted_image = cv2.cvtColor(adjusted_hsv_image, cv2.COLOR_HSV2BGR)
      
       return adjusted_image
  
   def apply_contrast_to_data(self, brightness_factor=1.4):
      
       images_path = self.images_folder
      
       for idx, filename in enumerate(os.listdir(images_path)):
          
           if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
              image_path = os.path.join(images_path, filename)
              image = cv2.imread(image_path)
             
              adjusted_image = self.adjust_contrast(image, brightness_factor)
             
              new_image = f"{idx}.1.jpg"
              output_path = os.path.join(images_path, new_image)
              cv2.imwrite(output_path, adjusted_image)
             
       labels_path = self.labels_folder


       for idx, filename in enumerate(sorted(os.listdir(labels_path))):  # Utiliza sorted para asegurar el orden
           if filename.endswith(".txt"):
               with open(os.path.join(labels_path, filename), 'r') as file:
                   content = file.read()


               base_filename, extension = os.path.splitext(filename)  # Separar el nombre del archivo y la extensión
               new_filename = f"{base_filename}.1{extension}"  # Agregar .1 antes de la extensión
               new_file_path = os.path.join(labels_path, new_filename)


               with open(new_file_path, "w") as new_file:
                   new_file.write(content)
 
       print("Brigtness augmented...")


   def up_arrow(self, master):
       self.show_next()
       self.show_image()
       self.auto_segment()


   def auto_segment(self):
       # Obtener la lista de archivos en la carpeta
       files = sorted(os.listdir(self.folder))
       number_of_files = len(files)
       # Iterar sobre cada archivo en la carpeta

       #self.current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
       

       if files:
           
           file_path = os.path.join(self.folder, files[self.current_index])
           imagen = cv2.imread(file_path)

           mask_predictor.set_image(imagen)

           predictions = mask_predictor.predict(
               point_coords=None,
               point_labels=None,
               box=self.input_box,
               multimask_output=True,
           )

           masks = predictions[0]
           scores = predictions[1]

           
           max_score_idx = np.argmax(scores)
           self.max_score_mask = masks[max_score_idx]

           #print(input_box)
           #print(self.max_score_mask)

           max_score_mask_int = self.max_score_mask.astype(int)

           mask_binary = (max_score_mask_int * 255).astype(np.uint8)

           _, mask_binary = cv2.threshold(mask_binary, 127, 255, cv2.THRESH_BINARY)

           imagen_pil = Image.fromarray(mask_binary.astype('uint8'))
           imagen_segmentada_tk = ImageTk.PhotoImage(imagen_pil)

           self.segmented_image = imagen_pil
           self.image_label.config(image=imagen_segmentada_tk)
           self.image_label.image = imagen_segmentada_tk

           self.mask_binary = mask_binary

           contornos, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

           coordenadas_poligonales = []

           for contorno in contornos:
               poligono = cv2.approxPolyDP(contorno, 0.01 * cv2.arcLength(contorno, True), True)


               for punto in poligono:
                   x,y = punto [0]
                   coordenadas_norm = (x/mask_binary.shape[1], y / mask_binary.shape[0])
                   coordenadas_poligonales.append(coordenadas_norm)


           if hasattr(self, 'max_score_mask') and mask_binary.any():
               selected_class = self.class_selector_var.get()
               if not selected_class:
                   selected_class = "None"


               if mask_binary is not None:
                    print("Saving mask_binary to cache folder...")
                    #image_folder = os.path.join(self.cache_folder, f"{count}_{selected_class}_{time}")
                    #os.makedirs(image_folder, exist_ok=True)

                    mask_save_path = os.path.join(self.cache_folder, f"{self.current_index}_{selected_class}{selected_class}.png")
                    cv2.imwrite(mask_save_path, mask_binary)
               else:
                   print("Error: mask_binary is NONE")

           if selected_class in self.classes:
                   yolo_annotation = f"{self.classes.index(selected_class)}"
                   for coordenada in coordenadas_poligonales:
                       yolo_annotation += f" {coordenada[0]:.6f} {coordenada[1]:.6f}"

           print(yolo_annotation)

           self.yolo_annotations_general.append(yolo_annotation)



   def save_annotations(self):
       
       # Obtener la lista de archivos en la carpeta de imágenes
        image_files = os.listdir(self.images_folder)
        
        # Ordenar la lista de archivos para asegurar que se procesen en orden
        image_files.sort()

        for image_file in image_files:
            # Obtener el nombre del archivo sin la extensión
            image_name, _ = os.path.splitext(image_file)

            # Crear el nombre del archivo de texto en la carpeta de etiquetas
            label_file = os.path.join(self.labels_folder, image_name + ".txt")

            # Obtener el primer elemento de self.yolo_annotations_general
            if self.yolo_annotations_general:
                yolo_annotation = self.yolo_annotations_general.pop(0)

                # Escribir el yolo_annotation en el archivo de texto
                with open(label_file, "w") as f:
                    f.write(yolo_annotation)

        print("Anotaciones guardadas con éxito.")
       
      
   def start_training(self):
      
       #Augmenting brightness on all images
       #brightness_contrast = 1.4
       #self.apply_contrast_to_data(brightness_factor)
      
       #Path were .yaml will be stored
       config_file_path = self.create_config_file()

       model = YOLO("yolov8n.yaml")

       result = model.train(data = config_file_path, epochs = 100)
      
       messagebox.showinfo("Training Finished")
      
def main():
   root = Tk()
   root.title("Image Labeler")

   app = PhotoViewer(root)

   root.mainloop()

if __name__ == "__main__":
   main()
