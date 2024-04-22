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


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = '/home/alan/gui_sam/sam_vit_b_01ec64.pth'
MODEL_TYPE = 'vit_b'



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
       self.yolo_annotation_bbox_general = []
       self.yolo_annotation_segmentation_dict = {}
       self.yolo_annotation_bbox_dict = {}
       master.bind("<Right>", self.on_right_arrow)
       master.bind("<Left>", self.on_left_arrow)
       master.bind("<Control_L>", self.ctrl_key)
       master.bind("<Up>", self.up_arrow)
       master.bind("<Shift_R>", self.shift_point_segment)
       self.current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

       self.cache_folder = os.path.join(os.getcwd(), "cache")
       os.makedirs(self.cache_folder, exist_ok=True)

       #This is the main frame, allows the visualization of images on the GUI
       self.image_frame = Frame(master)
       self.image_frame.pack()

       #This is the foto counter, indicates the number of images
       self.photos_info_label = Label(self.image_frame, text="Images: {}/{}".format(self.current_index + 1, self.total_photos))
       self.photos_info_label.pack(side='top')
      
       #Left frame label, placeholder for saved masks
       self.left_band = Frame(self.image_frame)
       self.left_band.pack(side = 'left')
      
       self.label_left = Label(self.left_band, text ='Label Left', bg = 'green') #example label
       self.label_left.pack(side = 'top', fill = 'both', expand = True)

       #Right side frame, placeholder for start training button
       self.right_band = Frame(self.image_frame)
       self.right_band.pack(side = "right")

       #Main image viewer tkinter label
       self.image_label = Label(self.image_frame)
       self.image_label.pack(side = 'left')
     
       #Frame for the class selector
       self.class_frame = Frame(master)
       self.class_frame.pack()
       
       #Buttons and frames for the app
       self.auto_segment_button = Button(self.class_frame, text = "Create Annotations", command = self.save_annotations)
       self.auto_segment_button.pack(side = "left", fill= "both", expand = True)

       self.class_label = Label(self.class_frame, text="Class:")
       self.class_label.pack(side="left", fill= "both", expand= True)

       self.class_selector_var = StringVar(master)
       self.class_selector_var.set("") 

       self.start_train_button = Button(self.class_frame, text = "Start Training", command = self.start_training)
       self.start_train_button.pack(side = "right", fill = "both", expand = True)

       self.start_train_button = Button(self.class_frame, text = "Show Image", command = self.show_image)
       self.start_train_button.pack(side = "right", fill = "both", expand = True)

       self.class_selector = OptionMenu(self.class_frame, self.class_selector_var, "")
       self.class_selector.pack(side="left")

       self.button_frame = Frame(master)
       self.button_frame.pack(side="bottom", fill="x", padx=10, pady=10)

       self.prev_button = Button(self.button_frame, text="Previous", command=self.show_prev, width=15)
       self.prev_button.pack(side="left", expand=True, fill='both')

       self.segment_button = Button(self.button_frame, text="Save Mask", command=self.save_mask)
       self.segment_button.pack(side="left", expand=True, fill='both')

       self.next_button = Button(self.button_frame, text="Next", command=self.show_next, width=15)
       self.next_button.pack(side="right", expand=True, fill='both')
       
       #Calling the functions required to start the GUI
       self.choose_folder()
       self.create_class()

   def choose_folder(self):
       """
       In this main function, the system folder selection window is loaded, if the folder 
       is chosen the directory path is saved and based on it the main folders cache and 
       data are created, within data 3 folders are created: images , labels, labels_bbox 
       and their paths are saved in global variables respectively.
       Each time this function is executed, it will be checked to see if these directories 
       already exist; if they exist previously, they will not be replaced.
       """
       self.folder = filedialog.askdirectory()
       if self.folder:
           parent_folder = os.path.dirname(self.folder)  #Getting the parent folder (The one that is selected)
           self.cache_folder = os.path.join(parent_folder, "cache")  #Creating cache folder, in where the segmentation masks will be stored for processing and visualization
           self.data_folder = os.path.join(parent_folder, "data")  #Creating the data folder, that is the base folder in order to train with yolo
           self.images_folder = os.path.join(self.data_folder, "images")  #Creating the subfolder images, where copy of the images will be stored
           self.labels_folder = os.path.join(self.data_folder, "labels")  #Creating the subfolder labels, where the segmentation annotations will be stored
           self.labels_bbox_folder = os.path.join(self.data_folder, "labels_bbox") #Creating the subfolder labels_bbox, where the bounding box annotation will be stored.
           os.makedirs(self.cache_folder, exist_ok=True)  #Reviewing if the folders exists
           os.makedirs(self.data_folder, exist_ok=True)   
           os.makedirs(self.images_folder, exist_ok=True)  
           os.makedirs(self.labels_folder, exist_ok=True)  
           os.makedirs(self.labels_bbox_folder, exist_ok=True) 
           self.total_photos = len(os.listdir(self.folder)) 
           self.update_photos_info()
           self.show_image()
           self.create_config_file()

   def show_image(self):
       """ 
       In this main function, the path of the folder selected by the user is taken, 
       if the path is valid, the path of the raw image will be constructed, subsequently 
       the necessary conversions of the image are made so that it can be transformed into 
       an image to work in tkinter (it is read with open cv, converted to RGB scale 
       and loaded in tkinter format).
       When doing this conversion, the image is displayed on the screen for the user.
       Finally, the possible operations with which the user can interact with the 
       image when using mouse clicks and the functions associated with them are configured.
       """
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


   def show_next(self):
       """
       This function is used to update the photo counter and to update the image being displayed.
       When this function is called the image counter will add up to one and the next 
       image from the selected folder will be displayed. 
       """
       if self.current_index < self.total_photos - 1:
           self.current_index += 1
           self.show_image()
           self.update_saved_masks_display()
           
       else:
           self.clear_left_label()
           
           
   def on_right_arrow(self, event):
       """
       This function allows the activation of the show_next function, 
       by using the right navigation arrow on the keyboard.
       """
       self.show_next()
       

   def show_prev(self):
       """
       This function subtracts the photo counter variable by one and 
       displays the previous photo in the selected folder, updating the photo counter
       and the image that is displayed.
       """
       if self.current_index > 0:
           self.current_index -= 1
           self.show_image()
           self.update_saved_masks_display()
       else:
           self.clear_left_label()
           

   def on_left_arrow(self, master):
       """
       This function allows the activation of the show prev function 
       by pressing the left navigation key on the keyboard.
       """
       self.show_prev()
                  

   def clear_left_label(self):
       """ 
       This function deletes the widget associated with the created mask, these masks are 
       created when the instance of many annotations is used in an image, 
       the selected mask is cleared.
       """
       for widget in self.label_left.winfo_children():
           widget.destroy()


   def save_image_and_label(self):
       """ 
       This function is used to create tag files with unique names, creating a 
       random variable of 8 characters, getting the current time tag into a variable.
       Once these unique variables are obtained, the name of the .jpg files 
       (copy of the current image displayed) and the respective .txt label where 
       the yolo annotations will be saved are constructed.
       At the moment this function is not in use.
       """
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
       """ 
       This main function is executed at the beginning, allowing the creation of the possible 
       classes in the class selector.
       The class creation window is created by accessing the tkinter window creation instance, 
       with two buttons, the confirmation button that confirms the creation of the class and 
       the end button, which ends the window (These buttons are created as tkinter instance).
       """
       self.class_creator_window = Toplevel(self.master)
       self.class_creator_window.title("Class Creator")

       self.class_name_label = Label(self.class_creator_window, text="Enter classes names:")
       self.class_name_label.pack()

       self.class_name_entry = Entry(self.class_creator_window)
       self.class_name_entry.pack()

       self.confirm_button = Button(self.class_creator_window, text="Confirm", command=self.confirm_class)
       self.confirm_button.pack(side='right')

       self.end_button = Button(self.class_creator_window, text="End", command=self.class_creator_window.destroy)
       self.end_button.pack(side='left')


   def confirm_class(self):
       """ 
       This function accesses the class creation window, being the function associated with 
       the confirmation button, the function saves the characters of the created classes, 
       these new classes are written in the fill_bar of the app. window, also allows several 
       classes to be written in the same message by dividing them using the ',' character, 
       confirming their creation by pressing the associated button.
       """
       classes_text = self.class_name_entry.get()
       if classes_text:
            classes_list = [cls.strip() for cls in classes_text.split(',')]  # Separar las clases por comas
            self.classes.extend(classes_list)
            self.update_class_selector()
            self.class_name_entry.delete(0, END)


   def update_class_selector(self):
       """ 
       This function accesses the information from the class creation window, updating 
       the list of available classes after they are confirmed in the aforementioned window. 
       Each class is assigned a value of index 0, in order to be able to build the classes,
       YOLO annotations based on this value.
       """
       self.class_selector['menu'].delete(0, 'end')
       for cls in self.classes:
           self.class_selector['menu'].add_command(label=cls, command=lambda value=cls: self.class_selector_var.set(value))


   def update_photos_info(self):
       """ 
       This function updates the photo counter displayed in the app, based on the total 
       number of images in the directory selected by the user and the 
       current number of photos being displayed.
       """
       self.photos_info_label.config(text="Images: {}/{}".format(self.current_index + 1, self.total_photos))


   def start_rect (self, event):
       """ 
       This function is the initiator of one of the possible actions to be performed 
       using the mouse on the images shown in the GUI, this function detects when 
       the event of using the left click of the mouse on the image is started, 
       changing the state to true, finally Saves the x, y coordinates 
       where the mouse click started, for later use.
       """
       self.drawing = True
       self.ix, self.iy = event.x, event.y
       
      
   def draw_rect(self, event):
       """
       This function, after starting the rectangle drawing event, visually draws 
       in the tkinter image frame a rectangle according to the user's drawing using 
       opencv functions, visually showing the stroke made with the mouse.
       """
       if self.drawing:
           img_copy = cv2.cvtColor(cv2.imread(os.path.join(self.folder, sorted(os.listdir(self.folder))[self.current_index])), cv2.COLOR_BGR2RGB)
           cv2.rectangle(img_copy, (self.ix, self.iy), (event.x, event.y), (255,0,0),2)
           self.img_tk = ImageTk.PhotoImage(image = Image.fromarray(img_copy))
           self.image_label.config(image = self.img_tk)

          
   def end_rect(self, event, imagen_rgb):
       """ 
       This function saves the final mouse position values, when the left mouse
       click is no longer pressed, thus achieving, together with the initial values, 
       two points with which to obtain the coordinates with respect to the image
       of the rectangle drawn by the user.
       Some operations are done to obtain the matrix of these coordinates with respect 
       to the size of the image, saving this result in the global variable self.input_box, 
       finally the segment_object function is called 
       giving it the resulting rectangle as an argument.
       """
       self.drawing = False
       self.rect_end = (event.x, event.y)

       x1, y1 = self.ix, self.iy
       x2, y2 = self.rect_end
       x1, x2 = min(x1, x2), max(x1, x2)
       y1, y2 = min(y1, y2), max(y1, y2)

       imagen_size = imagen_rgb.shape[:2]
       img_height, img_width = imagen_size

       input_box = np.array([x1, y1, x2, y2])

       self.input_box = input_box
       self.segment_bbox(input_box)
       

   def generate_many_yolo_annotation(self, class_index):
       """ 
       This function generates the yolo annotation when multiple masks are drawn in the image, 
       it needs the class_index of the possible classes as an argument to build the annotation.
       In the logic, it first accesses the global variable self.mask_binary in which the result 
       of the last segmentation carried out is stored, the existing contours are drawn in it 
       and the list is created where the annotations will be saved.
       Using a for loop, the polygonal coordinates of the contours detected in the mask are obtained 
       using opencv functions that specify their position every certain distance and subsequently 
       the annotation is built in the list in the yolo format, obtaining the size of the image in a 
       range from 1 to 0 and making the coordinate annotations based on this information. At the end 
       of this annotation, the value of the index class is added to the beginning.
       Finally the function returns the annotation.
       """
       mask_binary = self.mask_binary
       contornos, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       coordenadas_poligonales = []

       for contorno in contornos:
           poligono = cv2.approxPolyDP(contorno, 0.01 * cv2.arcLength(contorno, True), True)
           for punto in poligono:
               x, y = punto[0]
               coordenadas_norm = (x / mask_binary.shape[1], y / mask_binary.shape[0])
               coordenadas_poligonales.append(coordenadas_norm)

       yolo_annotation = f"{class_index}"
       for coordenada in coordenadas_poligonales:
           yolo_annotation += f" {coordenada[0]:.6f} {coordenada[1]:.6f}"

       print(yolo_annotation) 
       return yolo_annotation


   def segment_bbox(self, input_box): #(Box segmentation)
        """ 
        This main function of the app configures the necessary parameters to access the 
        Segment Anything Model mask predictor and generates as a result the mask of interest 
        selected by the region bounding box marked by the user.
        The path to the image shown in the viewer is accessed, the predictor is loaded in this 
        image and then the area where the prediction of the mask of interest will be performed 
        is configured, the results of the prediction are saved and only we keep the mask with 
        the highest prediction score, the binary image that remains as a result is converted 
        to an image in opencv format by applying a treshold for viewing, applying bitwise operations 
        we only keep the area where the segmentation was generated and superimpose it on the original 
        image, converting this image to one compatible with the tkinter viewer so that the user can 
        review it, the result is displayed in the viewer and the binary mask of 
        interest is returned for further processing.
        """
        if self.folder:
            files = sorted(os.listdir(self.folder))
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

                max_score_mask_int = self.max_score_mask.astype(int)
                mask_binary = (max_score_mask_int * 255).astype(np.uint8)
                _, mask_binary = cv2.threshold(mask_binary, 127, 255, cv2.THRESH_BINARY)
                
                inverted_mask = cv2.bitwise_not(mask_binary)
                superimposed_mask = cv2.bitwise_and(imagen, imagen, mask= inverted_mask)

                imagen_pil = Image.fromarray(superimposed_mask.astype('uint8'))
                imagen_segmentada_tk = ImageTk.PhotoImage(imagen_pil)

                self.segmented_image = imagen_pil
                self.image_label.config(image=imagen_segmentada_tk)
                self.image_label.image = imagen_segmentada_tk

                self.mask_binary = mask_binary
                
                return mask_binary
            
                              
   def segment_point(self, event,x, y):
       """ 
        This main function of the app configures the necessary parameters to access the 
        Segment Anything Model mask predictor and generates as a result the mask of interest 
        selected by the region of the mouse point marked by the user.
        The path to the image shown in the viewer is accessed, the predictor is loaded in this 
        image and then the area where the prediction of the mask of interest will be performed 
        is configured, the results of the prediction are saved and only we keep the mask with 
        the highest prediction score, the binary image that remains as a result is converted 
        to an image in opencv format by applying a treshold for viewing, applying bitwise operations 
        we only keep the area where the segmentation was generated and superimpose it on the original 
        image, converting this image to one compatible with the tkinter viewer so that the user can 
        review it, the result is displayed in the viewer and the binary mask of 
        interest is returned for further processing.
        """   
       if self.folder: 
           files = sorted(os.listdir(self.folder))

           if files:
               file_path = os.path.join(self.folder, files[self.current_index])
               imagen = cv2.imread(file_path)
                   
           if event.num == 3: 
               relative_x = x 
               relative_y = y 

               input_point = np.array([[relative_x, relative_y]])

               self.input_point = input_point

               mask_predictor.set_image(imagen)

               input_label = np.array([1])

               predictions = mask_predictor.predict(
                   point_coords=self.input_point,
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
               
               inverted_mask = cv2.bitwise_not(mask_binary)
               superimposed_mask = cv2.bitwise_and(imagen, imagen, mask= inverted_mask)
            
               imagen_pil = Image.fromarray(superimposed_mask.astype('uint8'))
               imagen_segmentada_tk = ImageTk.PhotoImage(imagen_pil)
               self.segmented_image = imagen_pil

               self.image_label.config(image = imagen_segmentada_tk)
               self.image_label.image = imagen_segmentada_tk
               self.mask_binary = mask_binary
               
               return mask_binary

               
   def save_mask(self):
       """ 
       This function checks if the object has max_socre_mask and mask_binary attributes defined, 
       which ensures that a segmentation has been performed before, obtains the index of the 
       selected class to save the mask, creates a folder for each image to save the 
       segmentations and save the resulting masks there.
       Make a copy of the current image to work on to comply with the yolo format, if the object 
       class was selected, generate the YOLO annotations using find countours functions and polygon 
       approximation with opencv, save the annotations in the text file . specified with the same 
       name as the image, the original image is finally displayed again to continue working on it. 
       """
       if hasattr(self, 'max_score_mask') and self.mask_binary.any():
           selected_class = self.class_selector_var.get()
           if not selected_class:
               selected_class = "None"

           image_folder = os.path.join(self.cache_folder, f"image_{self.current_index}")
           os.makedirs(image_folder, exist_ok=True)
           
           files = sorted(os.listdir(self.folder))
           image_file_path = os.path.join(self.folder, files[self.current_index])
           copy_image_path = os.path.join(self.images_folder, f"{self.current_index}_{self.current_time}")
           
           if not os.path.exists(copy_image_path):
               copyfile(image_file_path, copy_image_path)
           
           mask_save_path = os.path.join(image_folder, f"{selected_class}.png")
           self.segmented_image.save(mask_save_path)

           if selected_class in self.classes:
               # Llamar a generate_yolo_annotation para obtener las anotaciones YOLO
               yolo_annotation = self.generate_many_yolo_annotation(self.classes.index(selected_class))
               if yolo_annotation:
                   txt_file_path = os.path.join(self.labels_folder, f"{self.current_index}_{self.current_time}.txt")
                   with open(txt_file_path, 'a') as txt_file:
                       txt_file.write(yolo_annotation + '\n')
               else:
                   print("No object detected.")
           else:
               print("Selected class not found in classes list.")

       self.update_saved_masks_display()
       self.show_image()
       
 
   def ctrl_key (self, master):
       """ 
       This function allows the GUI to activate the save_mask function 
       by pressing the left control key on the keyboard.
       """
       self.save_mask()
       
             
   def update_saved_masks_display(self):
       """ 
       This function, every time a mask is created, updates the information 
       about the created masks in the left label of the GUI, displaying them 
       and making the created annotations visible for the user's reference.
       Creates a folder for each annotation created by image and within 
       that folder store the result of the successful segmentations, these 
       images are resized to a 100*100 format and are all displayed in the left label 
       of the app, updating this reel of slicers every time a new one is created.
       """
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
       """ 
       This function associated with a button for each mask created in the 
       update_saved_masks_display function, allows you to delete the desired 
       mask to be deleted in the left label, this with the purpose that the 
       user can delete the masks and annotations that he no longer wants, 
       however for the moment This function does not delete the associated annotation.
       """
       os.remove(mask_path)
       self.update_saved_masks_display()
       

   def create_config_file(self):
       """ 
       This function is responsible for creating the .yaml file necessary to start training YOLO models.
       As a first point, the creation of the data.yaml follows the necessary format of YOLO training 
       instructions, containing and writing the directories in which the images and labels are located 
       in the appropriate format, accessing the variables that store the route information of the necessary 
       folders (path: general folder, train: training images folder, val: validation images folder).
       Subsequently, the names and indexes of the classes created in order are obtained and written 
       to the configuration file created in order to correspond to the annotation information.
       As a last point, the path of the file created for later use is returned.
       """
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
       """ 
       This function describes the logic to apply a contrast multiplication 
       to the selected image, making use of OpenCV functions and providing 
       the contrast value to be processed as an argument to the function, 
       the function is capable of applying these changes to the provided image.
       """
       hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
       h, s, v = cv2.split(hsv_image)
       v = cv2.multiply(v, contrast_factor)
       v = np.clip(v, 0, 255)

       adjusted_hsv_image = cv2.merge([h, s, v])
       adjusted_image = cv2.cvtColor(adjusted_hsv_image, cv2.COLOR_HSV2BGR)
      
       return adjusted_image
   
  
   def apply_contrast_to_data(self, brightness_factor=1.4):
       """
       This function iterates over each existing file in the images and labels folder 
       respectively, for the files in the images folder it creates a copy of the same 
       but with the contrast increased by applying the logic of the adjust_contrast 
       function, in turn it creates a copy of the annotation. txt in the labels folder 
       associated with the image to which the brightness was applied, in order to 
       maintain the yolo annotation for the new image created, these new annotations 
       (both .jpg image and .txt file) are assigned a name different from that of their 
       original versions so that they are not overwritten.
       """
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
       """ 
       This function allows the activation of the auto_segment function 
       by pressing the up navigation key on the keyboard.
       """
       self.auto_segment_bbox()
       
       
   def copy_image(self):
       """ 
       This function makes a copy of the current image with a unique name based on its 
       number and class and saves this image in the images folder of the GUI, 
       these copy images are used for the YOLO training format.
       """
       files = sorted(os.listdir(self.folder))
       file_path = os.path.join(self.folder, files[self.current_index])
       image_save_path = os.path.join(self.images_folder, "{}_{}.jpg".format(self.class_selector_var.get(), self.current_index))
       copyfile(file_path, image_save_path)
       
       return image_save_path


   def auto_segment_bbox(self):
       """
       This function first makes a copy of the current image by calling a function, then 
       generates the automatic segmentation in the last coordinates assigned to the 
       bounding box drawn by the user, then displays the next image of the chosen folder 
       and finally creates the annotations based on the binary mask and the dictionary 
       key that is the path of the current image.
       """
       image_save_path = self.copy_image()
       mask_binary = self.segment_bbox()
       self.show_next()
       self.create_simple_annotations(mask_binary= mask_binary, image_save_path= image_save_path)


   def point_auto_segment(self):
       """ 
       This function first makes a copy of the current image by calling a function, then 
       generates the automatic segmentation in the last coordinates assigned to the 
       point cordinates pointed by the user, then displays the next image of the chosen folder 
       and finally creates the annotations based on the binary mask and the dictionary 
       key that is the path of the current image.
       """
       image_save_path = self.copy_image()
       mask_binary = self.segment_point()
       self.show_next()
       self.create_simple_annotations(mask_binary= mask_binary, image_save_path= image_save_path)

       
   def create_simple_annotations(self,mask_binary,image_save_path):
       """ 
       The task of this function is to create and store yolo format polygonal and bbox 
       annotations in two respective dictionaries.
       As a first point it finds the contours of the mask_binary, then it approximates 
       the contours to polygons using cv2.approxPolyDP, then it normalizes the coordinates 
       of the polygon points in relation to the image size, stores the coordinates in 
       the polygon coordinates list, if the necessary elements exist it saves the binary 
       image in the cache folder and creates the YOLO annotation for segmentation, finally 
       the polygonal annotation is saved using the image path as the dictionary key.
       For the creation of the bounding box annotations, the largest contour is found in 
       the binary image, the coordinates and normalized dimensions of the bounding box 
       around the largest contour are calculated, the class index is obtained and the 
       annotation is saved in the bbox dictionary with the image path as the key.
       """
       #Creating poligonal annotations
       contornos, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       coordenadas_poligonales = []
       
       for contorno in contornos:
           
           epsilon = 0.01 * cv2.arcLength(contorno, True)
           poligono = cv2.approxPolyDP(contorno, epsilon, True)

           for punto in poligono:
               x, y = punto[0]
               coordenadas_norm = (x / mask_binary.shape[1], y / mask_binary.shape[0])
               coordenadas_poligonales.append(coordenadas_norm)

           if hasattr(self, 'max_score_mask') and mask_binary.any():
                selected_class = self.class_selector_var.get()
                if not selected_class:
                    selected_class = "None"

                if mask_binary is not None:
                    print("Guardando mask_binary en la carpeta de caché...")
                    mask_save_path = os.path.join(self.cache_folder, f"{self.current_index}_{selected_class}{selected_class}.png")
                    cv2.imwrite(mask_save_path, mask_binary)
                else:
                    print("Error: mask_binary es NONE")
                    
                if selected_class in self.classes:
                    yolo_annotation = f"{self.classes.index(selected_class)}"
                    for coordenada in coordenadas_poligonales:
                        yolo_annotation += f" {coordenada[0]:.6f} {coordenada[1]:.6f}"

                self.yolo_annotation_segmentation_dict[image_save_path] = yolo_annotation

       #Creating bbox annotations.
       largest_contour = max(contornos, key=cv2.contourArea)
       x, y, w, h = cv2.boundingRect(largest_contour)

       img_width, img_height = mask_binary.shape[::-1]
       x_center = (x + w / 2) / img_width
       y_center = (y + h / 2) / img_height
       width = w / img_width
       height = h / img_height

       selected_class = self.class_selector_var.get()
       class_index = self.classes.index(selected_class)

       yolo_annotation_bbox = f"{class_index} {x_center} {y_center} {width} {height}"
       self.yolo_annotation_bbox_dict[image_save_path] = yolo_annotation_bbox


   def shift_point_segment(self, master):
       """ 
       This function allows the activation of the point_auto_segment 
       function by pressing the right shift key on the keyboard
       """
       self.point_auto_segment()
       

   def save_annotations(self):
        """
        This function writes the annotations generated by the auto_segment 
        and auto_segment_point function to .txt files in the yolo format, 
        for each segmented image.
        The function accesses the information stored in the 
        self.yolo_annotation_segmentation_dict and self.yolo_annotation_bbox_dict 
        dictionaries and writes them to txt files in the labels and labels_bbox 
        folders respectively, each key of the dictionaries is associated with the 
        copies of the original images stored in the images folder and the for loop 
        iterates over this information, thus allowing the creation of annotations 
        only for the images segmented, respecting its same name and 
        following the YOLO dataset format (It is not applied to bbox annotations).
        """       
        for image_file in self.yolo_annotation_segmentation_dict.keys():
            
            anotation = self.yolo_annotation_segmentation_dict[image_file]
            image_name= os.path.basename(image_file)
            label_file = os.path.join(self.labels_folder, image_name).replace("jpg","txt")
            with open(label_file, "w") as f:
                f.write(anotation)

        for image_file in self.yolo_annotation_bbox_dict.keys():
            
            anotation = self.yolo_annotation_bbox_dict[image_file]

            image_name = os.path.basename(image_file)
            label_file_2 = os.path.join(self.labels_bbox_folder, image_name).replace("jpg","txt")
            with open(label_file_2, "w") as f:
                f.write(anotation)

        print("Anotaciones guardadas con éxito.")
       
      
   def start_training(self):
       """
       This function allows the initialization of the training of the YOLO model,
       as a first step the .yaml file created in the create_config_file instance
       is obtained, subsequently the size of the model to be retrained is chosen 
       and it is configured in training with the .yaml and the desired parameters,
       At the end, a warning message is displayed that notifies the user 
       of the completion of this process.
       The data augmentation instances generated in other functions 
       are not activated at the moment.
       """
       #Augmenting brightness on all images
       #brightness_contrast = 1.4
       #self.apply_contrast_to_data(brightness_factor)
       #Path were .yaml will be stored
       config_file_path = self.create_config_file()
       model = YOLO("yolov8n.yaml")
       result = model.train(data = config_file_path, epochs = 70, device = 0, patience = 15)
       messagebox.showinfo("Training Finished")
       
      
def main():
   root = Tk()
   root.title("Image Labeler")

   app = PhotoViewer(root)

   root.mainloop()

if __name__ == "__main__":
   main()
