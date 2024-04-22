In this file, I will describe the usage of the created graphical user interface, its main objective, prominent uses and limitations, as well as potential areas for improvement at first glance.

General Objective and Used Frameworks

The main objective of the created code is to label datasets in YOLO format easily and assisted offline. This is because, during robot competitions, there isn't always a stable internet connection to use services with this capability. To achieve this task, the Segment Anything Model provided by Meta was employed, which is open-source and has all the necessary documentation on its GitHub page. Other popular Python frameworks for image processing were also used, such as OpenCV, PIL, and Tkinter for the general creation of the graphical interface. Each created function has a detailed explanation within the code itself for better understanding.

General Code Explanation

In its main function, the code primarily accesses the SAM mask predictor to predict the mask with the highest percentage of interest within the selected area. The selected area can be a bounding box drawn with the left mouse click or a single point selected with a right-click. By doing this, the image updates to show the resulting mask from the segmentation process done by SAM. The GUI has two main compatible annotation methods: manual annotations (recommended when working on images with many objects to label in each) and assisted auto-segmentation. To access manual annotation, simply select each object of interest along with its class and press the "Save Mask" button, or perform the segmentation and choose the class, and press the "Left Control" button on the keyboard to save the YOLO mask and annotation to a .txt file. To generate another annotation, simply repeat the process for the number of objects of interest in each image. Finally, the saved annotations will be displayed in the "label_left" of the app automatically, to show the annotations made for each image for better visual reference. The other functionality the app has is assisted auto-segmentation, which is designed for folders containing only one object per image. To access this, there must first be a bounding box drawn with the left click or a point selected with the right click of the mouse. To access automatic segmentation using the information saved in the bounding box, simply press the "up navigation arrow" on the keyboard, and the segmentation of the current image will be done, moving to the next image and showing the segmentation result in the viewer. If the object is centered in the same position in the image, simply press the same button again to perform segmentation on the next image and so on for the others. When not satisfied with the result of automatic segmentation and the resulting mask shown in the viewer is not suitable, simply press the "left navigation arrow" to return to the previous image, correct the bounding box drawing in the new area of interest, and press the up navigation button again to continue with autonomous segmentation. When the result is not satisfactory, repeat this process again. To access automatic segmentation with the point selected by the user (right-click), the same process is done with the difference that now the "Left Shift" button is pressed instead of the "up navigation" button. These annotations, unlike when accessing the instance of many annotations, are not automatically saved in the created .txt files but are saved in special dictionaries. When satisfied with the result of all these annotations, simply press the "Save Annotations" button on the GUI, and the annotations made will be written to the corresponding .txt files. When the message "Annotations saved successfully" appears in the terminal, it indicates that the process has been completed correctly. To perform auto-segmentations of more objects once the current folder is finished, the app must be closed and reopened to select the new folder.

App Usage Instructions

When running the code, the first step is to choose the folder containing the images to segment. Once the folder is chosen, the contained images will be displayed, and a class creation window will open. Inside this window, which has a text writing area, classes can be entered one by one by pressing the "Confirm" button, or they can all be prepared in a .txt file, separated by commas, and simply copy and paste that message and press the "Confirm" button to save all those classes. To see if the classes were saved satisfactorily, minimize the class creation window and press the button on the GUI where they are shown. If all classes have been annotated satisfactorily, they will be displayed in the list of possible classes.

As a second point, the segmentation processes explained in the "code explanation" section can be carried out. Once segmentation and annotations in the current folder are completed, the app can be closed and reopened to choose a different folder for new annotations.

Finally, the file management in the app is as follows: first, the parent directory is taken, which is located in the same file as the selected folder. Inside this parent directory, the "cache" folder is created (folder where the results of the various segmentation processes are saved) and the "data" folder (folder where the dataset structure is formed in YOLO format). Within the mentioned "data" folder, 3 subdirectories are created: the "images" folder (where copies of the worked and segmented images are made), the "labels" folder (where the .txt files with the YOLO polygonal annotations are generated), and finally the "labels_bbox" folder (where the .txt files with the generated bounding box annotations are generated). As a final note, multiple segmentation processes, for images with many objects in them and multiple annotations for a single image, only generate polygonal annotations in those instances, while in auto-segmentation annotations, both (polygonal and bbox) are generated.

Physical Shortcuts Available in the App

Mouse shortcuts:

    Left Mouse Click: Ability to draw a bounding box.
    Right Mouse Click: Ability to choose a point on the image by clicking.

Keyboard shortcuts:

    Right Arrow Key: Allows moving to the next image in the parent folder when pressed.
    Left Arrow Key: Allows moving to the previous image in the parent folder when pressed.
    Left Control Key: Allows saving the selected mask in the segmentation process (usable when segmenting many objects in an image).
    Up Arrow Key: Allows accessing the auto segment bbox function (auto-segmentation with bounding box information).
    Right Shift Key: Allows accessing the auto segment point function (auto-segmentation with point coordinates information drawn).

Possible Areas for Improvement

    Add a function that allows opening a different folder than the one currently being worked on, with the intention that the app does not have to be closed and to avoid the process of rewriting the app's classes (for this, some variables like self.current_index and some counters must be reset).

    Add bounding box annotations also to the processes of segmenting many objects in an image since currently, the logic is described in other functions, but in that specific part, it is not implemented.

    Improve the app's visualization, choose more optimal locations for the existing buttons within the GUI frame, and add some sliders for future functionalities.

    Apply an instance of dataset division internally. Currently, the dataset generated by the GUI as a result is loaded into Roboflow for division into folders, which is not so optimal if there is no internet connection. It would be opportune to apply internal logic for dividing it into YOLO format and thus also have validation and test folders as output.

    Apply internal instances of data augmentation (90 and 180-degree rotations mainly). Currently, there is logic for contrast enhancement, however, it is not implemented in the GUI flow. Brightness enhancement could also be added. To apply rotations to the annotations in the cache folder, the results of the segmentation processes are saved, rotations could be applied to these images, and the logic of generating annotations could be used to generate those annotations as an idea.

    Generate unique names for variables if this causes overwrite problems. To do this, timestamps or unique name variables can be applied.
