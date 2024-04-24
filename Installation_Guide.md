Installation instructions for the GUI: 

1.- pip install git+https://github.com/facebookresearch/segment-anything.git

2.- pip install opencv-python pycocotools matplotlib onnxruntime onnx

3.- pip install ultralytics

4.- pip install datetime

5.- Download any of the 3 models avaiable on the SAM github repo, and change the path and the model name (vit_b, vit_l, vit_h).
I recomend to use the medium size model called vit_l, and after to change the path on the code, also change the name of the model on the python code.

Model vit_l link: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
