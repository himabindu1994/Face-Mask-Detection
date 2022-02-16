# Face-mask-detection

Below is the list of notebooks and python scripts developed in this project:

Training Mask detector: Three separate notebooks for training mask detector were developed. 
1.	Basic CNN : Train_Mask_detection_using_basic_CNN.ipynb
2.	Mobile NetV2 : Mask_detection_improvised_mobilenetv2.ipynb
3.	VGG16: VGG16 Model.ipynb

Face Mask detection: MobileNetV2 gave best results and the trained model was saved and used with a pre-trained face detection SSD model to identify masks in the faces in an image.

Pretrained Face Detection model was saved in the face_detector folder. Face_detector folder should be present in the folder of the python script along with the saved mask detector model.
Final Python script: integrate_faceandmask.py

To test on real time image run below command:
Integrate_faceandmask.py --image <image name>

Flask Local app:
Finally we also created a local flask app to run our model on real time images using a web portal. To run a local flask app, download Code_for_Flask_app as it is and follow the steps mentioned in readme.
  
  This folder contains the final trained mask detector model with best performance and code to create a local web app.

1. Download this folder as it is.
2. Run python app.py
3. Open the link displayed.
4. Select image for detection (jpg image)
5. Submit.
6. Result of the Face Mask detector is displayed.

