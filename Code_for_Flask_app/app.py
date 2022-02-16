# import the necessary packages
from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/mask_image', methods=['POST'])
def mask_image():
    img = request.files['image']
    img.save('static/{}.jpg')

	# load face detector model
    print("Loading face detector model...")
    prototxtpath = os.path.sep.join(['face_detector', "deploy.prototxt"])
    weightspath = os.path.sep.join(['face_detector',
		"res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNet(prototxtpath, weightspath)

	# load the face mask detector model
    print("Loading face mask detector model...")
    model = load_model('mask_detection_model.h5')

	# load the input image from disk, clone it, and grab the image spatial
	# dimensions
    image = cv2.imread('static/{}.jpg')
	#image = cv2.imread(img)
    orig = image.copy()
    (h, w) = image.shape[:2]

	# construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
    print("Computing face detections...")
    net.setInput(blob)
    detections = net.forward()

	# loop over the detections
    for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
        confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
        if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face_roi = image[startY:endY, startX:endX]
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            face_roi = cv2.resize(face_roi, (224, 224))
            face_roi = img_to_array(face_roi)
            face_roi = preprocess_input(face_roi)
            face_roi = np.expand_dims(face_roi, axis=0)

            # pass the face through the model to determine if the face has a mask or not
            pred_final= model.predict(face_roi)

            # determine the class label and color we'll use to draw the bounding box and text            
            label='No Mask' if pred_final < 0.5 else 'Mask'
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            #probability = list(map(lambda x:str(x),pred_final * 100))
            #label ="{}:{}%".format(label, probability)
            

            # display the label and bounding box rectangle on the output frame
            cv2.putText(image, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

	# show the output image
    #cv2.imshow("Output", image)
    #cv2.waitKey(0)
    filename='static/image.jpg'
    cv2.imwrite(filename, image)
    return render_template('prediction.html')
	
@app.route('/load_img')
def load_img():
    return send_from_directory('static', "image.jpg")

if __name__ == '__main__':
    app.run(debug=True)
