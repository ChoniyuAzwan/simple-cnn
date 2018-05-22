# import lib yang dibutuhkan
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.preprocessing.image import load_img
import numpy as np
import argparse
import imutils
import pickle
#import cv2
import os

# konstruksi argument
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path ke model yang telah dilatih")
ap.add_argument("-l", "--labelbin", required=True,
	help="path ke label binary")
ap.add_argument("-i", "--image", required=True,
	help="Gambar yang akan ditebak")
args = vars(ap.parse_args())

# load image
#image = cv2.imread(args['image'])
image = load_img(args['image'], target_size=(96, 96))
#output = image.copy()


# pra proses
#image = cv2.resize(image, (96, 96))
#image = image.astype("float") / 255.0
image = np.array(image, dtype="float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load model & label
print("[INFO] Memuat Jaringan ...")
model = load_model(args['model'])
lb = pickle.loads(open(args['labelbin'], 'rb').read())

# klasifikasi input
print('[INFO] Mencari Kelas Gambar')
proba = model.predict(image)[0]
idx = np.argmax(proba)
label = lb.classes_[idx]

# prediksi label + confidence score
label = "{}: {:.2f}%".format(label, proba[idx] * 100)
#output = imutils.resize(output, width=400)
#cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
#	0.7, (0, 255, 0), 2)
print(label)

# tampilkan gambar
#print("[INFO] Menampilkan Gambar")
#cv2.imshow("Hasil", output)
#cv2.waitKey(0)
