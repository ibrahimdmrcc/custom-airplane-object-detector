import numpy as np
import argparse
import time
import cv2
import os

# gerekli argumanlarin alınmasi.

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="resim dosyasinin bulundugu konum")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# Bir tane nesne egittigim icin LABELS listesi bir adet nesne iceriyor.
LABELS = ["Ucak"]

# cfg ve weight dosyasinin path'leri veriliyor.
weightsPath = os.path.sep.join([args["yolo"], "yolov3_airplane_2000.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3_custom.cfg"])

# config dosyası ve egitilen weights dosyasi yukleniyor.
print("[INFO] Gerekli dosyalar yukleniyor...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# opencv'ye image dosyasi veriliyor
image = cv2.imread(args["image"])
(H, W) = image.shape[:2]

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# verilem imageden blob olusturuluyor.(blob :  bazı ortak özellikleri paylaşan bağlı piksel grubudur (örneğin, gri tonlamalı değer).
#olusturulan blob cv2 ye veriliyor.
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

# kac saniyede aldiginin bilgisi.
print("[INFO] YOLO  {:.6f} saniyede tamamladi".format(end - start))

# nesne algilandiginda gelecek kordinatlar algilanan nesne sinifi(Ucak) vb seyler icin
# listeler olusturuluyor.

boxes = []
confidences = []
classIDs = []

for output in layerOutputs:

	for detection in output:

		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]

		# mininmum guven degeri biz default olarak 0.5 vermistik bundan buyukse nesneyi onayliyoruz.
		if confidence > args["confidence"]:

			box = detection[0:4] * np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")

			# nesnenin merkezinden genisligin ve yuksekligin yarisi cikarilarak
			# sol ust kose koordinatlari bulunuyor.
			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))

			# olusturulan listeler sınıf indeksi ve nesnenin kordinatlari
			# ile guncelleniyor.
			boxes.append([x, y, int(width), int(height)])
			confidences.append(float(confidence))
			classIDs.append(classID)


idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
	args["threshold"])

if len(idxs) > 0:

	for i in idxs.flatten():

		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])

		# algilanan nesnenin dikdortgen ile cizilmesi
		color = [51,243,243]
		cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
		text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
		cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, color, 2)

# isaretlenen resmin gosterilmesi.
cv2.imshow("Image", image)
cv2.waitKey(0)