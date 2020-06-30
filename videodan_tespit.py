import numpy as np
import argparse
import imutils
import time
import cv2
import os

# gerekli argumanlarin alınmasi.
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="path to output video")
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

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["video"])
writer = None
(W, H) = (None, None)

# videodaki frame sayisi sayiliyor
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} tane frameden olusan video gonderdiniz".format(total))

# eger video gecersizse hata mesaji dondurulur.
except:
	print("[INFO] videodaki frame sayisi belirlenemedi")
	total = -1

# videodaki her frame' in islenmesi icin donguye girilir.
while True:
	# frame okunur.
	(grabbed, frame) = vs.read()

	# frame yakalanmaz ise donguden cikilir
	if not grabbed:
		break

	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# yakalanan frameden  blob olusturuluyor.(blob :  bazı ortak özellikleri paylaşan bağlı piksel grubudur.
	# (örneğin, gri tonlamalı değer).olusturulan blob cv2 ye veriliyor.
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

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

			# mininmum confidance degerini biz default olarak (0.5) vermistik bundan buyukse nesneyi onayliyoruz
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
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]],
				confidences[i])
			cv2.putText(frame, text, (x, y - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	if writer is None:
		# video cevirme islemi baslar.
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

		if total > 0:
			elap = (end - start)
			print("[INFO] bir frame {:.4f} saniye surdu".format(elap))
			print("[INFO] toplamda yaklasik  {:.4f} saniye surecektir ".format(
				elap * total))

	# frame videoya kaydedilir.
	writer.write(frame)

print("[INFO] Bitiriliyor...")
writer.release()
vs.release()