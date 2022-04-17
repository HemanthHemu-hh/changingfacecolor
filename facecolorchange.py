import cv2
import numpy as np
import dlib
import time
import mediapipe as mp


img = cv2.imread('mahesh wallpaper.jpg')
img=cv2.resize(img,(800,600))
cv2.imshow("source image",img)
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
print(face_cascade)

gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray,None,2)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0))
cv2.imshow('img',img)
for (x,y,w,h) in faces:
    FaceImg = img[y:y+h,x:x+w]
    cv2.imshow('faceimage',FaceImg)
    filename = 'savedimage16.jpg'
    cv2.imwrite(filename,FaceImg)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()


img2 = cv2.imread('Junior-N.T.R.jpg')
height,width,_=img2.shape
cv2.imshow("orginal image",img2)
# img2=cv2.resize(img2,(800,600))
img2_gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
img2_rgb = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
mask=np.zeros_like(img2)


def color_transfer(source, target):
	source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
	target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")
	(lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
	(lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)
	(l, a, b) = cv2.split(target)
	l -= lMeanTar
	a -= aMeanTar
	b -= bMeanTar
	l = (lStdTar / lStdSrc) * l
	a = (aStdTar / aStdSrc) * a
	b = (bStdTar / bStdSrc) * b
	l += lMeanSrc
	a += aMeanSrc
	b += bMeanSrc
	l = np.clip(l, 0, 255)
	a = np.clip(a, 0, 255)
	b = np.clip(b, 0, 255)
	transfer = cv2.merge([l, a, b])
	transfer = cv2.cvtColor(transfer.astype("uint8"),cv2.COLOR_LAB2BGR)
	return transfer

def image_stats(image):
	(l, a, b) = cv2.split(image)
	(lMean, lStd) = (l.mean(), l.std())
	(aMean, aStd) = (a.mean(), a.std())
	(bMean, bStd) = (b.mean(), b.std())
	return (lMean, lStd, aMean, aStd, bMean, bStd)

cropedimage =cv2.imread("savedimage16.jpg")
colortransferd=color_transfer(cropedimage,img2)
cv2.imshow('color transfered',colortransferd) 

result=face_mesh.process(img2_rgb)

for facial_landamarks in result.multi_face_landmarks:
	landmarks_points=[]
	for i in range(0,468):
		pt1 = facial_landamarks.landmark[i]
		x = int(pt1.x * width)
		y = int(pt1.y * height)
		landmarks_points.append((x,y))
		# cv2.circle(img2, (x, y), 3, (0, 0, 255), -1)


points = np.array(landmarks_points,np.int32)
convexhull = cv2.convexHull(points)


cv2.polylines(img2,[convexhull],True,(255,0,0))
# cv2.imshow("canavas image",img2)

cv2.fillConvexPoly(mask,convexhull,(255,255,255))
# cv2.imshow("mask image",mask)

face_image_1 = cv2.bitwise_and(mask,colortransferd,mask = None)
cv2.imshow("extract image",face_image_1)

img2_face_mask=np.zeros_like(img2)
img2_head_mask= cv2.fillConvexPoly(img2_face_mask,convexhull,(255,255,255))
img2_face_mask = cv2.bitwise_not(img2_head_mask)
img2_head_noface = cv2.bitwise_and(img2_face_mask,img2,mask=None)
cv2.imshow("without mask",img2_head_noface)

result=cv2.add(img2_head_noface,face_image_1)
cv2.imshow("result",result)

(x,y,w,h)=cv2.boundingRect(convexhull)
center_face = (int((x+x+w)/2),int((y+y+h)/2))

seamlessclone = cv2.seamlessClone(colortransferd,result,img2_head_mask,center_face,cv2.NORMAL_CLONE)
cv2.imshow("seamlessclone",seamlessclone)

cv2.waitKey(0)
