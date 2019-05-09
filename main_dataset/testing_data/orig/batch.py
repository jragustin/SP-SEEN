import cv2, glob

images=glob.glob("*.jpg")
n0 = 1
n1 = 101
n2 = 201
n3 = 301


angle90 = 90
angle180 = 180
angle270 = 270

center = (256,256)

scale = 1.0
for image in images:

	img=cv2.imread(image,1)
	# re=cv2.resize(img,(512,512))

	# cv2.imshow("Checking",re)
	M = cv2.getRotationMatrix2D(center,0,scale)
	rotated0 = cv2.warpAffine(img,M,(512,512))
	M = cv2.getRotationMatrix2D(center,angle90,scale)
	rotated90 = cv2.warpAffine(img,M,(512,512))
	M = cv2.getRotationMatrix2D(center,angle180,scale)
	rotated180 = cv2.warpAffine(img,M,(512,512))
	M = cv2.getRotationMatrix2D(center,angle270,scale)
	rotated270 = cv2.warpAffine(img,M,(512,512))
	
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	string0=str(n0)
	string1=str(n1)
	string2=str(n2)
	string3=str(n3)
	cv2.imwrite("resized_"+string0+".jpg",rotated0)
	cv2.imwrite("resized_"+string1+".jpg",rotated90)
	cv2.imwrite("resized_"+string2+".jpg",rotated180)
	cv2.imwrite("resized_"+string3+".jpg",rotated270)
	n0=n0+1
	n1=n1+1
	n2=n2+1
	n3=n3+1
