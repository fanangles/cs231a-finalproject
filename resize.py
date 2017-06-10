from PIL import Image
from resizeimage import resizeimage

tids = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
for tid in tids:
	with open('TrainSmall2/TrainDotted/' + str(tid) + '.jpg', 'r+b') as f:
		with Image.open(f) as image:
			image = resizeimage.resize_height(image, 756)
			image.save(str(tid) + 'dotted.png', image.format)