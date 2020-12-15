import cv2
import glob

i = 1
for data in glob.glob("./data/donburi/*"):
    image = cv2.imread(data)
    flip_image = cv2.flip(image, 1)
    cv2.imwrite('./data/donburi/flip' + str(i) + '.jpg', flip_image)
    i += 1

i = 1
# for data in glob.glob("./data/metal_cont/*"):
#    image = cv2.imread(data)
#    flip_image = cv2.flip(image, 1)
#    cv2.imwrite('./data/metal_cont/flip' + str(i) + '.jpg', flip_image)
#    i += 1

i = 1
# for data in glob.glob("./data/smartsnap/*"):
#    image = cv2.imread(data)
#    flip_image = cv2.flip(image, 1)
#    cv2.imwrite('./data/smartsnap/flip' + str(i) + '.jpg', flip_image)
#    i += 1
i = 1
for data in glob.glob("./data/rice/*"):
    image = cv2.imread(data)
    flip_image = cv2.flip(image, 1)
    cv2.imwrite('./data/rice/flip' + str(i) + '.jpg', flip_image)
    i += 1
i = 1
# for data in glob.glob("./data/soup_s/*"):
#    image = cv2.imread(data)
#    flip_image = cv2.flip(image, 1)
#    cv2.imwrite('./data/soup_s/flip' + str(i) '.jpg', flip_image)
#    i += 1
