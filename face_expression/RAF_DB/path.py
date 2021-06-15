import glob
import os
images = glob.glob("./datasets/FER_2013/Image/*.jpg")
i=0
for img in images:
    # img = os.path.basename(img)
    # img = img + ' 7'
    # print(img)
    i+=1
print(i)