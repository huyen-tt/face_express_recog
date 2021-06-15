import glob
import os
images = glob.glob("./RAF_DB/datasets/FER_2013/Image/*.jpg")
i=0
for img in images:
    i+=1
    img = os.path.basename(img)
print(i)