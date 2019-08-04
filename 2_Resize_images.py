import re
import os
from PIL import Image

def resize_image(image_name, bfr_extension, count):
    basewidth = 100
    img = Image.open(image_name)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, basewidth), Image.ANTIALIAS) # replaced hsize by basewidth
    # print(hsize)
    img.save("/Users/svinjamara/Documents/Preprocess_5_Resized/{}_{}.png".format(bfr_extension, count))


if __name__ == '__main__':
    count = 0
    
    # Extracting all files and their names using os.walk, alternative is regex
    for dirpath, dirname, files in os.walk("/Users/svinjamara/Documents/Preprocess_5"):
        for file in files:
            path_name = os.path.join(dirpath, file)
            # print(path_name)
            count += 1
            
            bfr_extension = os.path.splitext(file)[0]
            bfr_extension = re.sub("\d+", "", bfr_extension)

            resize_image(path_name, bfr_extension, count)
    
    print("Resizing of images for CNN -> Done !!")

            

            
            
