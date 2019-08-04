import re
import os
from new_preprocess import *

count = 0

regexes = [
    re.compile("iiok"),
    re.compile("nothing"),
    re.compile("peace"),
    re.compile("punch"),
    re.compile("stop")
    ]


for dirpath, dirnames, files in os.walk("/Users/svinjamara/Documents/imgs_dataset_5"):
    for file in files:
        if any(regex.match(file) for regex in regexes):
            path_name = os.path.join(dirpath, file)
            count +=1
            
            extension = os.path.splitext(file)[0]
            extension = re.sub("\d+", "", extension)
            
            pre_process_images(path_name, extension, count)

#print(count)
print("Pre-processing of Images done!!!")
