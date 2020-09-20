from PIL import Image
import os

dir_list = os.listdir('datasets/')
for dir_name in dir_list:
    img_list = os.listdir('datasets/' + dir_name + "/")
    for img_name in img_list:
        img = Image.open('datasets/' + dir_name + "/" + img_name)
        img_resize = img.resize((420, 420))
        img_resize.save('datasets_resized/' + img_name)
