import os 
import random
import re
import glob

txt_file = ["image_class_labels.txt", "images.txt", "train_test_split.txt", "classes.txt"]

def create_txt_file(file):
    return open(file, "a+")

for h, i in enumerate(txt_file):
    create_txt_file(i)
    print(f"{h+1}:  {i} is created")

print("\n")

# classes
with open(txt_file[3], "w") as f:
    for i, j in enumerate(["person", "wheelchair"]):
        i = i+1
        f.write(f"{i} 00{i}.{j}\n")

print("classes.txt is done")

# train_test_split
test_size = 0
n_for_person = len(os.listdir("images/001.person"))
n_for_wheelchair = len(os.listdir("images/002.wheelchair"))

dict_ = {"1": "train", "0": "test"}
a = [1] * int((1-test_size)*n_for_person) + [0] * int(test_size*n_for_person)
b = [1] * int((1-test_size)*n_for_wheelchair) + [0] * int(test_size*n_for_wheelchair)

random.seed(123)
random.shuffle(a)
random.shuffle(b)
c = a+b  # total 200 images from person and wheelchair

with open(txt_file[2], "w") as f:
    for i, j in enumerate(c):
        f.write(f"{i+1} {j}\n")
print("train_test_split.txt is done")

#images
with open(txt_file[1], "w") as f:
    image_names = glob.glob("images/001.person/*") + glob.glob("images/002.wheelchair/*")#os.listdir("images/001.person") + os.listdir("images/002.wheelchair")
    image_names = [re.sub("images/", "", i) for i in image_names]
    for i, j in enumerate(image_names):
        f.write(f"{i+1} {j}\n")


print("images.txt is done")

#image_class_labels
with open(txt_file[0], "w") as f:
    labels = [1] * n_for_person + [2] * n_for_wheelchair
    for i, j in enumerate(labels):
        f.write(f"{i+1} {j}\n")

print("image_class_labels.txt is done")