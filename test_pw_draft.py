import cv2
import torch
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import data.cubloaderr
from net.Orderextractor import OEmbeddingNetwork
from experiments.cosineclassifer import CosineClassifier
from utils.parser import get_arg
import argparse
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from seaborn import heatmap
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from PIL import Image
import math

######## Args #########
parser = get_arg()
SUPPORT_NUM = parser.support_num #number of support sets
num_query = parser.val_support_num #number of query sets
novelonly = True
withAtt = False
model_path = "./models/closs_high.pkl"
baseclass_weight = './models/baseO.pkl'
baseclass_att_weight = './models/baseAttOrder.pkl'
support_data_path = parser.datapath #path of support sets
query_data_path = parser.datapath1 #path of query sets, same as support sets
######## Args #########



if withAtt:
    print("Foucus area could help you!")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('==> Reading from model checkpoint..')
embedding = OEmbeddingNetwork().to(device) #load the feature extractor
embedding.load_state_dict(torch.load(model_path)) #load its trainable params
def main():
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    novel_trasforms = transforms.Compose([
            transforms.Resize(82),
            transforms.CenterCrop(82),
            transforms.ToTensor(),
            normalize #input dimension reduction through resize and crop, then normalize the values
    ])
    att_trasforms = transforms.Compose([
        transforms.Resize((82, 82)),
        transforms.ToTensor(),
        normalize
    ])
    #using the DataLoader to set the number of iterations and batch size, train=True is support set and train=False is query set
    #train=True -> train_test_split.txt is 1  & train=False -> train_test_split.txt is 0
    #since no att_transforms, it will be same with the original images
    #num_classes=2, since there are only two class: person & person sit on wheelchair
    novel_dataset = data.cubloaderr.ImageLoader(
        support_data_path,
        novel_trasforms,
        att_transform=att_trasforms,
        train=True, num_classes=2,
        num_train_sample=SUPPORT_NUM,
        novel_only=True, aug=False)
    novel_loader = torch.utils.data.DataLoader(
        novel_dataset, batch_size=1, shuffle=False,
        num_workers=8, pin_memory=True)
    """
    val_loader = torch.utils.data.DataLoader(
        data.cubloaderr.ImageLoader(datapath, novel_trasforms
        , att_trasforms, num_classes=2, novel_only=novelonly),
        batch_size=32, shuffle=False,
        num_workers=8, pin_memory=True)
    """
    #fewcub function takes inputs including the feature extractor, support sets and query sets
    #the last CNN layer with total numbers of 1024 nodes generate a vector in 1024-dimensional vector space for each query set.
    #Given N-number of support sets and M-number of classes, it first will concatenate each query vector by row to form a matrix with N*M numbers of row and 1024 numbers of column.
    #Reshape the 2-dimensional matrix into (M*N*1024) 3-dimensional matrix
    #Get the mean value of total number (N) of support set for each class and become (M*1*1024) 3-dimensional matrix 
    #Image file (png, jpg) can open using PIL.Image, plt.imread or cv2.imread and here, we are using PIL.Image and then convert into RGB

    cosineclassifer = CosineClassifier(with_att=withAtt, novel_only=novelonly)
    acc = fewcub(cosineclassifer, novel_loader, embedding)
def fewcub(classifier, novel_loader, model):
    basetrain, basetrainatt = None, None
    if novelonly is False:
        basetrain = torch.load(baseclass_weight) #base train weight
        basetrainatt = torch.load(baseclass_att_weight) #base train of focus-area's weight
    model.eval()
    with torch.no_grad():
        for batch_idx, (input, target, att) in enumerate(novel_loader):
            input, target, att = input.to(device), target.to(device), att.to(device)
            output,_,_ = model(input)
            o_att,_,_ = model(att)
            if batch_idx == 0:
                output_stack = output
                output_cat = o_att
            else:
                output_stack = torch.cat((output_stack, output),0)
                output_cat = torch.cat((output_cat,o_att),0)
        output_stack = torch.sum(output_stack.view(2,SUPPORT_NUM,-1),1)/SUPPORT_NUM # 2 presents 2 classes, 100 presents 100 classes
        output_cat = torch.sum(output_cat.view(2,SUPPORT_NUM, -1), 1) / SUPPORT_NUM

        torch.save(output_stack, "./support_set.pt")

        people_query_dir = os.path.join(query_data_path, "images", "001.person") #person class path
        wheelchair_query_dir = os.path.join(query_data_path, "images", "002.wheelchair") #wheelchair class path
        #If the total number of image files in person class folder is 8, the support number I set is 2, 
        #and the query number I set is 5. The support set will choose first and second image file [1.png, 2.png] 
        #since the code for support sets is df["label"].iloc[:SUPPORT_NUM]. Hence, the number of remaining image 
        #files are 6 (8-2). Among this 6 image files, the first four will be randomly selected and will not be redundant.
        random_select = np.random.choice([i for i in range(SUPPORT_NUM, SUPPORT_NUM+num_query)], num_query, replace = False)

        dict_class = {1: "person", 2: "wheelchair"}
        total_pred = []
        total_actual = []
        query_image_list = []
        #for loop for each class
        for key, value in dict_class.items():
            print("The random selected class is {}.".format(value))
            dict_class_path = {1: people_query_dir, 2: wheelchair_query_dir}
            selected_class = dict_class_path[key]
            image_path1 = os.listdir(selected_class)
            df = pd.DataFrame({"dir": image_path1})
            query_images = df.iloc[random_select].values.flatten().tolist()
            query_image_list += query_images

            transforms_query = transforms.Compose([
                transforms.Resize([84,84]),
                transforms.CenterCrop(84),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            #for loop for subplots
            for idx, img_file in enumerate(query_images):
                img_path = os.path.join(selected_class, img_file)
                with open(img_path, "rb") as f:
                    img = Image.open(f)
                    img = img.convert("RGB") #read the random selected image file here
                    img_transform = transforms_query(img).unsqueeze(dim=0) #add the first dimension
                plt.subplot(math.ceil(len(query_images)/5), 5, idx + 1) #construct subplot for viewing query sets images
                plt.title(idx + 1)
                img_plot = plt.imshow(img)
                #currently named as query_images2, this variables is to form matrix and put into feature extractor model and cosine similarity metrics
                if idx == 0:
                    query_images_matrix = img_transform
                else:
                    query_images_matrix = torch.cat([query_images_matrix, img_transform], dim = 0)
            plt.show()
            print("Attention {}".format(query_images_matrix.size()))
            query_images_matrix_cuda = query_images_matrix.cuda()
            att_f,_,_ = model(query_images_matrix_cuda)
            print("Attention2 {}".format(att_f.size()))
            similarities = classifier(basefeat=basetrain, basefeat_att=basetrainatt,
                        supportfeat=output_stack, supportfeat_att=output_cat, queryfeat=att_f, queryfeat_att=att_f)
            _, preds = torch.max(similarities, 1)
            preds = preds.to("cpu").tolist()
            actuals = [int(key)-1 for i in range(len(preds))]
            total_pred += preds
            total_actual += actuals

        #using confusion matrix, classification report and confusion matrix plot to visualize the performance
        confusion_matrix_pw = confusion_matrix(total_actual, total_pred)
        classification_report_pw = classification_report(total_actual, total_pred)
        print(confusion_matrix_pw)
        print(classification_report_pw)
        print(ConfusionMatrixDisplay(confusion_matrix_pw, display_labels = ["People", "Wheelchair"]).plot())
        plt.show()

        #construct dataframe
        df = pd.DataFrame({"Image_path": [i[-10:] for i in query_image_list], "Actual": total_actual, "Prediction": total_pred})
        df["Actual"] = df["Actual"].apply(lambda x: "person" if x == 0 else "wheelchair")
        df["Prediction"] = df["Prediction"].map({0: "person", 1: "wheelchair"})
        df["isCorrect"] = df.apply(lambda x: "correct" if x["Actual"] == x["Prediction"] else "incorrect", axis = 1)
        print(df, "\n")
        print(df["isCorrect"].value_counts())

        print("==" * 18)
        print("The number of support set is {}".format(SUPPORT_NUM))
        print("The total number of query set is {}".format(num_query))
        print("This is 2-way {}-shot learning".format(SUPPORT_NUM))
        print("==" * 18)

if __name__ == '__main__':
    main()