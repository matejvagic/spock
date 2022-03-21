import torch
from PIL import Image
import os
import shutil



def run_image(path_for_labels, img,  model, path_for_output_images):
    name_of_file = os.path.basename(img).split(".")[0]

    im = Image.open(img)
    width, height = im.size

    results = model(img)
    results.save(os.path.join(os.getcwd(),  path_for_output_images))

    with open("{}/{}.txt".format(path_for_labels,name_of_file), "w") as f:
        for i in results.xyxy[0]:
            if (int(i[-1].item() == 0)):
                f.write("{} {} {} {} {}\n".format(int(i[-1].item()), i[0].item()/width, i[1].item()/height, i[2].item()/width, i[3].item()/height))


def load_model(name):
    return torch.hub.load('ultralytics/yolov5', name)

def check_dir(path):
    CHECK_FOLDER = os.path.isdir(path)
    if not CHECK_FOLDER:
        os.makedirs(path)
        print("created folder : ", path)


def load_images(path_for_images):
    curr_dir = os.getcwd()
    return [os.path.join(curr_dir,  path_for_images, i) for i in  os.listdir(path_for_images)]


def move_file(old, new):
    name_of_file = os.path.basename(old)
    shutil.copyfile(old, os.path.join(new, name_of_file))


def split_dataset(train_percentage, images):
    split_index = int(len(images) * train_percentage)
    image_train = images[0:split_index]
    image_test = images[split_index:]

    return image_train, image_test



def main():
    model_name  = "yolov5s"
    model = load_model(model_name)

    path_for_images  = "images"
    
    path_for_output_images =  "output"

    train_percentage = 0.7

    images  = load_images(path_for_images)
    

    image_train, image_test = split_dataset(train_percentage, images)

    #train
    ext = "train"
    check_dir( ext)
    check_dir(path_for_output_images +"_"+ext)

    
    for img in image_train:
        run_image( ext, img, model, path_for_output_images +  "_"+  ext)
        move_file(img,  ext)

    
    ext = "test"
    check_dir( ext)
    check_dir(path_for_output_images + "_"+ext)


    for img in image_test:
        run_image( ext, img, model, path_for_output_images +  "_"+ ext)
        move_file(img,  ext)
    







if __name__ == '__main__':
    main()
    