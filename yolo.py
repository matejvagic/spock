import torch
from PIL import Image
import os
import shutil


from  yolov5.yolov5.models.yolo   import *



def run_image(path_for_labels, img,  model, path_for_output_images):
    name_of_file = os.path.basename(img).split(".")[0]

    im = Image.open(img)
    width, height = im.size

    results = model(img)
    results.save(os.path.join(os.getcwd(),  path_for_output_images))

    with open("{}/{}.txt".format(path_for_labels,name_of_file), "w") as f:
        for i in results.xywhn[0]:
            if (int(i[-1].item() == 0)):
                f.write("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(int(i[-1].item()), i[0].item(), i[1].item(), i[2].item(), i[3].item()))
                #f.write(i[0].item()/width, i[1].item()/height, i[2].item()/width, i[3].item()/height)
                
                # out = convert_to_yolo_format((width, height), (i[0].item() - i[2].item()/2 ,i[0].item() +  i[2].item()/2, i[1].item() - i[3].item(), i[1].item() + i[3].item()/2))
                # #out = yolobbox2bbox(i[0].item(), i[1].item(), i[2].item(), i[3].item())
                # f.write("  {} {} {} {}\n".format(out[0], out[1], out[2], out[3]))
                # f.write("\n")




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
    test()
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
    


def test():
    model_name  = "yolov5s"
    weights = '/Users/matejvagic/Desktop/spock_kodovi/yolov5/yolov5/runs/train/exp25/weights/best.pt'
    # model = torch.hub.load('ultralytics/yolov5', model_name, classes=1)
    # model.load_state_dict(torch.load('/Users/matejvagic/Desktop/spock_kodovi/yolov5/yolov5/runs/train/exp25/weights/best.pt')['model'].state_dict())

    ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
    model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to("cpu")  # create
    exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
    csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
    csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
    model.load_state_dict(csd, strict=False)  




    ext = "proba"
    img  =  "/Users/matejvagic/Downloads/parent-child-relationships.jpg"
    check_dir("final" +"_"+ext)
    run_image( ext, img, model, path_for_output_images +  "_"+ ext)
    exit()





if __name__ == '__main__':
    main()
    