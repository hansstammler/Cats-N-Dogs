from fastai.vision.all import *
import warnings
import sys

# if true, is cat
# layers is from 0 to "layer"

def arg_handler():
    global override
    override = False
    global epochs
    epochs = 1
    global test_s
    test_s = False
    global lr
    lr = 3e-3
    global image_crop
    image_crop  = 224
    global lyr
    lyr = 0
    global aug
    aug = True
    global bn
    bn = True
    arguments = sys.argv
    for a in range(len(arguments)):
        if arguments[a] == "-o":    #Override previous network with -o
            override = True
        if arguments[a] == "-e":    #Specify amount of eppochs with ex. -e 10
            override = True
            epochs = int(arguments[a + 1])
        if arguments[a] == "-lr":   #Specify learning rate with ex. -lr 0.001
            lr = float(arguments[a + 1])
        if arguments[a] == "-crop": #Specify image crop size with ex. -crop 200
            image_crop = int(arguments[a + 1])
        if arguments[a] == "-lyr": #Specify how many parameter groups you want to freeze with ex. -lyr 2
            lyr = int(arguments[a + 1])
        if arguments[a] == "-noaug":#Specify that no augumentation should be used on images with -noaug
            aug = False
        if arguments[a] == "-nobn": #Specify that no batch normalization should be used with -nobn
            bn = False

def label_func(file):
    return file[0].isupper()


def initialize():
    path = untar_data(URLs.PETS)
    files = get_image_files(path/"images")
    return path, files


def learning(dls):
    learn = vision_learner(dls, resnet34, metrics=error_rate, train_bn=bn)
    path = dls.path
    #found_lr = learn.lr_find()
    #print(found_lr)
    #plt.savefig("lr.png")

    print("Overide status:", override)
    print("Running with", epochs, "epoch(s)")
    if (path/'models/learnerfile.pth').exists() and override == False:
        learn = learn.load('learnerfile')
        print("found learner, loaded")
    else:
        if lyr != 0:
            learn.unfreeze()
            learn.freeze_to(lyr)
            #print(learn.summary())
            learn.fit_one_cycle(epochs, lr)
        else:
            learn.fine_tune(epochs, lr)

        
        learn.save('learnerfile')
        print("saved to file")
    return learn


def main():
    arg_handler()

    warnings.filterwarnings("ignore")
    path, files = initialize()
    #dls = data loaders (the data set)
    #dls = ImageDataLoaders.from_name_func(path, files, label_func, item_tfms=Resize(224))
    pattern = r'^(.*)_\d+.jpg'

    # This is if data augmentations should be used or not
    if aug == True:
        dls = ImageDataLoaders.from_name_re(path, files, pattern, item_tfms=Resize(460), batch_tfms=aug_transforms(size=image_crop))
    else:
        dls = ImageDataLoaders.from_name_re(path, files, pattern, item_tfms=Resize(image_crop))
    
    learn = learning(dls)
    dls.show_batch(max_n=32)
    plt.savefig("labels.pdf")
    learn.show_results(max_n=100)
    plt.savefig("predicted.pdf")
    interp = Interpretation.from_learner(learn)
    interp.plot_top_losses(9, figsize=(15,10))
    plt.savefig("worstplot.pdf")

    print("done")


if __name__ == "__main__":
    main()
