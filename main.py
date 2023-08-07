from fastai.vision.all import *
import warnings
import sys

# if true, is cat

def arg_handler():
    global override
    override = False
    global epochs
    epochs = 1
    global test_s
    test_s = False
    arguments = sys.argv
    for a in range(len(arguments)):
        if arguments[a] == "-o":
            override = True
        if arguments[a] == "-e":
            override = True
            epochs = int(arguments[a + 1])
        if arguments[a] == "-t":
            test_s = True


def label_func(file):
    return file[0].isupper()


def initialize():
    path = untar_data(URLs.PETS)
    files = get_image_files(path/"images")
    if test_s == True:
        files[0:37]
    return path, files


def learning(dls):
    learn = vision_learner(dls, resnet34, metrics=error_rate)
    path = dls.path
    #found_lr = learn.lr_find()
    #print(found_lr)
    #plt.savefig("lr.pdf")
    lr = 3e-3

    print("Overide status:", override)
    print("Running with", epochs, "epoch(s)")
    if (path/'models/learnerfile.pth').exists() and override == False:
        learn = learn.load('learnerfile')
        print("found learner, loaded")
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
    dls = ImageDataLoaders.from_name_re(path, files, pattern, item_tfms=Resize(460), batch_tfms=aug_transforms(size=224))
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
