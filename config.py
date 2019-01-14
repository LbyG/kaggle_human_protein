class DefaultConfigs(object):
    train_data = "../input" # where is your train data
    test_data = "../input"   # your test data
    weights = "./checkpoints/"
    best_models = "./checkpoints/best_models/"
    submit = "./submit/"
    model_name = "bninception_bcelog"
    #model_name = "inceptionresnetv2_bcelog"
    #model_name = "inceptionv4_bcelog"
    num_classes = 28
    img_weight = 512
    img_height = 512
    channels = 3
    lr = 0.02
    batch_size = 28 #96
    epochs = 5
    f1_thr = 0.4
    kfoldN = 5

config = DefaultConfigs()
