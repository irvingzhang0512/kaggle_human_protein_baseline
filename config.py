import os


class DefaultConfigs(object):
    # base_data_dir = '/ssd/zhangyiyang/protein/'
    base_data_dir = '/home/tensorflow05/data/kaggle/protein/'
    train_dir = os.path.join(base_data_dir, 'train')
    test_dir = os.path.join(base_data_dir, 'test')
    train_csv = os.path.join(base_data_dir, 'train.csv')
    test_csv = os.path.join(base_data_dir, 'sample_submission.csv')

    loss_name = "f1"
    # loss_name = "focal"
    # loss_name = "ce"
    logs_dir = "./logs-%s" % loss_name
    weights = os.path.join(logs_dir, 'checkpoints')
    best_models = os.path.join(weights, 'best_models')
    submit = os.path.join(logs_dir, 'submit')
    model_name = "bninception_bcelog"
    # model_name = "inceptionresnetv2"
    gpu_id = "3"

    # basic config
    num_classes = 28
    img_weight = 512
    img_height = 512
    channels = 4
    val_percent = 0.13
    batch_size = 64
    epochs = 50
    weight_decay = 0.0001

    # training configs
    learning_rate_start = 0.03
    learning_rate_decay_epochs = 10
    learning_rate_decay_rate = 0.1

    # steps
    logging_every_n_steps = 10


config = DefaultConfigs()
