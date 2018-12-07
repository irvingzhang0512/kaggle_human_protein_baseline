import os


class DefaultConfigs(object):
    # base_data_dir = '/ssd/zhangyiyang/protein/'
    base_data_dir = '/home/tensorflow05/data/kaggle/protein/'
    train_dir = os.path.join(base_data_dir, 'train')
    test_dir = os.path.join(base_data_dir, 'test')
    train_csv = os.path.join(base_data_dir, 'train.csv')
    test_csv = os.path.join(base_data_dir, 'sample_submission.csv')

    # loss_name = "f1"
    loss_name = "focal"
    # loss_name = "ce"
    with_mse_loss = False
    logs_dir = "./logs-mse-%s" % loss_name if with_mse_loss else "./logs-%s" % loss_name
    weights = os.path.join(logs_dir, 'checkpoints')
    best_models = os.path.join(weights, 'best_models')
    submit = os.path.join(logs_dir, 'submit')
    model_name = "bninception_bcelog"
    # model_name = "inceptionresnetv2"
    gpu_id = "0"

    # basic config
    num_classes = 28
    img_width = 512
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

    # thresholds
    # # 0.6717
    # thresholds = 0.15
    # # 0.6870
    # thresholds = 0.2
    # 0.7096
    thresholds = 0.4
    # # 0.6646
    # thresholds = np.array([0.407, 0.441, 0.161, 0.145, 0.299, 0.129, 0.25, 0.414, 0.01, 0.028, 0.021, 0.125,
    #                        0.113, 0.387, 0.602, 0.001, 0.137, 0.199, 0.176, 0.25, 0.095, 0.29, 0.159, 0.255,
    #                        0.231, 0.363, 0.117, 0., ])
    # # 0.7038
    # thresholds = np.array([0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.2, 0.2, 0.2, 0.4,
    #                        0.4, 0.4, 0.4, 0.2, 0.4, 0.4, 0.4, 0.4, 0.2, 0.4, 0.4, 0.4,
    #                        0.4, 0.4, 0.4, 0.2, ])


config = DefaultConfigs()
