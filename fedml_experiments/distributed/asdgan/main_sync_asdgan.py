import argparse
import logging
import os
import random
import socket
import sys
import datetime

import numpy as np
import psutil
import setproctitle
import torch
import wandb
from mpi4py import MPI

# add the FedML root directory to the python path

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_api.distributed.utils.gpu_mapping import mapping_processes_to_gpu_device_from_yaml_file
from fedml_api.data_preprocessing.brats.data_loader_asdgan import load_partition_data_distributed_brats
from fedml_api.model.cv.asdgan import DadganModelG, DadganModelD
from fedml_api.distributed.asdgan.AsDGanAPI import FedML_init, FedML_AsDGan_distributed
from fedml_api.distributed.asdgan.utils import count_parameters


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='asdgan', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--backbone', type=str, default='resnet',
                        help='employ with backbone (default: xception)')

    parser.add_argument('--backbone_pretrained', type=str2bool, default=False,
                        help='pretrained backbone (default: False)')

    parser.add_argument('--backbone_freezed', type=str2bool, default=False,
                        help='Freeze backbone to extract features only once (default: False)')

    parser.add_argument('--dataset', type=str, default='brats', metavar='N',
                        choices=['brats', 'brats_t2', 'brats_t1c', 'brats_flair'],
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='./../../../data/brats',
                        help='data directory (default = ./../../../data/brats)')
 
    parser.add_argument('--checkname', type=str, default='asdgan', help='set the checkpoint name')

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--sample_method', type=str, default='uniform', metavar='N',
                        choices=['uniform', 'balance'],
                        help='how to partition the dataset on local workers')

    parser.add_argument('--client_num_in_total', type=int, default=3, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=3, metavar='NN',
                        help='number of workers')

    parser.add_argument('--save_client_model', type=str2bool, default=True,
                        help='whether to save locally trained model by clients (default: True')

    parser.add_argument('--batch_size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 32)')

    parser.add_argument('--sync_bn', type=str2bool, default=False,
                        help='whether to use sync bn (default: auto)')

    parser.add_argument('--freeze_bn', type=str2bool, default=False,
                        help='whether to freeze bn parameters (default: False)')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='adam or sgd')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    # parser.add_argument('--lr_scheduler', type=str, default='poly',
    #                     choices=['poly', 'step', 'cos'],
    #                     help='lr scheduler mode: (default: poly)')

    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')

    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')

    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')

    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')

    parser.add_argument('--epochs', type=int, default=200, metavar='EP',
                        help='how many epochs will be trained')

    # parser.add_argument('--comm_round', type=int, default=200,
    #                     help='how many round of communications we shoud use')

    parser.add_argument('--evaluation_frequency', type=int, default=1,
                        help='Frequency of model evaluation on training dataset (Default: every 5th round)')

    parser.add_argument('--gpu_mapping_file', type=str, default="gpu_mapping.yaml",
                        help='the gpu utilization file for servers and clients. If there is no \
                        gpu_util_file, gpu will not be used.')

    parser.add_argument('--gpu_mapping_key', type=str, default="mapping_default",
                        help='the key in gpu utilization file')

    parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
    parser.add_argument('--gan_mode', type=str, default='vanilla',
                        help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
    parser.add_argument('--netD', type=str, default='basic',
                        help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
    parser.add_argument('--netG', type=str, default='unet_256',
                        help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
    parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
    parser.add_argument('--norm', type=str, default='batch', help='instance normalization or batch normalization [instance | batch | none]')
    parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--no_dropout', default=False, action='store_true', help='no dropout for the generator')
    parser.add_argument('--lambda_L1', type=float, default=20.0, help='weight for L1 loss')
    parser.add_argument('--lambda_perceptual', type=float, default=1, help='weight for perceptual loss')
    parser.add_argument('--lambda_G', type=float, default=0.2, help='weight for dadgan G')
    parser.add_argument('--lambda_D', type=float, default=0.1, help='weight for dadgan D')
    parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')
    parser.add_argument('--continue_train', default=False, action='store_true', help='continue training: load the latest model')
    parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
    parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, for lr_policy == linear')
    parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate for linear policy, or T_max for cos policy')
    parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero, for lr_policy == linear')
    parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations, for lr_policy == step')
    parser.add_argument('--lr_decay_gamma', type=float, default=0.1, help='gamma for lr_policy == step')
    parser.add_argument('--verbose', default=False, action='store_true', help='if specified, print more debugging information')

    parser.add_argument('--brain_mask_input', type=int, default=0, choices=[0, 1], help='skull input if 0, else noise brain input')

    args = parser.parse_args()

    args.isTrain = True

    return args


def load_data(process_id, args, dataset_name):
    if dataset_name[:5].lower() == "brats":
        data_loader = load_partition_data_distributed_brats

    train_data_num, train_data_set_global, test_data_loader_global, local_data_num, train_data_local, test_data_local, class_num = data_loader(
        process_id, args.dataset, args.data_dir, args.partition_method, args.client_num_in_total, args.batch_size, args.sample_method, args.brain_mask_input==1)
    dataset = [train_data_num, train_data_set_global, test_data_loader_global, local_data_num, train_data_local, test_data_local,
               class_num]

    return dataset


def create_model(process_id, args, model_name, device):
    if model_name.lower() == 'asdgan':
        if process_id == 0:  # server : generator
            model = DadganModelG(args, device)
            num_params = count_parameters(model.netG)
            logging.info("G Model Size = {0} M".format(str(num_params)))
        else:
            model = DadganModelD(args, device)
            num_params = count_parameters(model.netD)
            logging.info("D Model Size = {0} M".format(str(num_params)))
    else:
        raise ('Not Implemented Error')

    return model


if __name__ == "__main__":
    now = datetime.datetime.now()
    time_start = now.strftime("%Y-%m-%d %H:%M:%S")
    
    logging.info("Executing AsynDGAN at time: {0}".format(time_start))
    
    # initialize distributed computing (MPI)
    comm, process_id, worker_number = FedML_init()

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    # customize the process name
    str_process_name = "AsynDGAN (distributed):" + str(process_id)
    setproctitle.setproctitle(str_process_name)


    ####Server side: process_id=0

    # customize the log format
    logging.basicConfig(filename='info.log',
                        level=logging.INFO,
                        format=str(process_id) + ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')

    hostname = socket.gethostname()
    logging.info("#############process ID = " + str(process_id) +
                 ", host name = " + hostname + "########" +
                 ", process ID = " + str(os.getpid()) +
                 ", process Name = " + str(psutil.Process(os.getpid())))
    logging.info(args)
    # initialize the wandb machine learning experimental tracking platform (https://www.wandb.com/).
    if process_id == 0:
        wandb.init(
            project="AsynDGAN",
            name=args.dataset + '-' + str(args.sample_method) + "-" + str(args.client_optimizer) + "-e" + str(args.epochs) + "-lr" + str(args.lr),
            config=args
        )

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = True  # fixed input size [256, 256]

    logging.info("process_id = %d, size = %d" % (process_id, worker_number))
    server_device = mapping_processes_to_gpu_device_from_yaml_file(process_id, worker_number, args.gpu_mapping_file, args.gpu_mapping_key)

    # load data
    dataset = load_data(process_id, args, args.dataset)
    [train_data_num, train_data_global, test_data_global, data_local_num_dict,
     train_data_local_dict, test_data_local_dict, class_num] = dataset

    # create model.
    # process various
    model = create_model(process_id, args, model_name=args.model, device=server_device)

    logging.info("Calling FedML AsynDGAN")

    # start "federated GAN (FedGan)"
    FedML_AsDGan_distributed(process_id, worker_number, device, comm, model, train_data_num, train_data_global, test_data_global, data_local_num_dict,
                             train_data_local_dict, test_data_local_dict, args)


