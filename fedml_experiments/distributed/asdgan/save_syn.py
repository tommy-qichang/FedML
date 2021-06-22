import matplotlib
matplotlib.use('agg')
import os
import sys
import argparse
import h5py
import random
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import torch
import torch.utils.data as data

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_api.data_preprocessing.brats.data_loader_asdgan import TestDataset
from fedml_api.data_preprocessing.brats.data_utility import init_transform
# from options.test_options import TestOptions
from fedml_api.model.cv.asdgan import DadganModelG
from fedml_api.distributed.asdgan.utils import float_to_uint_img


def add_args():
    """
    return a parser added with args required by fit
    """
    parser = argparse.ArgumentParser()
    # Test settings
    parser.add_argument('--model', type=str, default='asdgan', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='brats', metavar='N',
                        choices=['brats', 'brats_t2', 'brats_t1c', 'brats_flair'],
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='./../../../data/brats',
                        help='data directory (default = ./../../../data/brats)')

    parser.add_argument('--checkname', type=str, default='asdgan', help='set the checkpoint name')

    parser.add_argument('--batch_size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 32)')

    parser.add_argument('--epoch', type=int, default=20, metavar='EP',
                        help='which epoch model will be loaded')

    parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
    parser.add_argument('--gan_mode', type=str, default='vanilla',
                        help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
    parser.add_argument('--netG', type=str, default='unet_256',
                        help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
    parser.add_argument('--norm', type=str, default='batch', help='instance normalization or batch normalization [instance | batch | none]')
    parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--no_dropout', default=False, action='store_true', help='no dropout for the generator')

    parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
    parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
    parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
    parser.add_argument('--num_test', type=int, default=float("inf"), help='how many test images to run')
    # To avoid cropping, the load_size should be the same as crop_size
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--load_iter', type=int, default=0)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--load_filename', type=str)
    parser.add_argument('--GPUid', type=str, default='0')
    parser.add_argument('--verbose', default=False, action='store_true', help='if specified, print more debugging information')

    args = parser.parse_args()

    args.isTrain = False

    return args


def create_dataset(args, channel, test_bs):
    transforms_test = ["Resize", "ToTensorScale",  "Normalize"]
    transforms_args_test = {
        "Resize": [256],
        "ToTensorScale": ['float'],
        "Normalize": [0.5, 0.5]
    }
    transform_test = init_transform(transforms_test, transforms_args_test)

    h5_test = os.path.join(args.data_dir, 'General_format_BraTS18_train_2d_3ch_new.h5')

    test_ds = TestDataset(h5_test,
                          channel=channel,
                          path="train",
                          sample_rate=0.1,
                          transforms=transform_test)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)
    return test_dl


def save_data(A, B, fake_B, key, file):

    real_img = B
    syn_img = fake_B
    label = A

    syn_img = float_to_uint_img(syn_img, (240, 240), 1)
    real_img = float_to_uint_img(real_img, (240, 240), 1)
    # label = float_to_uint_img(label, (240, 240), 0)

    # restore labels
    values = np.unique(label)
    maxv = len(values) - 1
    for v in values[::-1]:
        label[label == v] = maxv
        maxv -= 1
    label = label.astype("uint8")
    if len(label.shape) == 3:
        label = label[0]
    label = resize(label, (240, 240), order=0, preserve_range=True)

    # if np.sum(label>0) < 10:
        # print("skip this seg")
        #     return

    # remove skull label
    gt = np.copy(label)
    assert(gt.max() == 4)
    gt[gt==4] = 0
    gt[gt==3] = 4  # no label 3

    save_type = "train"
    file.create_dataset(f"{save_type}/{key}/data", data=syn_img)
    file.create_dataset(f"{save_type}/{key}/label", data=gt)
    file.create_dataset(f"{save_type}/{key}/labels_with_skull", data=label)
    file.create_dataset(f"{save_type}/{key}/reference_real_image_please_dont_use", data=real_img)


def plot_syn(A, B, fake_B, key, save_dir, mod_names):
    nc = B.shape[0]
    num_r = 1
    num_c = 1 + 2 * nc
    ctr = 0

    syn_img = float_to_uint_img(fake_B, (240, 240), 1)

    label = A
    values = np.unique(label)
    maxv = len(values)-1
    for v in values[::-1]:
        label[label==v] = maxv
        maxv -= 1

    label = label.astype("uint8")
    if len(label.shape) == 3:
        label = label[0]
    label = resize(label, (240, 240), order=0, preserve_range=True)

    realdata = float_to_uint_img(B, (240, 240), 1)

    n_rot = 0

    if ctr == 0:
        plt.figure(figsize=(20, 10))
        showtitle = True

    ctr += 1
    plt.subplot(num_r, num_c, ctr)
    plt.imshow(np.rot90(label, n_rot), cmap="gray")
    if showtitle:
        plt.title("Label")
    plt.axis('off')

    for k in range(nc):
        ctr += 1
        plt.subplot(num_r, num_c, ctr)
        plt.imshow(np.rot90(syn_img[k], n_rot), cmap="gray")
        if showtitle:
            plt.title(mod_names[k])
        plt.axis('off')

    for k in range(nc):
        ctr += 1
        plt.subplot(num_r, num_c, ctr)
        plt.imshow(np.rot90(realdata[k], n_rot), cmap="gray")
        if showtitle:
            plt.title(mod_names[k])
        plt.axis('off')

    if ctr == num_r * num_c:
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "images-"+key))
        plt.close()
    else:
        showtitle = False


if __name__ == '__main__':

    args = add_args()
    print(args)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = True  # fixed input size [256, 256]

    device = torch.device("cuda:{}".format(args.GPUid)) if torch.cuda.is_available() else torch.device("cpu")

    model = DadganModelG(args, device)

    # hard-code some parameters for test

    if 't2' in args.dataset:
        channel = 1
        mod_names = ['T2']
    elif 't1' in args.dataset:
        channel = 0
        mod_names = ['T1c']
    elif 'flair' in args.dataset:
        channel = 2
        mod_names = ['Flair']
    else:
        channel = None
        mod_names = ['T1c', 'T2', 'Flair']

    dataloader = create_dataset(args, channel, args.batch_size)  # create a dataset given opt.dataset_mode and other options

    model.setup(args)  # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(args.results_dir, args.checkname, '%s_%s' % (args.phase, args.epoch))  # define the website directory

    ## ??? the output is blank image in eval mode with dropout disabled
    # model.eval()

    folder_name = f"{args.dataset}_{args.netG}_epoch{args.epoch}_batch_size{args.batch_size}"
    save_dir = os.path.join(web_dir, folder_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # file = h5py.File(os.path.join(web_dir, f"{folder_name}.h5"), 'w')

    for i, data in enumerate(dataloader):
        if i >= args.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        with torch.no_grad():
            syn_img = model.forward()

        img = data['B'].cpu().detach().numpy()

        for j in range(img.shape[0]):
            sys.stdout.flush()
            plot_syn(data['A'].cpu().detach().numpy()[j], img[j], syn_img[j], data['key'][j], save_dir, mod_names)
            # save_data(data['A'].cpu().detach().numpy()[j], img[j], syn_img[j], data['key'][j], file)

        print(f"{i} processing")

    # file.close()
