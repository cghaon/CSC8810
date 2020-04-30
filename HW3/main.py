import os

from GAN import GAN
from WGAN import WGAN

from utils import show_all_variables
from utils import check_folder

import tensorflow as tf
import argparse

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)
    #GAN OR WGAN
    parser.add_argument('--gan_type', type=str, default='GAN',
                        choices=['GAN', 'WGAN'],
                        help='The type of GAN')
    #data set
    parser.add_argument('--dataset', type=str, default='cifar-10', choices=['cifar-10'],
                        help='The name of dataset')
    #epoch 
    parser.add_argument('--epoch', type=int, default=100, help='The number of epochs to run')
    #batch size 100
    parser.add_argument('--batch_size', type=int, default=100, help='The size of batch')
    #noise size 100
    parser.add_argument('--z_dim', type=int, default=100, help='Dimension of noise vector')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    #save the results
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    #check if the argument valid
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --epoch
    assert args.epoch >= 1, 'number of epochs must be larger than or equal to one'

    # --batch_size
    assert args.batch_size >= 1, 'batch size must be larger than or equal to one'

    # --z_dim
    assert args.z_dim >= 1, 'dimension of noise vector must be larger than or equal to one'

    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    #make sure there is args
    if args is None:
      exit()

    # open session
    #the list of possible models
    models = [GAN, WGAN]
    #set config
    config=tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # declare instance for GAN

        gan = None
        for model in models:
            if args.gan_type == model.model_name:
                gan = model(sess,
                            epoch=args.epoch,
                            batch_size=args.batch_size,
                            z_dim=args.z_dim,
                            dataset_name=args.dataset,
                            checkpoint_dir=args.checkpoint_dir,
                            result_dir=args.result_dir,
                            log_dir=args.log_dir)
        #can not find the gan
        if gan is None:
            raise Exception(args.gan_type, + "is invalid")

        # build graph
        gan.build_model()

        # show network architecture
        show_all_variables()

        # launch the graph in a session
        gan.train()
        print(" [*] Training finished!")

        # visualize learned generator
        gan.visualize_results(args.epoch-1)
        print(" [*] Testing finished!")
         #calculate the Inception score
        gan.calculate_is()

if __name__ == '__main__':
    main()
