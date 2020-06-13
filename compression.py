import argparse
import textwrap
import os
import time
import tensorflow as tf
import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser(prog='compression',
                                     description='Script used to run Learned Image Compression models',
                                     formatter_class=argparse.RawTextHelpFormatter
                                     )
    parser.add_argument('paper',
                        nargs='+',
                        help=textwrap.dedent('''\
                                 Paper from which all models will be executed at all available quality levels
                                 
                                 paper = 1 -> "Full Resolution Image Compression with Recurrent Neural Networks"
                                 paper = 2 -> "Efficient Nonlinear Transforms for Lossy Image Compression", J. Ballé - PCS 2018
                                                    b2018-leaky_relu-128-[1-4]
                                                    b2018-leaky_relu-192-[1-4]
                                                    b2018-gdn-128-[1-4]
                                                    b2018-gdn-192-[1-4]
                                 paper = 3 -> "Variational Image Compression with a Scale Hyperprior", J. Ballé - ICLR 2018
                                                    bmshj2018-factorized-mse-[1-8]
                                                    bmshj2018-factorized-msssim-[1-8]
                                                    bmshj2018-hyperprior-mse-[1-8]
                                                    bmshj2018-hyperprior-msssim-[1-8]
                                 paper = 4 -> "Joint Autoregressive and Hierarchical Priors for Learned Image Compression", D. Minnen - NeurIPS 2018
                                                    mbt2018-mean-mse-[1-8]
                                                    mbt2018-mean-msssim-[1-8]
                                 paper = 5 -> "Learned Image Compression with Discretized Gaussian Mixture Likelihoods and Attention Modules", Zhengxue Cheng - CVPR 2020
                                                    MS-SSIM-lambda-14
                                     ''')
                        )
    return parser.parse_args().paper

def paper2():
    models = ['b2018-leaky_relu-128-', 'b2018-leaky_relu-192-', 'b2018-gdn-128-', 'b2018-gdn-192-']
    for model in models:
        for quality in range(1, 5):
            for image in range(1, 25):
                print('################\nCOMPRESSING\nMODEL = ' + model + str(quality)+'\nIMAGE = '+ str(image) + '.png\n################')
                t1 = time.perf_counter()
                os.system('python modeltensorflow/tfci.py compress ' + model + str(quality) + ' images/' + str(image) + '.png')
                t2 = time.perf_counter()
                print('COMPRESSING TIME = ' + str(t2 - t1))
                if not os.path.exists('modeltensorflow/Efficient_Nonlinear_Transforms_for_Lossy_Image_Compression/' + model[:-1] + '/' + model + str(quality) + '/'):
                    os.makedirs('modeltensorflow/Efficient_Nonlinear_Transforms_for_Lossy_Image_Compression/' + model[:-1] + '/' + model + str(quality) + '/')
                os.replace('images/' + str(image) + '.png.tfci','modeltensorflow/Efficient_Nonlinear_Transforms_for_Lossy_Image_Compression/' + model[:-1] + '/' + model + str(quality) + '/' + str(image) + '.tfci')
                print('################\nDECOMPRESSING\nMODEL = ' + model + str(quality) + '\nIMAGE = ' + str(image) + '.png\n################')
                t1 = time.perf_counter()
                os.system('python modeltensorflow/tfci.py decompress modeltensorflow/Efficient_Nonlinear_Transforms_for_Lossy_Image_Compression/' + model[:-1] + '/' + model + str(quality) + '/' + str(image) + '.tfci')
                t2 = time.perf_counter()
                print('DECOMPRESSING TIME = ' + str(t2 - t1))


def ms_ssim(path_original,path_reconstructed):
    im1 = tf.decode_png(path_original)
    im2 = tf.decode_png(path_reconstructed)
    im1 = tf.image.convert_image_dtype(im1, tf.float32)
    im2 = tf.image.convert_image_dtype(im2, tf.float32)
    psnr = tf.image.psnr(im1, im2, max_val=1.0)
    print(psnr)


def main(papers):
    model3 = ['bmshj2018-factorized-mse-','bmshj2018-factorized-msssim-','bmshj2018-hyperprior-mse-','bmshj2018-hyperprior-msssim-']
    model4 = ['mbt2018-mean-mse-','mbt2018-mean-msssim-']
    for paper in papers:
        if paper == '1':
            print('################################################################')
            print('Full Resolution Image Compression with Recurrent Neural Networks')
            print('################################################################')
            #for image in range(1,25):
            #os.system('python model1/image_encoder/encoder.py --input_image=1.png --output_codes=1codes.npz --iteration=15 --model=residual_gru.pb')
            #os.system('python model1/image_encoder/decoder.py --input_codes=1codes.npz --output_directory=/tmp/decoded/ --model=residual_gru.pb')
        elif paper == '2':
            print('##########################################################')
            print('Efficient Nonlinear Transforms for Lossy Image Compression')
            print('##########################################################')
            paper2()
        elif paper == '3':
            #os.system('conda deactivate')
            #os.system('conda activate modelgoogle')
            print('#####################################################')
            print('Variational Image Compression with a Scale Hyperprior')
            print('#####################################################')
            for image in range(1,25):
                os.system('python modeltesorflow/tfci.py compress bmshj2018-hyperprior-msssim-4 modeltensorflow/'+str(image)+'.png')
                os.system('python modeltensorflow/tfci.py decompress model2-3/'+str(image)+'.png.tfci ')
            #os.system('conda deactivate')
        elif paper == '5':
            print('#############################################################################################')
            print('Learned Image Compression with Discretized Gaussian Mixture Likelihoods and Attention Modules')
            print('#############################################################################################')
            os.system('conda deactivate')
            os.system('conda activate model4')
            os.system('python model5/encoder.py')
            os.system('python model5/decoder.py')
            os.system('conda deactivate')


if __name__ == '__main__':
    main(parse_arguments())