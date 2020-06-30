import argparse
import textwrap
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from math import log10, sqrt
from collections import defaultdict
import cv2
import numpy as np

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

def is_rgb_psnr(path_original,path_recons):
    original = cv2.imread(path_original, 1)
    compressed = cv2.imread(path_recons, 1)
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def is_y_psnr(path_original,path_recons):
    original = cv2.cvtColor(cv2.imread(path_original, 1), cv2.COLOR_BGR2YCR_CB)#Abre imagens em BGR
    compressed = cv2.cvtColor(cv2.imread(path_recons, 1), cv2.COLOR_BGR2YCR_CB)#e traduz para Y'CrCb
    y_original = original[:,:,0]       # Extrai somente o Y
    y_compressed = compressed[:,:,0]
    mse = np.mean((y_original - y_compressed) ** 2)
    if (mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def is_ycrcb_psnr(path_original,path_recons):
    original = cv2.cvtColor(cv2.imread(path_original, 1), cv2.COLOR_BGR2YCR_CB)#Abre imagens em BGR
    compressed = cv2.cvtColor(cv2.imread(path_recons, 1), cv2.COLOR_BGR2YCR_CB)#e traduz para Y'CrCb
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def is_psnr(path_original,path_recons):
    im1 = tf.io.decode_png(open(path_original,'rb').read(),channels=3)
    im2 = tf.io.decode_png(open(path_recons,'rb').read(),channels=3)
    im1 = tf.image.convert_image_dtype(im1, tf.float32)
    im2 = tf.image.convert_image_dtype(im2, tf.float32)
    psnr = tf.image.psnr(im1, im2, max_val=1.0)
    with tf.compat.v1.Session() as sess:
        return psnr.eval()


def is_ms_ssim(path_original, path_recons):
    im1 = tf.io.decode_png(open(path_original,'rb').read(),channels=1)
    im2 = tf.io.decode_png(open(path_recons,'rb').read(),channels=1)
    im1 = tf.image.convert_image_dtype(im1, tf.float32)
    im2 = tf.image.convert_image_dtype(im2, tf.float32)
    ms_ssim = tf.image.ssim_multiscale(im1, im2,255)
    with tf.compat.v1.Session() as sess:
        return ms_ssim.eval()

def paper_tf(paper,models,quality_range):
    models_metrics = defaultdict(dict)
    for model in models:
        for quality in quality_range:
            total_tc = 0
            total_td = 0
            total_psnr = 0
            total_ms_ssim = 0
            total_ms_ssim_db = 0
            total_bpp = 0
            for image in range(1, 25):
                ### PATHS TO ORIGINAL, LATENT REPRESENTATION AND RECONSTRUCTED DATASET IMAGES ###
                path_original = 'images/' + str(image) + '.png'
                path_latent = 'modeltensorflow/' + paper + '/' + model[:-1] + '/' + model + quality + '/' + str(image) + '.tfci'
                path_recons = 'modeltensorflow/' + paper + '/' + model[:-1] + '/' + model + quality + '/' + str(image) + '.tfci.png'
                ### COMPRESSING ###
                t1 = time.perf_counter()
                os.system('python modeltensorflow/tfci.py compress ' + model + quality + ' images/' + str(image) + '.png')
                t2 = time.perf_counter()
                tc = t2-t1
                ### MOVES COMPRESSED IMAGE TO ITS DIRECTORY ###
                if not os.path.exists('modeltensorflow/' + paper + '/' + model[:-1] + '/' + model + quality + '/'):
                    os.makedirs('modeltensorflow/' + paper + '/' + model[:-1] + '/' + model + quality + '/')
                os.replace('images/' + str(image) + '.png.tfci','modeltensorflow/' + paper + '/' + model[:-1] + '/' + model + quality + '/' + str(image) + '.tfci')
                ### DECOMPRESSING ###
                t1 = time.perf_counter()
                os.system('python modeltensorflow/tfci.py decompress modeltensorflow/' + paper + '/' + model[:-1] + '/' + model + quality + '/' + str(image) + '.tfci')
                t2 = time.perf_counter()
                td = t2-t1
                ### QUALITY METRICS ###
                total_tc += tc
                total_td += td
                psnr = is_psnr(path_original,path_recons)
                ycrcb_psnr = is_ycrcb_psnr(path_original, path_recons)
                y_psnr = is_y_psnr(path_original,path_recons)
                rgb_psnr = is_rgb_psnr(path_original,path_recons)
                total_psnr += psnr
                ms_ssim = is_ms_ssim(path_original,path_recons)
                total_ms_ssim += ms_ssim
                ms_ssim_db = -10 * log10(1 - ms_ssim)
                total_ms_ssim_db += ms_ssim_db
                bpp = (os.path.getsize(path_latent)*8)/(768*512)
                total_bpp += bpp
                ### PRINT LOG ###
                print('\n########################################')
                print('MODEL = ' + model + quality)
                print('IMAGE = '+ str(image) + '.png')
                print('COMPRESSING TIME = ' + str(tc) )
                print('DECOMPRESSING TIME = ' + str(td))
                print('PSNR TF= ' + str(psnr))
                print('PSNR YCbCR= ' + str(ycrcb_psnr))
                print('PSNR Y = ' + str(y_psnr))
                print('PSNR RGB = ' + str(rgb_psnr))
                print('MS-SSIM = ' + str(ms_ssim))
                print('MS-SSIM(DB) = ' + str(ms_ssim_db))
                print ('BPP = ' + str(bpp))
                print('########################################\n')
            print('\n########################################')
            print('MODEL = ' + model + quality)
            print('KODADK DATASET' )
            print('TOTAL COMPRESSING TIME = ' + str(total_tc/24))
            print('TOTAL DECOMPRESSING TIME = ' + str(total_td/24))
            print('TOTAL PSNR = ' + str(total_psnr/24))
            print('TOTAL MS-SSIM = ' + str(total_ms_ssim/24))
            print('TOTAL MS-SSIM(DB) = ' + str(total_ms_ssim_db/24))
            print('TOTAL BPP = ' + str(total_bpp/24))
            print('########################################\n')
            models_metrics[model + quality]['PSNR'] = (total_psnr/24)
            models_metrics[model + quality]['TC'] = (total_tc/24)
            models_metrics[model + quality]['TD'] = (total_td/24)
            models_metrics[model + quality]['PSNR'] = (total_psnr/24)
            models_metrics[model + quality]['MS-SSIM(db)'] = (total_ms_ssim_db/24)
            models_metrics[model + quality]['BPP'] = (total_bpp/24)
        print(models_metrics)







def main(papers):
    paper1 = ['Full_Resolution_Image_Compression_with_Recurrent_Neural_Networks']
    paper2 = ['Efficient_Nonlinear_Transforms_for_Lossy_Image_Compression',
               'b2018-leaky_relu-128-',
               'b2018-leaky_relu-192-',
               'b2018-gdn-128-',
               'b2018-gdn-192-',
               ['1','2','3','4']
               ]
    paper3 = ['Variational_Image_Compression_with_a_Scale_Hyperprior',
               'bmshj2018-factorized-mse-',
               'bmshj2018-factorized-msssim-',
               'bmshj2018-hyperprior-mse-',
               'bmshj2018-hyperprior-msssim-',
               ['1','2','3','4','5','6','7','8']
               ]
    paper4 = ['Joint_Autoregressive_and_Hierarchical_Priors_for_Learned_Image_Compression',
               'mbt2018-mean-mse-',
               'mbt2018-mean-msssim-',
               ['1', '2', '3', '4', '5', '6', '7', '8']
               ]
    paper5 = ['Learned_Image_Compression_with_Discretized_Gaussian_Mixture_Likelihoods_and_Attention_Modules']
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
            paper_tf(paper2[0], paper2[1:5], paper2[5])
        elif paper == '3':
            #os.system('conda deactivate')
            #os.system('conda activate modelgoogle')
            print('#####################################################')
            print('Variational Image Compression with a Scale Hyperprior')
            print('#####################################################')
            paper_tf(paper3[0], paper3[1:5], paper3[5])
        elif paper == '4':
            print('##########################################################################')
            print('Joint Autoregressive and Hierarchical Priors for Learned Image Compression')
            print('##########################################################################')
            paper_tf(paper4[0], paper4[1:3], paper4[3])
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