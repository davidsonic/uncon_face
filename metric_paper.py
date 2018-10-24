from __future__ import division
import skimage.measure as cmp
import os
from skimage.io import imread
from skimage.transform import resize

BASE_DIR='/home/jiali/program/uncon_face/results2'

cycleGAN=os.path.join(BASE_DIR,'cycleGAN_nature2style')
portGAN=os.path.join(BASE_DIR,'portraitGAN_nature2style')
starGAN=os.path.join(BASE_DIR, 'stargan_nature2style')

truth=os.path.join(BASE_DIR,'two_style_nature2style_groundtruth')

cycle_psnr_total=0
cycle_ssim_total=0
cycle_mse_total=0

port_psnr_total=0
port_ssim_total=0
port_mse_total=0

star_psnr_total=0
star_ssim_total=0
star_mse_total=0

length=len(os.listdir(truth))
for i in range(length):
    print i
    # path
    truth_file=os.path.join(truth,str(i)+'.png')

    cycle_file=os.path.join(cycleGAN, str(i)+'.png')
    port_file=os.path.join(portGAN, str(i)+'.png')
    star_file=os.path.join(starGAN,str(i)+'.png')

    # read and resize
    truth_file=imread(truth_file)
    cycle_file=imread(cycle_file)
    port_file=imread(port_file)
    star_file=imread(star_file)

    truth_file=resize(truth_file,(512,512))
    cycle_file=resize(cycle_file, (512,512))
    port_file=resize(port_file ,(512,512))
    star_file=resize(star_file, (512,512))

    # metric
    cycle_psnr=cmp.compare_psnr(truth_file, cycle_file)
    port_psnr =cmp.compare_psnr(truth_file, port_file)
    star_psnr=cmp.compare_psnr(truth_file, star_file)
    cycle_psnr_total+= cycle_psnr
    port_psnr_total+= port_psnr
    star_psnr_total+=port_psnr
    print 'cycle_psnr: {}, port_psnr: {}, star_psnr: {}'.format(cycle_psnr, port_psnr, star_psnr)

    cycle_mse=cmp.compare_mse(truth_file, cycle_file)
    port_mse=cmp.compare_mse(truth_file, port_file)
    star_mse=cmp.compare_mse(truth_file, star_file)
    cycle_mse_total+=cycle_mse
    port_mse_total+=port_mse
    star_mse_total+=star_mse
    print 'cycle_mse: {}, port_mse: {}, star_mse: {}'.format(cycle_mse, port_mse, star_mse)

    cycle_ssim=cmp.compare_ssim(truth_file, cycle_file, multichannel=True)
    port_ssim=cmp.compare_ssim(truth_file, port_file, multichannel=True)
    star_ssim=cmp.compare_ssim(truth_file, star_file, multichannel=True)
    cycle_ssim_total+=cycle_ssim
    port_ssim_total+=port_ssim
    star_ssim_total+=star_ssim
    print 'cycle_ssim: {}, port_ssim: {}, star_ssim: {}'.format(cycle_ssim, port_ssim, star_ssim)
    print '-'*20

# statistics
print 'average psnr: cycle_psnr: {}, port_psnr: {}, star_psnr: {}'.format(cycle_psnr_total/length, port_psnr_total/length, star_psnr_total/length)
print 'average mse: cycle_mse: {}, port_mse: {}, star_mse: {}'.format(cycle_mse_total/length, port_mse_total/length, star_mse_total/length)
print 'average ssim: cycle_ssim: {}, port_ssim: {}, star_ssim: {}'.format(cycle_ssim_total/length, port_ssim_total/length, star_ssim_total/length)


