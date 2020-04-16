#!/bin/bash

# Script to reproduce results with frozen weights.
# For headless remote server using ssh (install xvfb):
# xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python train.py [parameters]

for ((i=0;i<10;i+=2))
do
python train.py \
	--vae t \
	--ndim 64 \
	--img-stack 4 \
	--freeze \
	--tb \
	--rl-path contin_vae/pretrained_vae_64_stack4_conti.ckpt\
	--seed $i \
	--title vae_freeze_64_stack4

python train.py \
	--infomax t \
	--ndim 64 \
	--freeze \
	--tb \
	--rl-path contin_infomax/pretrained_infomax_64_stack4_earlystop_action_conti.ckpt
	--seed $i \
	--title infomax_freeze_64_stack4

python train.py \
	--raw t \
	--ndim 64 \
	--img-stack 4 \
	--freeze \
	--tb \
	--seed $i \
	--title rawpixel_freeze_64_stack4

done
