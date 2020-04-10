#!/bin/bash

# Script to reproduce results. No frozen weights.
# For headless remote server using ssh (install xvfb):
# xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python train.py [parameters]

for ((i=0;i<10;i+=1))
do 
	python train.py \
	--vae t \
	--ndim 32 \
	--img-stack 4 \
	--tb \
	--rl-path contin_vae/pretrained_vae_32_stack4_conti.ckpt \
	--seed $i \
	--title vae_32_stack4

	python train.py \
	--infomax t \
	--ndim 32 \
	--tb \
	--rl-path contin_infomax/pretrained_infomax_32_stack4_earlystop_action_conti.ckpt \
	--seed $i \
	--title infomax_32_stack4

	python train.py \
	--raw t \
	--ndim 32 \
	--img-stack 4 \
	--tb \
	--seed $i \
	--title rawpixel_32_stack4
done
