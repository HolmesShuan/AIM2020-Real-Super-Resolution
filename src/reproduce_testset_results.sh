# convert RGBA png to RGB images
python convert_images.py
# run basic models, self-ensemble + crop-ensemble
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --model EDSR --template EDSR_paper --data_test Demo --scale 2 --save AIM_EDSR_x2_TEST --test_only --dir_demo ../TestLRX2/TestLR --pre_train ../experiment/AIM_EDSR/model/AIM_EDSR_X2.pt --n_GPUs 4 --chop --chop-size 500 --shave-size 50 --save_results --self_ensemble
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --model DRLN --data_test Demo --scale 2 --save AIM_DRLN_x2_TEST --test_only --dir_demo ../TestLRX2/TestLR --pre_train ../experiment/AIM_DRLN/model/AIM_DRLN_X2.pt --n_GPUs 4 --chop --chop-size 450 250 --shave-size 50 100 --save_results --self_ensemble
CUDA_VISIBLE_DEVICES=2,3 python main.py --model DDDet --n_resblocks 32 --n_feats 128 --res_scale 1.0 --data_test Demo --scale 2 --save AIM_DDDet_x2_TEST --test_only --dir_demo ../TestLRX2/TestLR --pre_train ../experiment/AIM_DDet/model/AIM_DDET_X2.pt --n_GPUs 2 --chop --chop-size 450 450 450 450 --shave-size 80 80 10 10 --save_results
CUDA_VISIBLE_DEVICES=0,1 python main.py --model WDDet --n_resblocks 40 --n_feats 128 --res_scale 1.0 --data_test Demo --scale 2 --save AIM_WDDet_x2_TEST --test_only --dir_demo ../TestLRX2/TestLR --pre_train ../experiment/AIM_WDDet/model/AIM_WDDET_X2.pt --n_GPUs 2 --chop --chop-size 450 450 450 450 --shave-size 80 80 10 10 --save_results
# remove log/config files, rename SR images
python rename_output_images.py
# model-ensemble
python model_ensemble.py