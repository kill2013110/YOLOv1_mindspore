

python /home/ma-user/work/TOOD_MindSpore/train.py --device_id=0 --device_num=1 --platform=Ascend --train_path=/home/ma-user/work/coco_val/images/val2017 --anno_path=/home/ma-user/work/coco_val/annotations/instances_val2017.json --ckpt_save_path=/home/ma-user/work/TOOD_MindSpore/data/

--pretrain_ckpt_path=/home/ma-user/work/TOOD_MindSpore/resnet50_pth2ms_jit.ckpt

# run_standalone_train_gpu.sh coco_val/images/val2017 coco_val/annotations/instances_val2017.json TOOD_MindSpore/resnet50_pth2ms_jit.ckpt TOOD_MindSpore/data/ 0 1