# CDCD
 Conditioned Discrete Contrastive Diffusion

### To run the sample-wise contrastive diffusion for the image generation task, the env setup is the same as previous step-wise constrastive diffusion setting.
Compared to the previous step-wise diffusion, I modify the dataloader to include extra negative samples, and add the sample-wise auxiliary contrastive loss in diffusion.

### CUB-200 Dataset
I have finished the experiments on this dataset.

### COCO Dataset
1. First, we need to prepare some extra instance-level negative samples for each image from the training set. The extra negative samples are stored in the coco_negative_samples.json file in the ./image_synthesis/data folder.

2. Modify the config file coco_s.yaml in the ./configs folder. I have set the coco_s.yaml to the default setting for sample-wise contrastive diffusions, but it would probably to test different batch size in order to make full use of the memory in different machines.

3. Change the running script, I suggest to add the ''--output'' parameter to a path with enough space, since the intermediate checkpoint takes a lot of space

3. Launch the experiments by executing 
```
CUDA_VISIBLE_DEVICES=#IDS python running_command/run_train_coco.py
```
In case of running on the cluster, a bash script may be further needed to submit the job.


### ImageNet Dataset
1. ImageNet experiments use the class label embedding as the conditioning information. Similarly, 