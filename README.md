# CDCD
 This is the implementation of the **Conditional Discrete Contrastive Diffusion** approach for cross-modal and conditional generation.

 [Paper]() | [Project Page]()


 ## 1. Project Overview
 In this work, we introduce our Conditional Discrete Contrastive Diffusion (CDCD) approach to enhance the input-output connections in cross-modal and conditional generation. 
 Specifically, we tackle the problem by maximizing the mutual information between the given input and the generated output via contrastive learning.
 We demonstrate the efficacy of the proposed approach in evaluations with three diverse, multimodal conditional synthesis tasks: dance-to-music generation, text-to-image synthesis, and class-conditioned image synthesis.

 <p align="center">
    <img src="assets/teaser.png" width="700">


<!-- ## Sample-wise contrastive diffusion + inter negative samples

### To run the sample-wise contrastive diffusion for the image generation task, the env setup is the same as previous step-wise constrastive diffusion setting.
Compared to the previous step-wise diffusion, I modify the dataloader to include extra negative samples, and add the sample-wise auxiliary contrastive loss in diffusion.

### CUB-200 Dataset
I have finished the experiments on this dataset.

### COCO Dataset
1. First, we need to prepare some extra instance-level negative samples for each image from the training set. The extra negative samples are stored in the coco_negative_samples.json file in the ./image_synthesis/data folder. (This is already done)

2. Modify the config file coco_s.yaml in the ./configs folder. I have set the coco_s.yaml to the default setting for sample-wise contrastive diffusions, but it would probably to test different batch size in order to make full use of the memory in different machines. (Could try to test different batch size.)

3. Change the running script, I suggest to add the ''--output'' parameter to a path with enough space, since the intermediate checkpoint takes a lot of space (Modify the output path to your desired localtion)

4. Launch the experiments by executing 
```
CUDA_VISIBLE_DEVICES=#IDS python running_command/run_train_coco.py
```
In case of running on the cluster, a bash script may be further needed to submit the job.

5. For inference on the COCO dataset, use the inferece_coco.py script. Please specify the config, checkpoint, data annotations, and the output path in Line 153, 154, 155 and 168 before executing the command line below.
```
CUDA_VISIBLE_DEVICES=#ID python inference_coco.py
```


### ImageNet Dataset

For the experiments on the ImageNet, the pretrained vqgan model is different, need to download this model [vqgan_imagenet_f16_16384.pth](https://facevcstandard.blob.core.windows.net/t-shuygu/release_model/VQ-Diffusion/pretrained_model/taming_dvae/vqgan_imagenet_f16_16384.pth?sv=2019-12-12&st=2021-12-21T04%3A28%3A27Z&se=2028-12-22T04%3A28%3A00Z&sr=b&sp=r&sig=W5LJJ23tIdkLS7QIJJcxdTdo164i3X%2BCgdDZaAREapE%3D). This path need to be specified in the config/imagenet.yaml line 13.


1. ImageNet experiments use the class label embedding as the conditioning information. The extra negative samples are directly sampled in the dataloader, no extra file is needed. This option can be turned off by modifying the inter_negative_samples param in line 114 and 126 in the config/imagenet.yaml

2. Also try to modify the batch size in the imagenet.yaml to make better use of the memory/

3. Change the running script, I suggest to add the ''--output'' parameter to a path with enough space, since the intermediate checkpoint takes a lot of space (Modify the output path to your desired localtion)

4. Launch the experiments by executing 
```
CUDA_VISIBLE_DEVICES=#IDS python running_command/run_train_imagenet.py
```
In case of running on the cluster, a bash script may be further needed to submit the job.

5. For class-conditioned synthesis inference on ImageNet, use the inference_imgnet.py script. Specify the config, checkpoint and output path in Line 133 and 134, then use the following command.
```
CUDA_VISIBLE_DEVICES=#ID python inference_imgnet.py
```


## Step-wise contrastive diffusion + intra negative samples

To run the step-wise contrastive diffusion w/ intra negative samples, just need to modify the --contrastive_intra_loss_weight param to be a weight that not non-zero (5.0e-6 suggested, max 5.0e-5), and change the --contrastive_inter_loss_weight to be 0. 
 -->

