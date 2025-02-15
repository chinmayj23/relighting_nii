# relighting_nii
Project pursued @ National institute of Informatics, Tokyo

Internship student: Chinmay Joshi (Saarland University)
Supervisor: Prof. Satoshi Ikehata (National Institute of Informatics)

Duration of internship: ~5 months (3rd April, 2024 to 27th August, 2024)

Introduction

The world of Computer Vision was recently blessed by the introduction of 3D representations. During my internship at National Institute of Informatics, I explored this world of 3D Computer Vision under the supervision of Prof. Satoshi Ikehata. Initially, we regularly surveyed research papers related to novel view synthesis and brainstormed ideas to improve them. Our focus was mostly on synthesizing 3D representations of objects from a single image. 

One of the extensions to the research on 3D generation or novel view synthesis is relighting. To achieve relightable 3D objects, we focused on extrapolating ideas from a research work called Neural Gaffer. To implement this, we firstly created a synthetic dataset by collecting good quality 3D objects from Objaverse and rendered 12 different views under 5 different lighting conditions randomly sampled from the Polyhaven dataset. Then, we used the script published by the authors of Marigold, a renowned research work, to finetune Stable diffusion for relighting. While it did not produce the desired results, fine-tuning stable diffusion itself was a great learning experience.  

All stages of my internship - surveying literature, dataset creation, and fine tuning stable diffusion provided me with a thorough understanding of the topic and created a knowledge base to conduct further research in the field of 3D Computer Vision. 

Related Work

For exploring novel view synthesis, the biggest challenge was multi-view consistency. The following table contains a summary of some of the crucial work for generating 3D models from a single image consistently. Some of these can also be considered as baselines to extend future work on to. 
 
https://docs.google.com/document/d/1BND9Jg54HI8nD480C2cMtlJAODLQon9ApyxXXDXG1ak/edit?usp=sharing (Table1. Summary of research work focussing on Single image to 3D generation)

While exploring this, we realized that relighting complemented with 3D reconstruction is a promising direction of research. Hence, we decided to extend on a research called the Neural Gaffer (https://github.com/Haian-Jin/Neural_Gaffer?tab=readme-ov-file ) that finetunes Zero1-to-3 (https://github.com/cvlab-columbia/zero123 ) for relighting. 

Dataset Generation

To finetune either Zero1-to-3 or Stable Diffusion for relighting, we firstly require a dataset. For that, we collected a selected set of objects from the Objaverse dataset (https://objaverse.allenai.org/) and rendered each object from 12 different views and 5 different environments. The environments were randomly chosen from about 600 environments collected from the Polyhaven dataset (https://polyhaven.com/hdris ). For every view, we also store the camera coordinates, and render the depth map and normal map. While this was not used in the current project, it extends the possibility to pursue research complementing 3D reconstruction like material estimation, maintaining consistent high frequency details, and photometric stereo.

We have provided the script to download objects and render a dataset using blender.  

Relighting 
Relighting an object is a task where an object or an image is captured in a particular light setting, and transformed to another light setting (either randomly or provided by the user) using AI. 

To develop this model, we fine tuned Stable Diffusion 2.1 (https://huggingface.co/stabilityai/stable-diffusion-2-1 ). For doing so, we modified the script published by the authors of Marigold (https://github.com/prs-eth/Marigold ), since it was comprehensive and tackled a problem very similar to ours.  We have provided the modified codebase for the same. 

Using the code

*Downloading the Objaverse objects*: This is simply done by running the download_objaverse.py script. To further filter the quality of downloaded objects, we can add conditions to the annotations. At the moment, the only two conditions are the animation count set to 0 and the object is categorized in at least one annotation category. 

We can also create our own uid list by filtering from all the uids that are loaded using ‘objaverse.load_uids()’. For this version, we use the LVIS annotation list provided by the objaverse team as it contains relatively higher quality object files.  

*Dataset Generation*: Rendering the objects is a very important part of this project, as generating a high quality dataset is crucial before starting to train a neural network. Ideally, the current dataset is subpar and requires improvement. Either way, we provide a thorough script to render a dataset and arguments can be varied to get a more desirable outcome. 


Firstly, make sure that Blender 3.6 LTS is installed in the system. Then we can run the script blender_render_script_Era3D.py, which is a modified version of the blender script published by the authors of Era3D.  The terminal command to run this script in the background with the necessary arguments is: 

blender -b -P D:/CJ/MyBlender/blender_render_script_era3D.py -- --output_dir D:/CJ/MyBlender/new_folder1 --engine CYCLES

Make sure to change the osroot and objroot in the main. osroot is where the project is located and objroot is where the objects are placed. Other aspects like resolution can be modified by modifying the code. Most of the variables are placed in the first few lines. (Not the most sophisticated code, but it works fine). A subset of the dataset is available at https://1drv.ms/u/s!Akayu_8VtzKCgm00ngsk9sJkYmLJ?e=CILchb. 

*Fine-Tuning  the Neural Network*:  The directory Marigold_relight is where the training code is placed. Most of it is derived from the code published by authors of Marigold. 

First, make sure that all paths inside the script are correctly modified to the paths of the current system (I again apologize for not having a code sophisticated enough for using a single path, it is chaotic). 

Secondly, make sure to download the pre-trained model weights and place it in the checkpoints directory. We used weights from stable diffusion 2.1 - https://huggingface.co/stabilityai/stable-diffusion-2. 

Then we must install the requirements in ‘req.txt’ and export data directory and checkpoint directory to the environment, with the following commands:
export BASE_DATA_DIR=home/remote-dir/Marigold_relight/data
export BASE_CKPT_DIR=home/remote-dir/Marigold_relight/checkpoints

Finally, you can setup the config with accelerate config and start the training with,
accelerate launch train.py --config config/train_marigold.yaml .

If required, make the necessary changes in the config files placed in ‘config’ directory. To run this without wandb, make sure to add the argument –no_wand to the accelerate launch command. The output is visualized in the output/visualization directory.

Conclusion

Even though we could not get the desired results in the short span of the project, we could develop a complete framework to approach the relighting research problem that could potentially achieve relighting with consistent novel view synthesis. Simultaneously, we could render a comprehensive dataset that will be useful for 3D Computer Vision tasks. As future work, this can be potentially complemented with Universal Photometric Stereo (https://github.com/satoshi-ikehata/SDM-UniPS-CVPR2023/tree/main ).

The overall experience has been filled with new learnings and satisfaction of working on an interesting research problem. The critical analysis of literature was a crucial part of it too. I am grateful to Prof. Satoshi Ikehata for his support and guidance throughout the duration. 
