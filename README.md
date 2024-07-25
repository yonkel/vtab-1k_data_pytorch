# Simple function for downloading datasets for VTAB-1k benchmark in pytorch

You can read more about the Visual Task Adaptation Benchmark (VTAB) here https://github.com/google-research/task_adaptation

# Good to know
- I did not use the split from the paper, but I've used fixed generator seed your results can be replicated
- no Validation split yet
- some of the datasets are downloaded from Hugging Face because they are not in torchvision (or does not work)
- in dsprites location I use *label_x_position* as a label because I did not find the original label 
- retinopathy is not original dataset because the full version is too big and not in torchvision or HF so I've used this https://huggingface.co/datasets/NawinCom/Eye_diabetic
- do not forget to use greyscale transformation for **Greyscale** datasets, for example add --> *transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x)*


# Motivation
I had project where I finetuned vit_b_16 with LoRa on VTAB-1k and I could not find simple implementation for this benchmark in PyTorch

Some of the datasets were a bit of pain (Caltech101), so maybe this will save you some time in search of the data

**There are probably some bugs**

If you have any questions, feel free to contact me.
