<div align="center">

# Diffusion-Based Robust Image Editing using Residual Linear Interpolation

![main_figure](asset/first_figure.png)

[Hyunwook Jo](https://github.com/ugiugi0823/), &nbsp; [Jiseung Maeng](https://github.com/js43o/), &nbsp; [Junhyung Park](https://github.com/kevin20012), &nbsp; [Namhyuk Ahn](https://gcl-inha.github.io/), &nbsp; [In Kyu Park](https://sites.google.com/view/vcl-lab/)\* &nbsp; </br>
Department of Electrical and Computer Engineering, Inha University</br>
\*Corresponding Author

<p>
  <a href="https://arxiv.org/">Arxiv</a> |
  <a href="https://github.com/ugiugi0823/ICCVW-RLI">Code</a>
</p>

</div>

## ⚙️ Proposed Method

![main_figure](asset/framework.png)
This project is Demo for "Diffusion-Based Robust Image Editing using Residual Linear Interpolation" submitted to ACM MM

## 🌱 Environment Setting

**For Prompt-to-Prompt**

```bash
conda env create --file env/p2p.yaml
```

**For MasaCtrl**

```bash
conda env create --file env/masactrl.yaml
```

**For PnP**

```bash
conda env create --file env/pnp.yaml
```

## 🚀 Run

### 🔥 Run Bench described in paper

```bash
bash run_with_bench.sh
```

### 📝 How to write prompt

In **main.py**

```python
original_prompt = "A white horse running in the field"
editing_prompt = "Water color of a white horse running in the field"
image_path = "./img/horse.png"
editing_instruction = "" #You can write the instruction on it
blended_word = [] #Ex. ["horse", "dog"] if you want to change word "horse" in source prompt to word "dog" in target prompt
```

### 🎯 How to run to get result

You can obtain images by combining below:

| 🔒 Preserving source Image | 🎨 Editing image by target prompt | 🚀 Using Our method |
| :------------------------- | :-------------------------------- | :------------------ |
| null-text-inversion        | p2p                               | ALPHA               |
| negative-prompt-inversion  | masactrl                          |
| directinversion            | pnp                               |
| ddim                       |                                   |

(You can observe the results obtained without the image preservation technique by using "ddim".)

Since the Conda environment depends on the specific editing method, you must use appropriate environment for each editing method.

please make sure to execute commands using:
refer to **run.sh**

```bash
conda run -n p2p --no-capture-output python -u main.py --data_path img \
                --output_path output \
                --edit_method_list directinversion+p2p
```

simply using bash for all method

```bash
bash run_one_img.sh
```

And Finally, You can find the result in **output** directory.

## Acknowledgement

This code has been modified based on the [PnP_Inversion](https://github.com/cure-lab/PnPInversion/tree/main).  
Following the implementation from [null-text inversion](https://github.com/google/prompt-to-prompt/#null-text-inversion-for-editing-real-images), [negative-prompt inversion](https://arxiv.org/abs/2305.16807), [Direct inversion](https://arxiv.org/abs/2310.01506), [prompt-to-prompt](https://github.com/google/prompt-to-prompt), [MasaCtrl](https://github.com/TencentARC/MasaCtrl), [pix2pix-zero](https://github.com/pix2pixzero/pix2pix-zero) , [Plug-and-Play](https://github.com/MichalGeyer/plug-and-play).  
Sincerely thank all contributors.
