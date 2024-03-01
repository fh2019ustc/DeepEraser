🚀 **Exciting update! We have created a demo for our paper, showcasing the adaptive removal capabilities of our method. [Check it out here!](https://deeperaser.doctrp.top:20443/)**

# DeepEraser

<p>
    <a href='https://arxiv.org/abs/2402.19108' target="_blank"><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
    <a href='https://deeperaser.doctrp.top:20443/' target="_blank"><img src='https://img.shields.io/badge/Online-Demo-green'></a>
</p>

The official code for “DeepEraser: Deep Iterative Context Mining for Generic Text Eraser”.

<img width="932" alt="image" src="https://github.com/fh2019ustc/DeepEraser/assets/50725551/76e9dddc-e115-4b09-8a48-3de050e64823">


## 🚀 Demo [(Link)](https://deeperaser.doctrp.top:20443/)
1. Put the distorted images in `$ROOT/input_imgs/` and rename it to `input.png`.
2. Put the mask image in `$ROOT/input_imgs/` and rename it to `mask.png`.
3. Run the script and the processed image is saved in `$ROOT/output_imgs/` by default.
    ```
    python demo.py
    ```

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@inproceedings{feng2024DeepEraser,
  title={DeepEraser: Deep Iterative Context Mining \par for Generic Text Eraser},
  author={Feng, Hao and Wang, Wendi and Liu, Shaokai and Deng, Jiajun and Zhou, Wengang and Li, Houqiang},
  booktitle={arXiv},
  year={2024}
}
```
