🚀 **Exciting update! We have created a demo for our paper, showcasing the adaptive removal capabilities of our method. [Check it out here!](https://deeperaser.doctrp.top:20443/)**

# DeepEraser
The official code for “DeepEraser: Deep Iterative Context Mining \par for Generic Text Eraser”.

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
@inproceedings{feng2021doctr,
  title={DeepEraser: Deep Iterative Context Mining \par for Generic Text Eraser},
  author={Feng, Hao and Wang, Wendi and Liu, Shaokai and Deng, Jiajun and Zhou, Wengang and Li, Houqiang},
  booktitle={arXiv},
  year={2024}
}
```
