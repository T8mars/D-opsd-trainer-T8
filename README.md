<h1 align="center">D-OPSD<br><sub><sup>On-Policy Self-Distillation for Continuously Tuning Step-Distilled Diffusion Models</sup></sub></h1>

<div align="center">

[![Project Page](https://img.shields.io/badge/Official%20Site-333399.svg?logo=homepage)](https://vvvvvjdy.github.io/d-opsd/)&#160;
  <a href="https://github.com/vvvvvjdy/D-OPSD"><img src="https://img.shields.io/badge/GitHub(DOPSD)-9E95B7?logo=github"></a> &nbsp;
    <a href="https://github.com/Tongyi-MAI/Z-Image"><img src="https://img.shields.io/badge/GitHub(ZImage)-9E95B7?logo=github"></a> &nbsp;
<a href="https://arxiv.org/abs/1" target="_blank"><img src="https://img.shields.io/badge/Paper-b5212f.svg?logo=arxiv" height="21px"></a>

</div>

<p align="center">
  <img src="assets/method_01.jpg" alt="Result" style="width:100%;">
</p>



### 🕒 Note
Our training  code is currently undergoing internal check of the company. Once it passes, we will open source it. 

Our work can also be reproduced based on our [paper](https://arxiv.org/abs/1).




### 🎀 Highlight

D-OPSD is an on-policy self-distillation training framework for diffusion models especially timestep-distilled ones. It features in:

- D-OPSD identify an emergent property of modern text to image diffusion models with
LLM/VLM encoders and utilize this property to the continuous tuning of step-distilled
diffusion model.
- D-OPSD is a novel diffusion models on-policy self-distillation framework. By
assigning the same model two roles with different contexts, D-OPSD enables supervised
tuning on the student’s own roll-outs without requiring any external reward function or
extra modules.
- D-OPSD is validated in different settings. The results show that our method enables
the model to learn new concepts, styles, and domain preferences while preserving its
original few-step inference capability and previous knowledge.



---


<p align="center">
   <figcaption style="text-align: center; margin-top: 10px; font-size: 0.92em;">
           In full fine-tuning, D-OPSD adapts the model toward the target domain (anime) while retaining original-domain knowledge and few-step inference capability.
        </figcaption>
  <img src="assets/full_compare_01.jpg" alt="Result" style="width:96%;">
</p>

---


<p align="center">
   <figcaption style="text-align: center; margin-top: 10px; font-size: 0.92em;">
           In small customized LoRA training, D-OPSD learns new concepts from only a few image-text pairs while maintaining few-step generation quality and generalizing to unseen prompts.
        </figcaption>
  <img src="assets/lora_compare_01.jpg" alt="Result" style="width:90%;">
</p>

---


### 🌺 Citation
If you find D-OPSD useful, please kindly cite our paper:
```bibtex
@article{jiang2026dopsd,
      title={D-OPSD: On-Policy Self-Distillation for Continuously Tuning Step-Distilled Diffusion Models},
      author={Jiang, Dengyang and Jin, Xin and Liu, Dongyang and Wang, Zanyi and Zheng, Mingzhe and Du, Ruoyi and Yang, Xiangpeng and Wu, Qilong and Li, Zhen and Gao, Peng and Yang, Harry and Hoi, Steven},
      journal={arXiv preprint arXiv:},
      year={2026}
}
```
