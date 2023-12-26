# HD-Painter

This repository is the official implementation of [HD-Painter](https://arxiv.org/abs/2312.14091).


**[HD-Painter: High-Resolution and Prompt-Faithful Text-Guided Image Inpainting with Diffusion Models](https://arxiv.org/abs/2312.14091)**
</br>
Hayk Manukyan,
Andranik Sargsyan,
Barsegh Atanyan,
[Zhangyang Wang](https://www.ece.utexas.edu/people/faculty/atlas-wang),
Shant Navasardyan,
[Humphrey Shi](https://www.humphreyshi.com)
</br>

[Paper](https://arxiv.org/abs/2312.14091) | [Video](https://www.dropbox.com/scl/fi/t4lnssa9wbkd3bqo9kzgs/HDPainterTeaserVideoV4.mov?rlkey=vuk2zwm4z4pngt73cdmpmumpv&dl=0)


<p align="center">
<img src="__assets__/github/teaser.jpg" width="800px"/>  
<br>
<em>
We propose the <strong>Prompt-Aware Introverted Attention (PAIntA)</strong> layer enhancing self-attention scores by prompt information and resulting in better text alignment generations. To further improve the prompt coherence we introduce the <strong>Reweighting Attention Score Guidance (RASG)</strong> mechanism seamlessly integrating a post-hoc sampling strategy into general form of DDIM to prevent out-of-distribution latent shifts.
Moreover, our approach allows extension to larger scales by introducing a specialized super-resolution technique customized for inpainting, enabling the completion of missing regions in images of up to 2K resolution. 
</em>
</p>

## Code
Will be released soon!


## Method

<img src="__assets__/github/method_arch.png" raw=true>

---  

## Results

<table class="center">

<tr>
  <td width=5% align="center" style="font-size: 120%">"yellow headphones"</td>
  <td align="center"><img src="__assets__/github/results/masked/1.jpg" raw=true></td>
  <td align="center"><img src="__assets__/github/results/results/1.jpg"></td>
</tr>
<tr>
  <td width=5% align="center" style="font-size: 120%">"bench"</td>
  <td align="center"><img src="__assets__/github/results/masked/5.jpg" raw=true></td>
  <td align="center"><img src="__assets__/github/results/results/5.jpg"></td>
</tr>
<tr>
  <td width=5% align="center" style="font-size: 120%">"lake"</td>
  <td align="center"><img src="__assets__/github/results/masked/4.jpg" raw=true></td>
  <td align="center"><img src="__assets__/github/results/results/4.jpg"></td>
</tr>
<tr>
  <td width=5% align="center" style="font-size: 120%">"lion"</td>
  <td align="center"><img src="__assets__/github/results/masked/2.jpg" raw=true></td>
  <td align="center"><img src="__assets__/github/results/results/2.jpg"></td>
</tr>
<tr>
  <td width=5% align="center" style="font-size: 120%">"leather couch"</td>
  <td align="center"><img src="__assets__/github/results/masked/3.jpg" raw=true></td>
  <td align="center"><img src="__assets__/github/results/results/3.jpg"></td>
</tr>
</table>


## BibTeX
If you use our work in your research, please cite our publication:
```
@article{manukyan2023hd,
  title={HD-Painter: High-Resolution and Prompt-Faithful Text-Guided Image Inpainting with Diffusion Models},
  author={Manukyan, Hayk and Sargsyan, Andranik and Atanyan, Barsegh and Wang, Zhangyang and Navasardyan, Shant and Shi, Humphrey},
  journal={arXiv preprint arXiv:2312.14091},
  year={2023}
}
```