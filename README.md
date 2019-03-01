# Residual Invertible Spatio-Temporal Network (RISTN) For Video Super-Resolution.

This is AAAI 2019 poster paper, we provide code and model architecture for testing. But the code is not polished yet.

Offical paper link:  Please wait.

## abstract

Video super-resolution is a challenging task, which has at- tracted great attention in research and industry communi- ties. In this paper, we propose a novel end-to-end archi- tecture, called Residual Invertible Spatio-Temporal Network (RISTN) for video super-resolution. The RISTN can suffi- ciently exploit the spatial information from low-resolution to high-resolution, and effectively models the temporal con- sistency from consecutive video frames. Compared with ex- isting recurrent convolutional network based approaches, RISTN is much deeper but more efficient. It consists of three major components: In the spatial component, a lightweight residual invertible block is designed to reduce information loss during feature transformation and provide robust feature representations. In the temporal component, a novel recurrent convolutional model with residual dense connections is pro- posed to construct deeper network and avoid feature degrada- tion. In the reconstruction component, a new fusion method based on the sparse strategy is proposed to integrate the spa- tial and temporal features. Experiments on public benchmark datasets demonstrate that RISTN outperforms the state-of- the-art methods.

## The architecture of our network:

![Image text](https://github.com/lizhuangzi/RISTN/raw/master/screenshots/RISTN.png)

## The architecture of our (Residual dense convolutional LSTM) RDC-LSTM:

![Image text](https://github.com/lizhuangzi/RISTN/raw/master/screenshots/RDCLSTM.png)


## Dependence:

python 2.7

scikit-image 0.12.0

pytorch 0.4.0

torchvision 0.2.0

model file: https://pan.baidu.com/s/1DNvFwdjmpfzm-ZrCqID9Sw   extrat code: aky3

The directory "Vid4Result" is our output results for Vid4 dataset. you also can run: python Testout2.py for testing.

## Cite our paper:

You can cite as:
Xiaobin Zhu, Zhuangzi Li, Xiao-Yu Zhang, Changsheng Li, Yaqi Liu, Ziyu Xue. Residual Invertible Network for Video Super-Resolution. [C]//AAAI. 2019.

or cite by bib:

@inproceedings{AAAI19-RISTN,

  author    = {Xiaobin Zhu and
               Zhuangzi Li and
                Xiao-Yu Zhang and
                 Changsheng Li and
                 Yaqi Liu and
                 Ziyu Xue
                },

  title     = {Residual Invertible Network for Video Super-Resolution},

  booktitle = {{AAAI} Conference on Artificial Intelligence},

  year      = {2019},

}
