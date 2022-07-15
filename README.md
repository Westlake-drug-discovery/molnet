# MolNet

<!-- <div align="center">
  <img src="resources/mmdet-logo.png" width="600"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab website</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab platform</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div> -->

<!-- [![PyPI](https://img.shields.io/pypi/v/mmdet)](https://pypi.org/project/mmdet)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmdetection.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmdetection/workflows/build/badge.svg)](https://github.com/open-mmlab/mmdetection/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmdetection/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmdetection)
[![license](https://img.shields.io/github/license/open-mmlab/mmdetection.svg)](https://github.com/open-mmlab/mmdetection/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmdetection.svg)](https://github.com/open-mmlab/mmdetection/issues)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmdetection.svg)](https://github.com/open-mmlab/mmdetection/issues) -->

<!-- [📘Documentation](https://mmdetection.readthedocs.io/en/stable/) |
[🛠️Installation](https://mmdetection.readthedocs.io/en/stable/get_started.html) |
[👀Model Zoo](https://mmdetection.readthedocs.io/en/stable/model_zoo.html) |
[🆕Update News](https://mmdetection.readthedocs.io/en/stable/changelog.html) |
[🚀Ongoing Projects](https://github.com/open-mmlab/mmdetection/projects) |
[🤔Reporting Issues](https://github.com/open-mmlab/mmdetection/issues/new/choose) -->

<!-- </div>

<div align="center">

English | [简体中文](README_zh-CN.md)

</div> -->

## Introduction

MolNet is an open source molecule property prediction toolbox based on PyTorch. 


<details open>
<summary>Major features</summary>

- **Backbone Design**

  We compare different network backbones in the same framework to provide a comprehensive understanding for molecular property prediction.

- **Reproduciable results**

  We reproduce papers under a consistent protocol and provide pre-trained parameters to ensure the reproducibility.

</details>

## Tutorial
```python
    cd /gaozhangyang/experiments/molnet/molnet
    python train.py
```

## Overview of Benchmark and Model Zoo

<!-- Results and models are available in the [model zoo](docs/en/model_zoo.md). -->

<div align="center">
  <b>Components</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>GNNs</b>
      </td>
      <td>
        <b>Transformers</b>
      </td>
      <!-- <td>
        <b>Loss</b>
      </td>
      <td>
        <b>Common</b>
      </td> -->
    </tr>
    <tr valign="top">
      <td>
      <ul>
        <li> <a href="https://arxiv.org/pdf/1609.02907.pdf"> GCN(ICLR'2017)</a></li>
        <li> <a href="https://arxiv.org/pdf/1810.00826.pdf"> GIN(ICLR'2019)</a></li>
        <li> <a href="https://arxiv.org/pdf/2102.05013.pdf"> MXMNet(NIPS'2020 workshop)</a></li>
        <li> <a href="https://arxiv.org/pdf/2106.08903.pdf"> GemNet(NIPS'2021) </a></li>
        <li> <a href="https://arxiv.org/pdf/2102.05013.pdf"> SphereNet(ICLR'2022) </a></li>
        <li> <a href="https://arxiv.org/pdf/2003.03123.pdf"> DimeNet(ICLR'2020) </a></li>
      </ul>
      </td>
      <td>
      <ul>
        <li><a href="https://arxiv.org/pdf/2002.08264.pdf">MAT (arxiv'2020)</a></li>
        <li><a href="https://arxiv.org/pdf/2110.01191.pdf">Molformer (arxiv'2021)</a></li>
        <li><a href="https://arxiv.org/pdf/2106.05234.pdf">Graphormer (NIPS'2021)</a></li>
      </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

<!-- Some other methods are also supported in [projects using MMDetection](./docs/en/projects.md). -->

<!-- ## FAQ

Please refer to [FAQ](docs/en/faq.md) for frequently asked questions. -->

<!-- ## Contributing

We appreciate all contributions to improve MMDetection. Ongoing projects can be found in out [GitHub Projects](https://github.com/open-mmlab/mmdetection/projects). Welcome community users to participate in these projects. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMDetection is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new detectors. -->

<!-- ## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
``` -->

<!-- ## License

This project is released under the [Apache 2.0 license](LICENSE).

## Projects in OpenMMLab

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab rotated object detection toolbox and benchmark.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab text detection, recognition, and understanding toolbox.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D human parametric model toolbox and benchmark.
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab self-supervised learning toolbox and benchmark.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab model compression toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab fewshot learning toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab image and video generative models toolbox.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab model deployment framework. -->