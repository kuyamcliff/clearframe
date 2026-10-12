# Model licenses

ClipFX Enhance's own code is MIT licensed. The AI models it can download are separate works with their own licenses.
Nothing below is bundled with ClipFX Enhance: each model is downloaded from its original publisher only when you press **Install**, and its license is shown in the Model Library before you do.

**Your responsibility:** some licenses restrict commercial use of the weights or of results. ClipFX Enhance shows what each publisher states; it is not legal advice. Check the original license before using output commercially.

This file is generated from `model-catalog/catalog.json` (a test checks that it stays in sync).

| Model | Publisher | Weights license | Code license | Commercial use | Redistribution |
|---|---|---|---|---|---|
| Real-ESRGAN AnimeVideo v3 (`xinntao.realesr-animevideo-v3`) | Xintao Wang et al. | BSD-3-Clause | BSD-3-Clause | allowed | allowed |
| Real-ESRGAN General v3 (`xinntao.realesr-general-v3`) | Xintao Wang et al. | BSD-3-Clause | BSD-3-Clause | allowed | allowed |
| Real-ESRGAN x4plus (`xinntao.realesrgan-x4plus`) | Xintao Wang et al. | BSD-3-Clause | BSD-3-Clause | allowed | allowed |
| Real-ESRGAN x4plus Anime 6B (`xinntao.realesrgan-x4plus-anime-6b`) | Xintao Wang et al. | BSD-3-Clause | BSD-3-Clause | allowed | allowed |
| Real-ESRGAN x2plus (`xinntao.realesrgan-x2plus`) | Xintao Wang et al. | BSD-3-Clause | BSD-3-Clause | allowed | allowed |
| FFDNet (color) (`kair.ffdnet-color`) | Kai Zhang et al. (KAIR) | MIT | MIT | allowed | allowed |
| DRUNet (color) (`kair.drunet-color`) | Kai Zhang et al. (KAIR) | MIT | MIT | allowed | allowed |
| NAFNet GoPro (motion deblur) (`megvii.nafnet-gopro-w32`) | Megvii Research | MIT | MIT | allowed | allowed |
| RIFE v4.26 (`hzwer.rife-v4.26`) | hzwer (Zhewei Huang) et al. | MIT | MIT | allowed | allowed |
| GFPGAN v1.4 + RetinaFace (`tencentarc.gfpgan-v1.4`) | Tencent ARC | Apache-2.0 (GFPGAN); MIT (RetinaFace via facexlib) | Apache-2.0 | allowed | allowed |

## Real-ESRGAN AnimeVideo v3

- ID: `xinntao.realesr-animevideo-v3` (version 1.0.0)
- Source: https://github.com/xinntao/Real-ESRGAN
- Weights license: BSD-3-Clause; code license: BSD-3-Clause
- Training data: Anime video frames prepared by the Real-ESRGAN authors (see the AnimeVideo-v3 report).
- Notes: Weights are published by the authors as release assets of the BSD-3-Clause Real-ESRGAN repository. Keep the copyright notice when redistributing.
- File `realesr-animevideov3.pth` (weights): https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth — sha256 `b8a8376811077954d82ca3fcf476f1ac3da3e8a68a4f4d71363008000a18b75d`

> Real-ESRGAN, Xintao Wang, Liangbin Xie, Chao Dong, Ying Shan (2021). Copyright (c) 2021, Xintao Wang.

## Real-ESRGAN General v3

- ID: `xinntao.realesr-general-v3` (version 1.0.0)
- Source: https://github.com/xinntao/Real-ESRGAN
- Weights license: BSD-3-Clause; code license: BSD-3-Clause
- Training data: DIV2K, Flickr2K and OutdoorSceneTraining with synthetic degradations (Real-ESRGAN paper).
- Notes: Weights are published by the authors as release assets of the BSD-3-Clause Real-ESRGAN repository. Keep the copyright notice when redistributing.
- File `realesr-general-x4v3.pth` (weights): https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth — sha256 `8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292`
- File `realesr-general-wdn-x4v3.pth` (wdn): https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth — sha256 `1641f8c4464b9f097c9fdda5589273713f67cf59f3d909e0bd688f0cee269dca`

> Real-ESRGAN, Xintao Wang, Liangbin Xie, Chao Dong, Ying Shan (2021). Copyright (c) 2021, Xintao Wang.

## Real-ESRGAN x4plus

- ID: `xinntao.realesrgan-x4plus` (version 1.0.0)
- Source: https://github.com/xinntao/Real-ESRGAN
- Weights license: BSD-3-Clause; code license: BSD-3-Clause
- Training data: DIV2K, Flickr2K and OutdoorSceneTraining with synthetic degradations (Real-ESRGAN paper).
- Notes: Weights are published by the authors as release assets of the BSD-3-Clause Real-ESRGAN repository. Keep the copyright notice when redistributing.
- File `RealESRGAN_x4plus.pth` (weights): https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth — sha256 `4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1`

> Real-ESRGAN, Xintao Wang, Liangbin Xie, Chao Dong, Ying Shan (2021). Copyright (c) 2021, Xintao Wang.

## Real-ESRGAN x4plus Anime 6B

- ID: `xinntao.realesrgan-x4plus-anime-6b` (version 1.0.0)
- Source: https://github.com/xinntao/Real-ESRGAN
- Weights license: BSD-3-Clause; code license: BSD-3-Clause
- Training data: Anime illustrations prepared by the Real-ESRGAN authors.
- Notes: Weights are published by the authors as release assets of the BSD-3-Clause Real-ESRGAN repository. Keep the copyright notice when redistributing.
- File `RealESRGAN_x4plus_anime_6B.pth` (weights): https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth — sha256 `f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da`

> Real-ESRGAN, Xintao Wang, Liangbin Xie, Chao Dong, Ying Shan (2021). Copyright (c) 2021, Xintao Wang.

## Real-ESRGAN x2plus

- ID: `xinntao.realesrgan-x2plus` (version 1.0.0)
- Source: https://github.com/xinntao/Real-ESRGAN
- Weights license: BSD-3-Clause; code license: BSD-3-Clause
- Training data: DIV2K, Flickr2K and OutdoorSceneTraining with synthetic degradations (Real-ESRGAN paper).
- Notes: Weights are published by the authors as release assets of the BSD-3-Clause Real-ESRGAN repository. Keep the copyright notice when redistributing.
- File `RealESRGAN_x2plus.pth` (weights): https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth — sha256 `49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb`

> Real-ESRGAN, Xintao Wang, Liangbin Xie, Chao Dong, Ying Shan (2021). Copyright (c) 2021, Xintao Wang.

## FFDNet (color)

- ID: `kair.ffdnet-color` (version 1.0.0)
- Source: https://github.com/cszn/KAIR
- Weights license: MIT; code license: MIT
- Training data: Waterloo Exploration Database, BSD and ImageNet subsets (FFDNet paper).
- Notes: Released as a release asset of the MIT-licensed KAIR repository.
- File `ffdnet_color.pth` (weights): https://github.com/cszn/KAIR/releases/download/v1.0/ffdnet_color.pth — sha256 `286f04f7e3a20254c067ba95efb5d3339f4fc8a09a365a93cd469184d4cdb279`

> FFDNet: Kai Zhang, Wangmeng Zuo, Lei Zhang (2018). KAIR, Copyright (c) 2019 Kai Zhang.

## DRUNet (color)

- ID: `kair.drunet-color` (version 1.0.0)
- Source: https://github.com/cszn/KAIR
- Weights license: MIT; code license: MIT
- Training data: BSD, Waterloo Exploration, DIV2K and Flickr2K (DPIR paper).
- Notes: Released as a release asset of the MIT-licensed KAIR repository.
- File `drunet_color.pth` (weights): https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth — sha256 `479abe3c5327dfd10ff54a80ec7d4098ca80752a5c9492cdff31cee430bec4b4`

> DRUNet (DPIR): Kai Zhang, Yawei Li, Wangmeng Zuo, Lei Zhang, Luc Van Gool, Radu Timofte (2021). KAIR, Copyright (c) 2019 Kai Zhang.

## NAFNet GoPro (motion deblur)

- ID: `megvii.nafnet-gopro-w32` (version 1.0.0)
- Source: https://github.com/megvii-research/NAFNet
- Weights license: MIT; code license: MIT
- Training data: GoPro deblurring dataset (Nah et al., 2017).
- Notes: Official weights are distributed by Megvii via Google Drive; this catalog uses a community mirror pinned to a commit and verified by SHA-256 (identical file).
- File `NAFNet-GoPro-width32.pth` (weights): https://huggingface.co/mikestealth/nafnet-models/resolve/cd3a91ddc41c9c3a53ee809b3ddd5263c9e94e51/NAFNet-GoPro-width32.pth — sha256 `19394e6155d12ef6371d1d57496f87f0ec88f92bdffa27c0792690722d5d1a5c`

> NAFNet: Liangyu Chen, Xiaojie Chu, Xiangyu Zhang, Jian Sun (2022). Copyright (c) 2022 megvii-model.

## RIFE v4.26

- ID: `hzwer.rife-v4.26` (version 1.0.0)
- Source: https://github.com/hzwer/Practical-RIFE
- Weights license: MIT; code license: MIT
- Training data: Vimeo90K and additional data described by the Practical-RIFE project.
- Notes: Weights published by the RIFE author on Hugging Face (hzwer/RIFE), pinned to a commit. Only the weight file is extracted; bundled Python files are never executed.
- File `RIFEv4.26_0921.zip` (weights): https://huggingface.co/hzwer/RIFE/resolve/01fdc7e97404120c243c3ea7b427046e5dc7643e/RIFEv4.26_0921.zip — sha256 `1fa9b9cda3d9b8c3e301359e2595960902f97bf926c08598b0e9957a3f3f760e`

> RIFE: Zhewei Huang, Tianyuan Zhang, Wen Heng, Boxin Shi, Shuchang Zhou (ECCV 2022). Copyright (c) 2021 hzwer.

## GFPGAN v1.4 + RetinaFace

- ID: `tencentarc.gfpgan-v1.4` (version 1.0.0)
- Source: https://github.com/TencentARC/GFPGAN
- Weights license: Apache-2.0 (GFPGAN); MIT (RetinaFace via facexlib); code license: Apache-2.0
- Training data: FFHQ (CC BY-NC-SA 4.0); RetinaFace on WIDER FACE.
- Notes: GFPGAN's repository license lists third-party components; ClipFX Enhance uses the 'clean' architecture that needs no custom CUDA operators. The model was trained on FFHQ, whose images are under CC BY-NC-SA 4.0 — review this if your use is commercial.
- File `GFPGANv1.4.pth` (restorer): https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth — sha256 `e2cd4703ab14f4d01fd1383a8a8b266f9a5833dacee8e6a79d3bf21a1b6be5ad`
- File `detection_Resnet50_Final.pth` (detector): https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth — sha256 `6d1de9c2944f2ccddca5f5e010ea5ae64a39845a86311af6fdf30841b0a5a16d`

> GFPGAN: Xintao Wang, Yu Li, Honglun Zhang, Ying Shan (CVPR 2021). Copyright (c) 2021 Tencent ARC.
> RetinaFace weights via facexlib (Xintao Wang) and Pytorch_Retinaface (biubug6), MIT.
