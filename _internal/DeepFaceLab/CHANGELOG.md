# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),

and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 10/02/2022
### Added
-  New custom saving time; default is 25 minutes - DeepFake ENG ITA
### Updated
 - Added random blur to shadow augmentation
 - [Splittable random shadows](https://github.com/MachineEditor/DeepFaceLab/commit/9bf3eb5851c6d0fbd2cea201332a60b047bb9113)
 - Added a handful of checks to indicate that the dataset is in zip or pak form when the functions used require a bulk dataset
### Fixed
- [true face power only avaible when gan power > 0](https://github.com/MachineEditor/DeepFaceLab/commit/c833e2a2642e993409018e1f92c2565739056024)
 - [wrong def. cpu cap](https://github.com/MachineEditor/DeepFaceLab/commit/18afb868bf486e4dd4bf5eba5d41fb14a5925620)
- Other random fixes

## [1.1.0] - 29/12/2021
### Added
 -  'random_hsv_power' from Official fork - seranus
 - New 'force_full_preview' to force to do not separate dst, src and pred views in different frames 	 - randomfaker
 ### Updated
 - Refactored two pass splitting it into 3 mode: None, face, face + mask - randomfaker
 - Updated shadow augmentation splitting it in: None, src, dst, all - DeepFake ENG ITA
 - [Update requirements-colab.txt](https://github.com/MachineEditor/DeepFaceLab/commit/bfaf6255ba5c70d831151099c67b65d87a9f5466)
### Fixed
- [config-training-file supports now files](https://github.com/MachineEditor/DeepFaceLab/commit/424469845960b06652af81e77409c70a6aa73003)

## [1.0.0] - 10/12/2021
### Initialized
We created this fork from several other forks of DeepFaceLab.
Many features of this fork comes mainly from [JH's fork](https://github.com/faceshiftlabs/DeepFaceLab).
#### Features from JH's fork
- [Web UI for training preview](doc/features/webui/README.md)
- [Random color training option](doc/features/random-color/README.md)
- [Background Power training option](doc/features/background-power/README.md)
- [MS-SSIM loss training option](doc/features/ms-ssim)
- [GAN label smoothing and label noise options](doc/features/gan-options)
- MS-SSIM+L1 loss function, based on ["Loss Functions for Image Restoration with Neural Networks"](https://research.nvidia.com/publication/loss-functions-image-restoration-neural-networks)
- Autobackup options:
	- Session name
	- ISO Timestamps (instead of numbered)
	- Max number of backups to keep (use "0" for unlimited)
- New sample degradation options (only affects input, similar to random warp):
	- Random noise (gaussian/laplace/poisson)
	- Random blur (gaussian/motion)
	- Random jpeg compression
	- Random downsampling
- New "warped" preview(s): Shows the input samples with any/all distortions.
#### Features from other forks
- FaceSwap-Aug in the color transfer modes
- Custom face types
#### Features from MVE Development team
- External configuration files by [Cioscos](https://github.com/Cioscos) aka DeepFake ENG ITA
	- use --auto_gen_config CLI param to auto generate config. file or resume its configuration
	- use --config_training_file CLI param external configuration file override
- Tensorboard support by [JanFschr](https://github.com/JanFschr) aka randomfaker
- AMP training updates - DeepFake ENG ITA & randomfaker
- shadow augmentation (needs testing to see if it can generalise well) - randomfaker
- filename labels by [Ognjen](https://github.com/seranus) aka JesterX aka seranus
- zip faceset support - randomfaker
- exposed new configuration parameters (cpu, lr, preview samples)
- Added pre-sharpen into the merger. It helps the model to fit better to the target face. Idea taken from [DeepFaceLive](https://github.com/iperov/DeepFaceLive)
- Added two pass option into the merger. It processes the generated face twice. Idea taken from [DeepFaceLive](https://github.com/iperov/DeepFaceLive)

[1.2.0]: https://github.com/MachineEditor/DeepFaceLab/tree/a7e0cbb0295ae35e9098ab383bc6e0a8bdd0f944
[1.1.0]: https://github.com/MachineEditor/DeepFaceLab/tree/bfaf6255ba5c70d831151099c67b65d87a9f5466
[1.0.0]: https://github.com/MachineEditor/DeepFaceLab/tree/6c5a5934452e174779561885fccf3f1ed38be9ae
