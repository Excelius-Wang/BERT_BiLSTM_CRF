

# 北京交通大学NLP课程命名实体识别（NER）实验

北京交通大学自然语言处理课程命名实体识别，基于 BERT + BiLSTM + CRF，实验所需的语料库为散文数据集[lancopku/Chinese-Literature-NER-RE-Dataset: A Discourse-Level Named Entity Recognition and Relation Extraction Dataset for Chinese Literature Text (github.com)](https://github.com/lancopku/Chinese-Literature-NER-RE-Dataset)。

本次实验基于 Pytorch 编码实现，实验结果为**根目录**的 **predict_result.json** 文件。

> 参考：
>
> [taishan1994/BERT-BILSTM-CRF: 使用BERT-BILSTM-CRF进行中文命名实体识别。 (github.com)](https://github.com/taishan1994/BERT-BILSTM-CRF)
>
> （！待完善）

<!-- PROJECT SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]


## 目录

- [代码运行环境](#代码运行环境)
  * [Python 版本](#Python版本：)
  * [所需相关库](#所需相关库：)
  * [安装步骤](#安装步骤)
- [json 文件转换为 BIOS txt文件](#json 文件转换为 BIOS txt文件[)
- [模型训练](#模型训练)
- [模型验证](#模型验证)
- [词向量二维展示](#词向量二维展示)
- [作者](#作者)

### 代码运行环境

#### Python版本：

​	Python 3.9.19

#### 所需相关库：

- scikit-learn==1.1.3
- scipy==1.10.1
- seqeval==1.2.2
- transformers==4.27.4
- pytorch-crf==0.7.2
- opencc==1.1.1
- pandas==2.2.3
- pytorch==2.5.1

​	已导出`requirement.txt`文件，可以使用`pip install -r requirements.txt`或`conda install --yes --file requirements.txt`命令一键安装相关的库；或安装相关库。

​	建议使用 anaconda 管理 Python 环境，创建本项目的环境安装好相应的库再运行以下步骤。

#### 安装步骤

1. 配置实验运行环境

### json 文件转换为 BIOS txt文件

​	使用 PyCharm 打开本项目，打开 `src/data/precess.py`，右键代码任意部分运行；

​	生成 dev.txt，train.txt，entities_count.txt 文件位于`项目根目录/data/pre_data`。

### 模型训练

​	打开`config/config.py`里面有一些模型参数配置以及本地路径配置，可以根据实际情况进行修改，本次实验使用的是 4090 云服务器。打开`src/models/train.py`，右键代码任意部分即可运行。	

​	训练保存在本地的 bin 文件保存在 output 文件夹中，具体文件为 `best_model.json`和`模型名_args.json`。

### 模型验证

​	运行 `src/model/predict.py`可以对 test.json 进行命名实体识别，生成 `predict_result.json`文件于 output 文件夹中。

### 作者

[Ther's World (excelius.xyz)](https://www.excelius.xyz/)

邮箱：excelius@qq.com

 *您也可以在贡献者名单中参看所有参与该项目的开发者。*

### 版权说明

该项目签署了MIT 授权许可，详情请参阅 [LICENSE.txt](https://github.com/Excelius-Wang/NLP_exp_1 /blob/master/LICENSE.txt)

<!-- links -->

[your-project-path]:Excelius-Wang/NLP_exp_1
[contributors-shield]: https://img.shields.io/github/contributors/Excelius-Wang/NLP_exp_1.svg?style=flat-square
[contributors-url]: https://github.com/Excelius-Wang/NLP_exp_1/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Excelius-Wang/NLP_exp_1.svg?style=flat-square
[forks-url]: https://github.com/Excelius-Wang/NLP_exp_1/network/members
[stars-shield]: https://img.shields.io/github/stars/Excelius-Wang/NLP_exp_1.svg?style=flat-square
[stars-url]: https://github.com/Excelius-Wang/NLP_exp_1/stargazers
[issues-shield]: https://img.shields.io/github/issues/Excelius-Wang/NLP_exp_1.svg?style=flat-square
[issues-url]: https://img.shields.io/github/issues/Excelius-Wang/NLP_exp_1.svg
[license-shield]: https://img.shields.io/badge/License-MIT-yellow.svg
[license-url]: https://github.com/Excelius-Wang/NLP_exp_1/blob/master/LICENSE.txt
