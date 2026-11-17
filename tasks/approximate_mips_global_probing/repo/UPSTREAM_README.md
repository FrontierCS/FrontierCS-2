# The Source Code for FARGO (PVLDB 2023)
-----------------------------------------------------------------------------------------------------------------
## Introduction
This is a source code for the algorithm described in the paper **[FARGO: Fast Maximum Inner Product Search via Global Multi-Probing (PVLDB 2023)]**. We call it as **fg** project.
## Compilation
The **fg** project is written by **C++** (under `C++17` standard) and is simple and easy to use. It can be complied by **g++** in **Linux** or **MSVC** in **Windows**. To completely support `C++17` standard, the **g++** version is suggested to be at least **version 8** and **MSVC** version is suggested to be at least **MSVC 19.15 (Visual Studio 2017 15.8)**.

### Installation
#### Windows
We can use **Visual Studio 2019** (Other version of Visual Studio should also work but remains untested) to build the project with importing all the files in the directory `./code/Fargo/src/`.

#### Linux
```bash
cd ./code/Fargo
make
```
The excutable file is then in `./code/Fargo` directory, called as `fg`

## Usage

### Command Usage

-------------------------------------------------------------------
> ./fg datasetName
-------------------------------------------------------------------
(the first parameter specifies the procedure be executed and change)

FOR EXAMPLE, YOU CAN RUN THE FOLLOWING CODE IN COMMAND LINE AFTER BUILD ALL THE TOOLS:

```bash
cd ./code/Fargo
make
./fg audio
```

## Dataset

In our project, the format of the input file (such as `audio.data_new`, which is in `float` data type) is the same as that in [LSHBOX](https://github.com/RSIA-LIESMARS-WHU/LSHBOX). It is a binary file but not a text file, because binary file has many advantages. The binary file is organized as the following format:

>{Bytes of the data type (int)} {The size of the vectors (int)} {The dimension of the vectors (int)} {All of the binary vector, arranged in turn (float)}


For your application, you should also transform your dataset into this binary format, then rename it as `[datasetName].data_new` and put it in the directory `./dataset`.

A sample dataset `audio.data_new` has been put in the directory `./dataset`.
Also, you can get it, `audio.data`, from [here](http://www.cs.princeton.edu/cass/audio.tar.gz)(if so, rename it as `audio.data_new`). If the link is invalid, you can also get it from [data](https://github.com/RSIA-LIESMARS-WHU/LSHBOX-sample-data).

For other dataset we use, you can get the raw data from following links: [MNIST](http://yann.lecun.com/exdb/mnist/index.html), [Cifar](http://www.cs.toronto.edu/~kriz/cifar.html), [Trevi](http://phototour.cs.washington.edu/patches/default.htm), [NUS](https://pan.baidu.com/share/init?surl=kVKfXFx)(Extraction code: hpxg), [Deep1M](https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/deep1M.tar.gz), [GIST](http://corpus-texmex.irisa.fr/), [TinyImages80M](https://hyper.ai/tracker/download?torrent=6552), [SIFT](http://corpus-texmex.irisa.fr/). Next, you should transform your raw dataset into the mentioned binary format, then rename it is `[datasetName].data_new` and put it in the directory `./dataset`.


## Result
The experimental result is saved in the directory `./code/Fargo/results/` as the file `Running_result.txt`.



## Reference
Please use the following bibtex to cite this work when you use **FARGO** in your paper.
```tex
@article{DBLP:journals/pvldb/ZhaoZYLXZJ23,
	author       = {Xi Zhao and
	Bolong Zheng and
	Xiaomeng Yi and
	Xiaofan Luan and
	Charles Xie and
	Xiaofang Zhou and
	Christian S. Jensen},
	title        = {{FARGO:} Fast Maximum Inner Product Search via Global Multi-Probing},
	journal      = {Proc. {VLDB} Endow.},
	volume       = {16},
	number       = {5},
	pages        = {1100--1112},
	year         = {2023}
}
```
If you meet any issue on the code or take interest in our work, please feel free to contact me (xi.zhao@connect.ust.hk). Thank you.
