# CASC Predictor

Blizzard Filenames + Machine Learning...  
An example model, training data and arguments can be found under [Example](Example).

#### Prerequisites
- .Net Core 2.2
- [Microsoft CTNK](https://github.com/Microsoft/CNTK/releases) v2.6 GPU ([Setup Guide](https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-CNTK-on-your-machine))
  - The binary path needs to be added to Windows' `Path` environment variable
  - CTNK is optimised for Intel CPUs and CUDA GPUs expect AMD CPUs to be slow!

#### Arguments


| Argument | Description | Mode |
| ------- | :---- | :-------: |
| --mode | 0 = Sample, 1 = Train **(Required)** | - |
| --modelprefix | Model name prefix **(Required)** | - |
| --datapath | Path to the Training Data **(Required)** | - |
| --samplesize | Sample character length. Defaults to 100 | - |
| --epochs | Amount of times to train. Defaults to 50 | 0 |
| --samplefrequency | Amount of minibatches between each sample. Defaults to 1000 | 0 |
| --samplecount | Amount of samples to produce. Defaults to 100 | 1 |
| --sampleprime | Sample prefix. Randomly selected if not provided. Defaults to random | 1 |

