# Mixed-CNN-and-Transformer-for-Radio-Frequency-Fingerprint-Recognition

## <div align="center">Quick Start Examples</div>

<details open>
<summary>Install</summary>


Clone repo and
install [requirements.txt](https://github.com/DuBianJun-007/Mixed-CNN-and-Transformer-for-Specific-Emitter-Identification/blob/main/requirements.txt)
in a
[**Python>=3.10.0**](https://www.python.org/) environment, including
[**PyTorch>=2.3**](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/DuBianJun-007/Mixed-CNN-and-Transformer-for-Specific-Emitter-Identification.git  # clone
cd Mixed-CNN-and-Transformer-for-Specific-Emitter-Identification
pip install -r requirements.txt  # install
```

</details>

<details open>
<summary>Training weight files</summary>

All training weights are publicly available in the folder [best_models](https://github.com/DuBianJun-007/Mixed-CNN-and-Transformer-for-Specific-Emitter-Identification/blob/main/best_models)
</details>

<details open>

<summary>Training</summary>

The training data should be placed in the _dataset_train_ folder, directory structure:

step1 - Download the public dataset from [Zenodo](https://zenodo.org/records/3876140) \
step2 - Use the script [Bluetooth_ADC2IQ.py]() to convert ADC data to IQ data  \
step3 - Use the script [data partitioning.py]() to partition the IQ data into separate files with fixed sequence lengths         \
step4 - Use the script [data reconstitution.py]() to convert a _.txt_ text file to a _.pt_ file in tensor format for quick readability         \
step5 - run [trainer.py]()         



</details>