### Gradient Obfuscation Gives a False Sense of Security in Federated Learning
![](https://img.shields.io/badge/Python-3-blue) ![](https://img.shields.io/badge/Pytorch-1.9.0-blue)

Federated learning has been proposed as a privacy-preserving machine learning framework that enables multiple clients to collaborate without sharing raw data. However, client privacy protection is not guaranteed by design in this framework.  Prior work has shown that the gradient sharing strategies in federated learning can be vulnerable to server data reconstruction attacks. In practice, though, clients may not transmit raw gradients considering the high communication cost or due to privacy enhancement requirements. Empirical studies have demonstrated that gradient obfuscation, including the intentional gradient noise injection and the unintentional gradient compression, can provide more privacy protection against reconstruction attacks. In this work, we present a new data reconstruction attack framework targeting the image classification task in federated learning.

<br />

<img src="doc/recon.png" width=600>

#### Prerequisites

- install Python packages
    ```bash
    pip3 install -r requirements.txt
    ```

- Unzip Data
    ```bash
    unzip data.zip
    ```
