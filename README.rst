=======
Shatter
=======

Repository for privacy-preserving decentralized learning using Shatter.

Decentralized Learning (DL) enables collaborative learning without a server and without training data leaving the users' devices. However, the models shared in DL can still be used to infer training data. Conventional privacy defenses such as differential privacy and secure aggregation fall short in effectively safeguarding user privacy in DL, either sacrificing model utility or efficiency. We introduce SHATTER, a novel DL approach in which nodes create Virtual Nodes to disseminate chunks of their full model on their behalf. This enhances privacy by (i) preventing attackers from collecting full models from other nodes, and (ii) hiding the identity of the original node that produced a given model chunk. We theoretically prove the convergence of SHATTER and provide a formal analysis demonstrating how SHATTER reduces the efficacy of attacks compared to when exchanging full models between participating nodes. We evaluate the convergence and attack resilience of SHATTER with existing DL algorithms, with heterogeneous datasets, and against three standard privacy attacks, including gradient inversion. Our evaluation shows that SHATTER not only renders these privacy attacks infeasible when each node operates 16 VNs but also exhibits a positive impact on model utility compared to standard DL. In summary, SHATTER enhances the privacy of DL while maintaining model utility and efficiency.

Installation
============

Before cloning, ensure that you have Git LFS installed. If not, you can install it using the following command:

.. code-block:: bash

    sudo apt-get update && sudo apt-get install git-lfs

To install the necessary dependencies, use the following command:

.. code-block:: bash

    pip install -r requirements_base.txt && pip install -r requirements_all.txt

Usage
=====

To start using Shatter, follow `ARTIFACT-EVALUATION.md`.

Contributing
============

We welcome contributions from the community! Please see the `CONTRIBUTING.rst` file for guidelines on how to contribute to this project.

Credits
=======

This project is built on the following amazing tools:

- `DecentralizePy <https://github.com/sacs-epfl/decentralizepy>`_ - A decentralized learning framework.
- `ROG <https://github.com/KAI-YUE/rog>`_ - A framework for privacy-preserving distributed optimization.

License
=======

This project is licensed under the MIT License. See the `LICENSE` file for more details.
