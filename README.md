# AuDrA Extended
Extension of the AuDrA model, originally built by a team of researchers to rate the creativity of human drawings, as published in this paper:
````
Patterson, J. D., Barbot, B., Lloyd-Cox, J., & Beaty, R. E. (2023). AuDrA: An automated drawing assessment platform for evaluating creativity. Behavior Research Methods. https://doi.org/10.3758/s13428-023-02258-3
````
We worked on their existing code base, available on [OSFHome](https://osf.io/kqn9v/). The orignal code is contained in the first commit of this repository and can be found on the `original_AuDrA` branch.

## Instructions

The conda environment used in this repository is made to be run locally, on CPU.

1- Make sure you have Python and conda installed (with [Anaconda](https://www.anaconda.com/download/success) for example)

2- Clone the repository
```shell
git clone https://github.com/atrudel/AuDrA_extended.git
```

3- Make sure you are at the root of the repository
```shell
cd AuDrA_extended
```

2- Run the setup script that will
- Create the conda environment
- Download the training data and store it in a `Drawings` folder
```shell
sh setup.sh
```