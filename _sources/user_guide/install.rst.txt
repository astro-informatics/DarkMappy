.. _install:

Installation
============
There are a variety of methods for installing darkmappy, the most straightforward of which is through the python package manager PyPi.

Quick install (PyPi)
--------------------
Install ``darkmappy`` from PyPi with a single command

.. code-block:: bash

    pip install darkmappy

Check that the package has installed by running 

.. code-block:: bash 

	pip list 

and locate darkmappy.


Install from source (GitHub)
----------------------------

When installing from source we recommend working within an existing conda environment, or creating a fresh conda environment to avoid any dependency conflicts,

.. code-block:: bash

    conda create -n darkmappy_env python=3.8
    conda activate darkmappy_env

Once within a fresh environment ``darkmappy`` may be installed by cloning the GitHub repository

.. code-block:: bash

    git clone https://github.com/astro-informatics/DarkMappy
    cd DarkMappy

and running the install script, within the root directory, with one command 

.. code-block:: bash

    bash build_darkmappy.sh

To check the install has worked correctly run the unit tests with 

.. code-block:: bash

	pytest --black darkmappy/tests/ 

.. note:: For installing from source a conda environment is required by the installation bash script, which is recommended, due to a pandoc dependency.
