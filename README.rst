.. image:: https://img.shields.io/badge/GitHub-DarkMappy-brightgreen.svg?style=flat
    :target: https://github.com/astro-informatics/DarkMappy
.. image:: https://github.com/astro-informatics/DarkMappy/actions/workflows/python.yml/badge.svg?branch=main
    :target: https://github.com/astro-informatics/DarkMappy/actions/workflows/python.yml
.. image:: https://readthedocs.org/projects/ansicolortags/badge/?version=latest
    :target: https://astro-informatics.github.io/DarkMappy
.. image:: https://codecov.io/gh/astro-informatics/DarkMappy/branch/main/graph/badge.svg?token=A5ogGPslpU
    :target: https://codecov.io/gh/astro-informatics/DarkMappy
.. image:: https://img.shields.io/badge/License-GPL-blue.svg
    :target: http://perso.crans.org/besson/LICENSE.html
.. image:: http://img.shields.io/badge/arXiv-2004.07855-orange.svg?style=flat
    :target: https://arxiv.org/abs/2004.07855
.. image:: http://img.shields.io/badge/arXiv-1812.04014-orange.svg?style=flat
    :target: https://arxiv.org/abs/1812.04014

|logo| DarkMappy: mapping the dark universe
=================================================================================================================

.. |logo| raw:: html

   <img src="./docs/assets/darkmappy_alt_no_text.png" align="center" height="100" width="100">

``darkmappy`` is a lightweight python package which implements the hybrid Bayesian dark-matter reconstruction techniques 
outlined on the plane in `Price et al. 2019 <https://academic.oup.com/mnras/article-abstract/506/3/3678/6319513>`_, and on the celestial sphere in `Price et al. 2021 <https://academic.oup.com/mnras/article/500/4/5436/5986632>`_. For comparison (and as initilaisiation for our iterations) the spherical Kaiser-Squires estimator of the convergence is implemented (see `Wallis et al. 2021 <https://academic.oup.com/mnras/article-abstract/509/3/4480/6424933>`_). These techniques are based on *maximum a posteriori* estimation which, by construction, support principled uncertainty quantification, see `Pereyra 2016 <https://epubs.siam.org/doi/10.1137/16M1071249>`_. Further examples of such uncertainty quantification techniques developed for the weak lensing setting can be found in related articles `Price et al. 2019a <https://academic.oup.com/mnras/article/489/3/3236/5554769>`_ and `Price et al. 2019b <https://academic.oup.com/mnras/article/492/1/394/5672642>`_.

INSTALLATION
============
``darkmappy`` can be installed through PyPi by running 

.. code-block:: bash

    pip install darkmappy 

or alternatively from source by running the following 

.. code-block:: bash

    git clone https://github.com/astro-informatics/DarkMappy.git
    cd DarkMappy 
    bash build_darkmappy.sh 

following which the test suite can be executed by running 

.. code-block:: bash

    pytest --black darkmappy/tests

BASIC USAGE
===========
For planar reconstructions across the flat-sky the estimator can be run by the following, note that images must be square.

.. code-block:: python

    import numpy as np
    import darkmappy.estimators as dm

    # LOAD YOUR DATA
    data = np.load(<path_to_shear_data>)
    ngal = np.load(<path_to_ngal_per_pixel_map>)
    mask = np.load(<path_to_observation_mask>)

    # BUILD THE ESTIMATOR 
    dm_estimator = dm.DarkMappyPlane(
               n = n,                   # Dimension of image
            data = data,                # Observed shear field
            mask = mask,                # Observational mask
            ngal = ngal,                # Map of number density of observations per pixel
             wav = [<select_wavelets>], # see https://tinyurl.com/mrxeat3t
          levels = level,               # Wavelet levels
     supersample = supersample)         # Super-resolution factor (typically <~2)

    # RUN THE ESTIMATOR
    convergence, diagnostics = dm_estimator.run_estimator()

For spherical reconstructions across the full-sky the estimator can be run by the following, note images must be of dimension L by 2L-1, see `McEwen & Wiaux 2011 <https://ieeexplore.ieee.org/document/6006544>`_.

.. code-block:: python

    import numpy as np
    import darkmappy.estimators as dm

    # LOAD YOUR DATA
    data = np.load(<path_to_shear_data>)
    ngal = np.load(<path_to_ngal_per_pixel_map>)
    mask = np.load(<path_to_observation_mask>)

    # BUILD THE ESTIMATOR
    dm_estimator = dm.DarkMapperSphere(
               L = L,             # Angular Bandlimit    
               N = N,             # Azimuthal Bandlimit (wavelet directionality)
            data = data,          # Observational shear data
            mask = mask,          # Observation mask
            ngal = ngal)          # Map of number density of observations per pixel
    
    # RUN THE ESTIMATOR 
    convergence, diagnostics = dm_estimator.run_estimator()

CONTRIBUTORS
============
`Matthew A. Price <https://cosmomatt.github.io>`_, `Jason D. McEwen <http://www.jasonmcewen.org>`_ & Contributors

ATTRIBUTION
===========
A BibTeX entry for ``darkmappy`` is:

.. code-block:: 

    @article{price:2021:spherical,
            title = {Sparse Bayesian mass-mapping with uncertainties: Full sky observations on the celestial sphere},
           author = {M.~A.~Price and J.~D.~McEwen and L.~Pratley and T.~D.~Kitching},
          journal = {Monthly Notices of the Royal Astronomical Society},
             year = 2021,
            month = jan,
           volume = {500},
           number = {4},
            pages = {5436-5452},
              doi = {10.1093/mnras/staa3563},
        publisher = {Oxford University Press}
    }



.. code-block:: 

    @article{price:2021:hypothesis,
            title = {Sparse Bayesian mass mapping with uncertainties: hypothesis testing of structure},
           author = {M.~A.~Price and J.~D.~McEwen and X.~Cai and T.~D.~Kitching and C.~G.~R.~Wallis and {LSST Dark Energy Science Collaboration}},
          journal = {Monthly Notices of the Royal Astronomical Society},
             year = 2021,
            month = jul,
           volume = {506},
           number = {3},
            pages = {3678--3690},
              doi = {10.1093/mnras/stab1983},
        publisher = {Oxford University Press}
    }

If, at any point, the direction inverse functionality (i.e. spherical Kaiser-Squires) please cite 

.. code-block::

    @article{wallis:2021:massmappy,
            title = {Mapping dark matter on the celestial sphere with weak gravitational lensing},
           author = {C.~G.~R.~Wallis and M.~A.~Price and J.~D.~McEwen and T.~D.~Kitching and B.~Leistedt and A.~Plouviez},
          journal = {Monthly Notices of the Royal Astronomical Society},
             year = 2021,
            month = Nov,
           volume = {509},
           number = {3},
            pages = {4480-4497},
              doi = {10.1093/mnras/stab3235},
        publisher = {Oxford University Press}
    }

Finally, if uncertainty quantification techniques which rely on the approximate level-set threshold (derived by `Pereyra 2016 <https://epubs.siam.org/doi/10.1137/16M1071249>`_) are performed please consider citing relating articles appropriately.

LICENSE
=======

``darkmappy`` is released under the GPL-3 license (see `LICENSE.txt <https://github.com/astro-informatics/DarkMappy/blob/main/LICENSE.txt>`_).

.. code-block::

     DarkMappy
     Copyright (C) 2022 Matthew A. Price, Jason D. McEwen & contributors

     This program is released under the GPL-3 license (see LICENSE.txt).

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
