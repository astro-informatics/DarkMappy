|GitHub| |Build Status| |Docs| |CodeCov| |GPL license| |ArXiv 1| |ArXiv 2|

.. |GitHub| image:: https://img.shields.io/badge/GitHub-DarkMappy-brightgreen.svg?style=flat
    :target: https://github.com/astro-informatics/DarkMappy
.. |Build Status| image:: https://github.com/astro-informatics/DarkMappy/actions/workflows/python.yml/badge.svg
    :target: https://github.com/astro-informatics/DarkMappy/actions/workflows/python.yml
.. |Docs| image:: https://readthedocs.org/projects/ansicolortags/badge/?version=latest
    :target: https://astro-informatics.github.io/DarkMappy
.. |CodeCov| image:: https://codecov.io/gh/astro-informatics/DarkMappy/branch/main/graph/badge.svg?token=A5ogGPslpU
    :target: https://codecov.io/gh/astro-informatics/DarkMappy
.. |GPL License| image:: https://img.shields.io/badge/License-GPL-blue.svg
    :target: http://perso.crans.org/besson/LICENSE.html
.. |ArXiv 1| image:: http://img.shields.io/badge/arXiv-2004.07855-orange.svg?style=flat
    :target: https://arxiv.org/abs/2004.07855
.. |ArXiv 2| image:: http://img.shields.io/badge/arXiv-1812.04014-orange.svg?style=flat
    :target: https://arxiv.org/abs/1812.04014

DarkMappy: hybrid Bayesian inference of the dark universe
=================================================================================================================

``darkmappy`` is a lightweight python package which implements the hybrid Bayesian dark-matter reconstruction techniques 
outlined on the plane in `Price 2019 <https://academic.oup.com/mnras/article-abstract/506/3/3678/6319513>`_, and on the celestial sphere in `Price 2021 <https://academic.oup.com/mnras/article/500/4/5436/5986632>`_ and `Wallis 2021 <https://academic.oup.com/mnras/article-abstract/509/3/4480/6424933>`_. These techniques are based on *maximum a posteriori* estimation, and by construction support principled uncertainty quantification, by leveraging recent advances in probability concentration theory (`Pereyra 2016 <https://epubs.siam.org/doi/10.1137/16M1071249>`_).


BASIC USAGE
============

Add some basic usage instructions here.
    
DOCUMENTATION
=============

Link to the full documentation (when deployed).

CONTRIBUTORS
============
Matthew A. Price, Jason D. McEwen & Contributors

ATTRIBUTION
===========
A BibTeX entry for ``darkmappy`` is:

.. code-block:: 

    @article{price:2021:spherical,
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

``darkmappy`` is released under the GPL-3 license (see `LICENSE.txt <https://github.com/astro-informatics/DarkMappy/blob/main/LICENSE.txt>`_), subject to 
the non-commercial use condition (see `LICENSE_EXT.txt <https://github.com/astro-informatics/DarkMappy/blob/main/LICENSE_EXT.txt>`_)

.. code-block::

     DarkMappy
     Copyright (C) 2022 Matthew A. Price, Jason D. McEwen & contributors

     This program is released under the GPL-3 license (see LICENSE.txt), 
     subject to a non-commercial use condition (see LICENSE_EXT.txt).

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

.. bibliography:: 
    :notcited:
    :list: bullet

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: User Guide

   user_guide/install


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Background

   background/index

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Interactive Tutorials
   
   tutorials/example_notebook.nblink

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API

   api/index



