|GitHub| |Build Status| |Docs| |CodeCov| |GPL license| |ArXiv|

.. |GitHub| image:: https://img.shields.io/badge/GitHub-DarkMappy-brightgreen.svg?style=flat
    :target: https://github.com/astro-informatics/DarkMappy
.. |Build Status| image:: https://github.com/astro-informatics/DarkMappy/actions/workflows/python.yml/badge.svg
    :target: https://github.com/astro-informatics/DarkMappy/actions/workflows/python.yml
.. |Docs| image:: https://readthedocs.org/projects/ansicolortags/badge/?version=latest
    :target: https://astro-informatics.github.io/DarkMappy
.. |CodeCov| image:: https://codecov.io/gh/astro-informatics/code_template/branch/main/graph/badge.svg?token=A5ogGPslpU
    :target: https://codecov.io/gh/astro-informatics/DarkMappy
.. |GPL License| image:: https://img.shields.io/badge/License-GPL-blue.svg
    :target: http://perso.crans.org/besson/LICENSE.html
.. |ArXiv| image:: http://img.shields.io/badge/arXiv-xxxx.xxxxx-orange.svg?style=flat
    :target: https://arxiv.org/abs/xxxx.xxxxx

|logo| DarkMappy: hybrid Bayesian inference of the dark universe
=================================================================================================================

``darkmappy`` is a lightweight python package which implements the hybrid Bayesian dark-matter reconstruction techniques 
outlined on the plane in `Price *et al.* 2019 <https://academic.oup.com/mnras/article-abstract/506/3/3678/6319513>`_, and on the celestial sphere in `Price *et al.* 2021 <https://academic.oup.com/mnras/article/500/4/5436/5986632>`_ and `Wallis *et al.* 2021 <https://academic.oup.com/mnras/article-abstract/509/3/4480/6424933>`_. These techniques are based on *maximum a posteriori* estimation, and by construction support principled uncertainty quantification, by leveraging recent advances in probability concentration theory (`Pereyra *et al.* 2016 <https://epubs.siam.org/doi/10.1137/16M1071249>`_).


Installation
============

Add some basic installation instructions here.
    
Documentation
=============

Link to the full documentation (when deployed).

Contributors
============
Matthew A. Price & Contributors

Attribution
===========
A BibTeX entry for <project-name> is:

.. code-block:: 

     @article{<project-name>, 
        author = {Author~List},
         title = {"A totally amazing name"},
       journal = {ArXiv},
        eprint = {arXiv:0000.00000},
          year = {what year is it?!}
     }

License
=======

``darkmappy`` is released under the GPL-3 license (see `LICENSE.txt <https://github.com/astro-informatics/DarkMappy/blob/main/LICENSE.txt>`_), subject to 
the non-commercial use condition (see `LICENSE_EXT.txt <https://github.com/astro-informatics/DarkMappy/blob/main/LICENSE_EXT.txt>`_)

.. code-block::

     LatentWaves
     Copyright (C) 2022 Matthew A. Price & contributors

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



