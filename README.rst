
=======================================
Glasbey Categorical Color Palette Tools
=======================================

Algorithmically create or extend categorical colour palettes.

------------
Installation
------------

Glasbey requires:
 * numba
 * numpy
 * colorspacious
 * matplotlib

Glasbey can be installed via pip:

.. code:: bash

    pip install galsbey

To manually install this package:

.. code:: bash

    wget https://github.com/lmcinnes/glasbey/archive/master.zip
    unzip master.zip
    rm master.zip
    cd glasbey-master
    python setup.py install

----------------
Acknowledgements
----------------

This library is heavily indebted to the [original glasbey library](https://github.com/taketwo/glasbey) by Sergey Alexandrov. 

----------
References
----------

1) Glasbey, C., van der Heijden, G., Toh, V. F. K. and Gray, A. (2007),
   `Colour Displays for Categorical Images <http://onlinelibrary.wiley.com/doi/10.1002/col.20327/abstract>`_.
   Color Research and Application, 32: 304-309

2) Luo, M. R., Cui, G. and Li, C. (2006),
   `Uniform Colour Spaces Based on CIECAM02 Colour Appearance Model <http://onlinelibrary.wiley.com/doi/10.1002/col.20227/abstract>`_.
   Color Research and Application, 31: 320–330