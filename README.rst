
=======================================
Glasbey Categorical Color Palette Tools
=======================================

The glasbey library allows for the algorithmic creation of colour palettes designed for use with categorical data
using techniques from the paper Colour Displays for Categorical Images by Glasbey, Heijden, Toh and Gray. You don't
need to worry about the technical details however -- the glasbey library is easy to use.

It is quite common to require a colour palette for some categorical data such that each category has a visually
distinctive colour. Usually one relies upon predefined colour palettes such as those from
`ColorBrewer <https://colorbrewer2.org/#type=qualitative&scheme=Accent&n=3>`_, or provided by your plotting library of
choice. Unfortunately such palettes do not always meet your needs: perhaps they don't have enough distinct colours and
you don't want to re-use or cycle the palette; perhaps you have specific constraints you want to apply to get a
certain look to your palette. Fortunately we can use math and perceptual colour spaces to create new palettes that
maximize the perceptual visual distinctiveness of colours within constraints. It is also easy to extend an
existing palette, or seed a created palette with some initial colours (perhaps your company or institutions colours).
Lastly, glasbey makes it easy to generate block palettes, suitable for working with hierarchical categories.

-----------
Basic Usage
-----------

Creating new categorical colour palettes is as easy as single function call.

.. code:: python3

    import glasbey

    # Create a categorical palette with 15 colours
    glasbey.create_palette(palette_size=15)
    # Create a muted palette with 12 colours
    glasbey.create_palette(palette_size=12, lightness_bounds=(20, 40), chroma_bounds=(40, 50))

It is also easy to extend an existing palette, or create a new palette from some seed colours.

.. code:: python3

    import glasbey

    # Add an extra 5 colours to matplotlib's tab10 palette
    glasbey.extend_palette("tab10", palette_size=15)
    # Seed a palette with some initial colours
    glasbey.extend_palette(["#2a3e63", "#7088b8", "#fcaf3e", "#b87088"], palette_size=8)

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

    pip install glasbey

To manually install this package:

.. code:: bash

    wget https://github.com/lmcinnes/glasbey/archive/main.zip
    unzip main.zip
    rm main.zip
    cd glasbey-main
    python setup.py install

----------------
Acknowledgements
----------------

This library is heavily indebted to the `original glasbey library  <https://github.com/taketwo/glasbey>`_ by Sergey Alexandrov.

----------
References
----------

1) Glasbey, C., van der Heijden, G., Toh, V. F. K. and Gray, A. (2007),
   `Colour Displays for Categorical Images <http://onlinelibrary.wiley.com/doi/10.1002/col.20327/abstract>`_.
   Color Research and Application, 32: 304-309

2) Luo, M. R., Cui, G. and Li, C. (2006), `Uniform Colour Spaces Based on CIECAM02 Colour Appearance Model <http://onlinelibrary.wiley.com/doi/10.1002/col.20227/abstract>`_.
   Color Research and Application, 31: 320–330

-------
License
-------

Glasbey is MIT licensed. See the LICENSE file for details.

------------
Contributing
------------

Contributions are more than welcome! If you have ideas for features of projects please get in touch. Everything from
code to notebooks to examples and documentation are all *equally valuable* so please don't feel you can't contribute.
To contribute please `fork the project <https://github.com/lmcinnes/glasbey/issues#fork-destination-box>`_ make your
changes and submit a pull request. We will do our best to work through any issues with you and get your code merged in.