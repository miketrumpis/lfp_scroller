LFP Scroller
============

A fast & light scroller based on `pyqtgraph <http://www.pyqtgraph.org/>`_, Enthought `traitsui <http://docs.enthought.com/traitsui/>`_, and HDF5.

Installation
------------

First steps: install `ecogdata`_ and `ecoglib`_ in order, from source (follow links for instructions).
At this point, you have chosen your flavor of virtual environment.

Next, clone this repository:

.. code-block:: bash

    $ git clone https://github.com/miketrumpis/lfp_scroller.git

**Choose whether to use PyQt5 or PySide2.**

* PyQt5: this is probably the best option (presently), but it is known not to work on Windows 8
* PySide2: also works, has a less restrictive license

The LFP scroller can use either package.
With PySide2, you must set the ``QT_API=pyside2`` environment variable (a la matplotlib).

*Note: if using conda, you may prefer to conda install the requirements in ``setup.cfg``.*

Pip install either as:

.. code-block:: bash

    $ pip install ./lfp_scroller[pyqt]

or as

.. code-block:: bash

    $ pip install ./lfp_scroller[pyside2]

Entry point
-----------

The program is run with a dead simple script:

.. code-block:: bash

    $ launch_scroller.py

.. image:: docs/images/newdemo.png?raw=true

For usage, see "docs" directory.


.. _ecogdata: https://github.com/miketrumpis/ecogdata
.. _ecoglib: https://bitbucket.org/tneuro/ecoglib/src/master/
