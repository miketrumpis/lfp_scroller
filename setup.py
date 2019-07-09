from setuptools import setup, find_packages

setup(
    name='lfp_scroller',
    version='0.1',
    packages=find_packages(),
    package_data={'fast_scroller.pyqtgraph_extensions': ['*.png']},
    scripts=['launch_scroller.py']
    )
