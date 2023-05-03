from setuptools import setup

# get the version here
pkg_vars = {}

with open("version.py") as fp:
    exec(fp.read(), pkg_vars)

setup(
    name='sn_phystools',
    version=pkg_vars['__version__'],
    description='Set of tools for SN studies',
    url='http://github.com/lsstdesc/sn_phystools',
    author='Philippe Gris',
    author_email='philippe.gris@clermont.in2p3.fr',
    license='BSD',
    packages=['sn_desc_ddf_strategy', 'sn_analysis', 'sn_cosmology'],
    # All files from folder sn_phystools_input
    # package_data={'sn_metrics_input': ['*.txt']},
    python_requires='>=3.5',
    zip_safe=False,
    install_requires=[
        'sn_tools>=0.1',
    ],
)
