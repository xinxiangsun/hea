# *Based on heapy，add astro-gdt, soxs, astro-elisa ... higher python version friendly.*

[![License: GPL v3](https://img.shields.io/github/license/jyangch/heapy?color=blue&logo=open-source-initiative&logoColor=white&style=for-the-badge)](https://www.gnu.org/licenses/gpl-3.0)

## Prerequisites

### HEASoft

_Heapy_ will invoke certain software and commands from HEASoft, such as `xselect` and `ximage`. Please ensure that [`HEASoft`](https://heasarc.gsfc.nasa.gov/docs/software/heasoft/#install) is correctly installed on your system, and that the [`Calibration Database`](https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/install.html) (CALDB) for the mission (e.g., `Swift`)  you are processing is also properly installed.

### Fermi GBM Response Generator

_Heapy_ generates the response matrix files for Fermi GBM by invoking [`gbm_drm_gen`](https://github.com/grburgess/gbm_drm_gen). It is recommended to install my forked Python packages, which have been fine-tuned to resolve compatibility issues with newer versions of `numpy` and `astropy`, and to use TTE data instead of CSPEC data. The specific installation procedure is as follows:

```
git clone https://github.com/xinxiangsun/responsum.git
pip install ./responsum

git clone https://github.com/xinxiangsun/gbmgeometry.git
pip install ./gbmgeometry

git clone https://github.com/xinxiangsun/gbm_drm_gen.git
pip install ./gbm_drm_gen
```


## Installation

_original Heapy_ is available via `pip`:

```bash
git clone https://github.com/xinxiangsun/hea.git
pip install ./hea
```

**NOTE**: The package is forked from https://github.com/jyang/heapy, 做了一些调整, 使用了更为现代的包以适应更新的scipy和numpy版本. 接口上略有不同
The package name of _heapy_ in pypi is registered as `heapyx` rather than `heapy`, as the latter has already been taken.

## Documentation

If you wish to learn about the usage, you may check the [`examples`](https://github.com/jyangch/heapy/tree/main/examples).

## License

_Heapy_ is distributed under the terms of the [`GPL-3.0`](https://www.gnu.org/licenses/gpl-3.0-standalone.html) license.
