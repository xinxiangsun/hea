import os
import numpy as np
import matplotlib.pyplot as plt
from gdt.missions.fermi.gbm.poshist import GbmPosHist
from gdt.missions.fermi.gbm.finders import ContinuousFinder
from gdt.missions.fermi.gbm.detectors import GbmDetectors
from gdt.missions.fermi.plot import FermiEarthPlot
from gdt.missions.fermi.gbm.saa import GbmSaa
from gdt.missions.fermi.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
from gdt.core.plot.sky import EquatorialPlot
from gdt.core.plot.plot import SkyPoints


class gbmGeometry:
    def __init__(self, poshist_file, ra=None, dec=None, utc=None):
        self._poshist_file = poshist_file
        self._file = poshist_file
        self._ra = ra
        self._dec = dec
        self._utc = utc
        self._read()

    @classmethod
    def from_utc(cls, utc, ra=None, dec=None, datadir=None):
        finder = ContinuousFinder(utc)
        if datadir is None:
            datadir = './'
        finder.get_poshist(datadir)
        from glob import glob
        y, m, d = utc[:4], utc[5:7], utc[8:10]
        poshist_files = glob(os.path.join(datadir, f'glg_poshist_all_{y}{m}{d}_v*.fit'))
        if not poshist_files:
            raise FileNotFoundError('No poshist file found!')
        return cls(poshist_files[0], ra=ra, dec=dec, utc=utc)

    def _read(self):
        self._poshist = GbmPosHist.open(self._poshist_file)
        self._frame = self._poshist.get_spacecraft_frame()
        self._states = self._poshist.get_spacecraft_states()
        if self._utc:
            self._srctime = Time(self._utc, scale='utc', precision=9)
            if self._ra is not None and self._dec is not None:
                self._coord = SkyCoord(self._ra, self._dec, frame='icrs', unit='deg')
                self._one_frame = self._frame.at(self._srctime)

    @property
    def file(self):
        return self._file

    @file.setter
    def file(self, new_file):
        self._file = new_file
        self._read()

    def saa_passage(self, met):
        # 判断 met 时刻是否在 SAA 区域
        t = Time(met, format='fermi')
        frame = self._frame.at(t)
        saa = GbmSaa()
        return saa.in_saa(frame.earth_location.lat, frame.earth_location.lon)

    def location_visible(self, ra, dec, met):
        # 判断某天区在 met 时刻是否可见
        coord = SkyCoord(ra, dec, frame='icrs', unit='deg')
        t = Time(met, format='fermi')
        frame = self._frame.at(t)
        return frame.location_visible(coord)

    def sun_visible(self, met):
        # 判断 met 时刻太阳是否可见
        t = Time(met, format='fermi')
        idx = np.argmin(np.abs(self._states['time'].value - met))
        return self._states['sun'][idx]

    def detector_angle(self, ra, dec, det, met):
        # 计算某探测器与天区的夹角
        coord = SkyCoord(ra, dec, frame='icrs', unit='deg')
        t = Time(met, format='fermi')
        frame = self._frame.at(t)
        return frame.detector_angle(det, coord)

    def extract_skymap(self, ra, dec, met, savepath='./geometry'):
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        t = Time(met, format='fermi')
        frame = self._frame.at(t)
        eqplot = EquatorialPlot()
        eqplot.add_frame(frame)
        eqplot.ax.scatter(ra, dec, marker='*', s=75, c='r', edgecolors='r')
        plt.savefig(os.path.join(savepath, 'sky_map.pdf'))

    def extract_earthmap(self, met, dt=1000, savepath='./geometry'):
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        t = Time(met, format='fermi')
        frame = self._frame
        earthplot = FermiEarthPlot(saa=GbmSaa())
        earthplot.add_spacecraft_frame(
            frame,
            tstart=Time(met-dt, format='fermi'),
            tstop=2*dt,
            trigtime=t
        )
        earthplot.standard_title()
        plt.savefig(os.path.join(savepath, 'earth_map.pdf'))

