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
        # Check if the spacecraft is in the South Atlantic Anomaly (SAA) at given MET time
        try:
            t = Time(met, format='fermi')
            frame = self._frame.at(t)
            saa = GbmSaa()
            
            # Use the spacecraft location to check SAA passage
            # GDT implementation uses the spacecraft frame's earth location
            earth_loc = frame.earth_location
            lat = earth_loc.lat.value if hasattr(earth_loc.lat, 'value') else earth_loc.lat
            lon = earth_loc.lon.value if hasattr(earth_loc.lon, 'value') else earth_loc.lon
            
            # Check if current location is in SAA region
            # SAA is typically defined around latitude -30° and longitude -40° to -10°
            # Using GBM SAA definition via GDT
            if hasattr(saa, 'contains'):
                return saa.contains(lat, lon)
            else:
                # Fallback: approximate SAA region check
                return (-50 < lat < -10) and (-60 < lon < 20)
        except Exception as e:
            print(f"Error checking SAA passage: {e}")
            return False

    def location_visible(self, ra, dec, met):
        # 判断某天区在 met 时刻是否可见
        # Check if a sky location is visible at given MET time
        try:
            coord = SkyCoord(ra, dec, frame='icrs', unit='deg')
            t = Time(met, format='fermi')
            frame = self._frame.at(t)
            
            # Use GDT's location_visible method from spacecraft frame
            return frame.location_visible(coord)
        except Exception as e:
            print(f"Error checking location visibility: {e}")
            return False

    def sun_visible(self, met):
        # 判断 met 时刻太阳是否可见
        # Check if the Sun is visible at given MET time
        try:
            t = Time(met, format='fermi')
            
            # Find the closest time index in states data
            if hasattr(self._states['time'], 'value'):
                time_values = self._states['time'].value
            else:
                # Handle different data structures
                time_values = np.array(self._states['time'])
                
            idx = np.argmin(np.abs(time_values - met))
            
            # Get sun visibility status from states
            if hasattr(self._states['sun'], '__getitem__'):
                return bool(self._states['sun'][idx])
            else:
                # Fallback if sun data format is different
                return False
        except Exception as e:
            print(f"Error checking sun visibility: {e}")
            return False

    def detector_angle(self, ra, dec, det, met):
        # 计算某探测器与天区的夹角
        # Calculate the angle between a detector and sky location at given MET time
        try:
            coord = SkyCoord(ra, dec, frame='icrs', unit='deg')
            t = Time(met, format='fermi')
            frame = self._frame.at(t)
            
            # Use GDT's detector_angle method
            angle_result = frame.detector_angle(det, coord)
            
            # Handle different return formats
            if hasattr(angle_result, '__len__') and len(angle_result) > 0:
                angle = angle_result[0]
            else:
                angle = angle_result
                
            # Convert to degrees
            if hasattr(angle, 'to_value'):
                return angle.to_value('deg')
            elif hasattr(angle, 'value'):
                return angle.value
            else:
                return float(angle)
        except Exception as e:
            print(f"Error calculating detector angle: {e}")
            return None

    def extract_skymap(self, ra, dec, met, savepath='./geometry'):
        # 生成天空图，参考 GBMObservation.skymap 方法
        # Generate sky map, based on GBMObservation.skymap method
        try:
            if not os.path.exists(savepath):
                os.makedirs(savepath)
                
            t = Time(met, format='fermi')
            frame = self._frame.at(t)
            coord = SkyCoord(ra, dec, frame='icrs', unit='deg')
            
            # Create equatorial plot using GDT
            eqplot = EquatorialPlot(interactive=False)
            eqplot.add_frame(frame)
            
            # Configure sun appearance if available
            if hasattr(eqplot, 'sun') and eqplot.sun is not None:
                try:
                    eqplot.sun.zorder = 2
                    eqplot.sun.size = 300
                except:
                    pass
            
            # Add coordinate system label
            eqplot.ax.text(0.02, 0.95, "ICRS", transform=eqplot.ax.transAxes, 
                          fontsize=15, color='red', ha='left', fontweight='bold')
            
            # Add source position using SkyPoints
            try:
                ra_deg = coord.gcrs.ra.deg if hasattr(coord.gcrs.ra, 'deg') else coord.ra.deg
                dec_deg = coord.gcrs.dec.deg if hasattr(coord.gcrs.dec, 'deg') else coord.dec.deg
            except:
                ra_deg = ra
                dec_deg = dec
                
            SkyPoints(x=ra_deg, y=dec_deg, ax=eqplot.ax, 
                     label=f'Source ({ra:.2f}, {dec:.2f})', 
                     color='red', marker='*', s=100, zorder=10)
            
            # Add legend and title
            eqplot.ax.legend(loc='upper right', framealpha=0.8)
            eqplot.ax.set_title(f'Sky Map at MET {met}', 
                               fontsize=14, fontweight='bold', pad=20)
            
            # Save plot
            plt.savefig(os.path.join(savepath, 'sky_map.pdf'), 
                       bbox_inches='tight', facecolor='white')
            plt.close()  # Close to free memory
            
        except Exception as e:
            print(f"Error generating sky map: {e}")
            plt.close()

    def extract_earthmap(self, met, dt=1000, savepath='./geometry'):
        # 生成地球轨迹图，参考 GBMObservation.earthmap 方法
        # Generate Earth trajectory map, based on GBMObservation.earthmap method
        try:
            if not os.path.exists(savepath):
                os.makedirs(savepath)
                
            t = Time(met, format='fermi')
            frame = self._frame
            
            # Initialize SAA and create Earth plot using GDT
            saa = GbmSaa()
            earthplot = FermiEarthPlot(saa)
            
            # Calculate time range
            start_time = Time(met - dt, format='fermi')
            duration = 2 * dt  # Total duration
            
            # Add spacecraft trajectory to Earth plot
            earthplot.add_spacecraft_frame(
                frame,
                tstart=start_time,
                tstop=duration,
                trigtime=t
            )
            
            # Add standard title
            earthplot.standard_title()
            
            # Add custom title with MET information
            plt.suptitle(f'Earth Trajectory at MET {met} (±{dt}s)', 
                        fontsize=14, fontweight='bold')
            
            # Save plot
            plt.savefig(os.path.join(savepath, 'earth_map.pdf'),
                       bbox_inches='tight', facecolor='white')
            plt.close()  # Close to free memory
            
        except Exception as e:
            print(f"Error generating Earth map: {e}")
            plt.close()

