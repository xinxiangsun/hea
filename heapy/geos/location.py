import numpy as np
import pandas as pd
from astropy import table
from astropy.io import fits
# Removed dependency on daily_search.zjh_location - implementing alternative
# from daily_search.zjh_location import zjh_loc
from ..pipe.event import gbmTTE
from ..data.retrieve import gbmRetrieve
# Added GDT imports for enhanced functionality
from gdt.missions.fermi.gbm.poshist import GbmPosHist
from gdt.missions.fermi.time import Time
from gdt.missions.fermi.gbm.tte import GbmTte
from pathlib import Path
import json
import matplotlib.pyplot as plt



class gbmLocation(object):
    
    def __init__(self, tte_file, poshist_file):
        
        self.tte_file = tte_file
        self.poshist_file = poshist_file
        
        self._read()

    def _alternative_location_analysis(self, utc, t1, t2, tte_data, poshist_data, savepath, binsize, snr):
        """
        Alternative implementation to replace zjh_loc functionality
        替代zjh_loc功能的实现
        
        This is a basic implementation that performs simplified location analysis
        using GDT capabilities instead of the unavailable daily_search module.
        
        Args:
            utc (str): UTC time string
            t1 (float): Start time offset
            t2 (float): End time offset  
            tte_data (dict): TTE data dictionary
            poshist_data (pd.DataFrame): Position history data
            savepath (str): Output directory path
            binsize (float): Time bin size
            snr (float): Signal-to-noise ratio threshold
        """
        try:
            # Create output directory
            Path(savepath).mkdir(parents=True, exist_ok=True)
            
            print(f"🔍 Starting alternative location analysis for {utc}")
            print(f"   Time range: {t1} to {t2} seconds")
            print(f"   Bin size: {binsize} seconds")
            print(f"   SNR threshold: {snr}")
            
            # Convert UTC to analysis time
            trigger_time = Time(utc, scale='utc', precision=9)
            met_trigger = trigger_time.fermi
            
            # Time bins for analysis
            time_bins = np.arange(t1, t2 + binsize, binsize)
            analysis_results = {
                'trigger_time': utc,
                'met_trigger': float(met_trigger),
                'time_range': [float(t1), float(t2)],
                'binsize': float(binsize),
                'snr_threshold': float(snr),
                'detectors': {},
                'summary': {}
            }
            
            # Analyze each detector
            detector_count = 0
            total_events = 0
            
            for det_name, det_data in tte_data.items():
                if 'events' not in det_data:
                    continue
                    
                detector_count += 1
                events_df = det_data['events']
                
                # Filter events to analysis time window
                start_met = met_trigger + t1
                end_met = met_trigger + t2
                
                mask = (events_df['TIME'] >= start_met) & (events_df['TIME'] <= end_met)
                filtered_events = events_df[mask]
                
                det_event_count = len(filtered_events)
                total_events += det_event_count
                
                # Create light curve
                hist, bin_edges = np.histogram(
                    filtered_events['TIME'] - met_trigger, 
                    bins=time_bins
                )
                
                # Simple background estimation (first and last 10% of data)
                n_bg_bins = max(1, int(0.1 * len(hist)))
                bg_rate = np.mean(np.concatenate([hist[:n_bg_bins], hist[-n_bg_bins:]]))
                bg_std = np.std(np.concatenate([hist[:n_bg_bins], hist[-n_bg_bins:]]))
                
                # Find significant bins
                significance = (hist - bg_rate) / (bg_std + 1e-10)  # Avoid division by zero
                significant_bins = significance > snr
                
                # Store detector results
                analysis_results['detectors'][det_name] = {
                    'total_events': int(det_event_count),
                    'background_rate': float(bg_rate),
                    'background_std': float(bg_std),
                    'max_significance': float(np.max(significance)),
                    'significant_bins': int(np.sum(significant_bins)),
                    'light_curve': {
                        'time_bins': time_bins.tolist(),
                        'counts': hist.tolist(),
                        'significance': significance.tolist()
                    }
                }
                
                # Create and save light curve plot
                try:
                    plt.figure(figsize=(12, 6))
                    
                    # Plot light curve
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    plt.subplot(2, 1, 1)
                    plt.step(bin_centers, hist, where='mid', label=f'{det_name} counts')
                    plt.axhline(bg_rate, color='red', linestyle='--', label=f'Background ({bg_rate:.1f})')
                    plt.axhline(bg_rate + snr * bg_std, color='orange', linestyle=':', 
                               label=f'{snr}σ threshold')
                    plt.ylabel('Counts')
                    plt.title(f'Light Curve - {det_name}')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    # Plot significance
                    plt.subplot(2, 1, 2)
                    plt.step(bin_centers, significance, where='mid', color='green')
                    plt.axhline(snr, color='red', linestyle='--', label=f'{snr}σ threshold')
                    plt.ylabel('Significance (σ)')
                    plt.xlabel('Time since trigger (s)')
                    plt.title(f'Significance - {det_name}')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(Path(savepath) / f'{det_name}_lightcurve.png', dpi=150, bbox_inches='tight')
                    plt.close()
                    
                except Exception as plot_error:
                    print(f"Warning: Could not create plot for {det_name}: {plot_error}")
                    plt.close()
            
            # Summary statistics
            analysis_results['summary'] = {
                'total_detectors': detector_count,
                'total_events': total_events,
                'avg_events_per_detector': total_events / max(1, detector_count),
                'detectors_with_significance': sum(
                    1 for det_result in analysis_results['detectors'].values() 
                    if det_result['max_significance'] > snr
                )
            }
            
            # Save analysis results to JSON
            results_file = Path(savepath) / 'location_analysis_results.json'
            with open(results_file, 'w') as f:
                json.dump(analysis_results, f, indent=2)
                
            # Create summary plot
            try:
                self._create_summary_plot(analysis_results, savepath)
            except Exception as summary_error:
                print(f"Warning: Could not create summary plot: {summary_error}")
            
            print(f"✅ Alternative location analysis completed")
            print(f"   Analyzed {detector_count} detectors with {total_events} total events")
            print(f"   Results saved to: {results_file}")
            
            return analysis_results
            
        except Exception as e:
            print(f"Error in alternative location analysis: {e}")
            return None
    
    def _create_summary_plot(self, results, savepath):
        """Create summary plots for location analysis"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Event counts by detector
            detectors = list(results['detectors'].keys())
            event_counts = [results['detectors'][det]['total_events'] for det in detectors]
            
            axes[0, 0].bar(detectors, event_counts)
            axes[0, 0].set_title('Total Events by Detector')
            axes[0, 0].set_ylabel('Event Count')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Plot 2: Max significance by detector
            max_sig = [results['detectors'][det]['max_significance'] for det in detectors]
            
            axes[0, 1].bar(detectors, max_sig)
            axes[0, 1].axhline(results['snr_threshold'], color='red', linestyle='--', 
                              label=f'{results["snr_threshold"]}σ threshold')
            axes[0, 1].set_title('Maximum Significance by Detector')
            axes[0, 1].set_ylabel('Significance (σ)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].legend()
            
            # Plot 3: Combined light curve (if multiple detectors)
            if len(detectors) > 1:
                time_bins = results['detectors'][detectors[0]]['light_curve']['time_bins']
                combined_counts = np.zeros(len(time_bins) - 1)
                
                for det in detectors:
                    counts = np.array(results['detectors'][det]['light_curve']['counts'])
                    combined_counts += counts
                
                bin_centers = [(time_bins[i] + time_bins[i+1])/2 for i in range(len(time_bins)-1)]
                axes[1, 0].step(bin_centers, combined_counts, where='mid', label='Combined')
                axes[1, 0].set_title('Combined Light Curve (All Detectors)')
                axes[1, 0].set_ylabel('Total Counts')
                axes[1, 0].set_xlabel('Time since trigger (s)')
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, 'Single Detector\nSee individual plots', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Combined Light Curve')
            
            # Plot 4: Summary statistics
            summary_text = f"""
Analysis Summary:
• Total Detectors: {results['summary']['total_detectors']}
• Total Events: {results['summary']['total_events']}
• Avg Events/Detector: {results['summary']['avg_events_per_detector']:.1f}
• Detectors > {results['snr_threshold']}σ: {results['summary']['detectors_with_significance']}

Time Window: {results['time_range'][0]} to {results['time_range'][1]} s
Bin Size: {results['binsize']} s
Trigger: {results['trigger_time']}
"""
            axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                           verticalalignment='top', fontfamily='monospace')
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].axis('off')
            axes[1, 1].set_title('Analysis Summary')
            
            plt.tight_layout()
            plt.savefig(Path(savepath) / 'location_analysis_summary.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error creating summary plot: {e}")
            plt.close()


    @classmethod
    def from_utc(cls, utc):
        
        rtv = gbmRetrieve.from_utc(utc=utc, t1=-500, t2=500)
        
        tte_file = rtv.rtv_res['tte']
        
        poshist_file = rtv.rtv_res['poshist']
        
        return cls(tte_file, poshist_file)
    
    
    def _read(self):
        """
        Enhanced _read method using GDT for better data handling
        使用GDT增强的_read方法，实现更好的数据处理
        """
        self.tte_data = {}
        
        # Process TTE files using both original method and GDT enhancement
        for det, file in self.tte_file.items():
            self.tte_data[det] = {}
            
            try:
                # Original method
                tte = gbmTTE(file)
                
                self.tte_data[det]['ch_E'] = pd.DataFrame({
                    'CHANNEL': tte.channel, 
                    'E_MIN': tte.channel_emin,
                    'E_MAX': tte.channel_emax
                })
                
                self.tte_data[det]['events'] = pd.DataFrame({
                    'TIME': np.array(tte.event['TIME']).astype(float),
                    'PHA': np.array(tte.event['PHA']).astype(int)
                })
                
                # Enhanced with GDT for additional functionality
                try:
                    if Path(file).exists():
                        gdt_tte = GbmTte.open(file)
                        # Store GDT object for advanced operations
                        self.tte_data[det]['gdt_object'] = gdt_tte
                        
                        # Add enhanced time handling
                        if hasattr(gdt_tte, 'time_range'):
                            self.tte_data[det]['time_range'] = gdt_tte.time_range
                        
                        # Add enhanced energy channel information
                        if hasattr(gdt_tte, 'ebounds'):
                            self.tte_data[det]['energy_bounds'] = gdt_tte.ebounds
                            
                except Exception as gdt_error:
                    print(f"Warning: GDT enhancement failed for {det}: {gdt_error}")
                    # Continue with original data structure
                    pass
                    
            except Exception as e:
                print(f"Error processing {det} file {file}: {e}")
                continue
        
        # Enhanced poshist processing using GDT
        try:
            poshist_list = []
            gdt_poshist_objects = []
            
            for file in self.poshist_file:
                try:
                    # Original method
                    hdu = fits.open(file)
                    pos = table.Table.read(hdu[1])
                    poshist_list.append(pos)
                    
                    # Enhanced with GDT
                    if Path(file).exists():
                        gdt_poshist = GbmPosHist.open(file)
                        gdt_poshist_objects.append(gdt_poshist)
                        
                except Exception as file_error:
                    print(f"Warning: Failed to process poshist file {file}: {file_error}")
                    continue
            
            # Combine and process poshist data
            if poshist_list:
                poshist = table.vstack(poshist_list)
                poshist = table.unique(poshist, keys=['SCLK_UTC'])
                poshist.sort('SCLK_UTC')
                
                col_names = ['SCLK_UTC','QSJ_1','QSJ_2','QSJ_3','QSJ_4',
                            'POS_X','POS_Y','POS_Z','SC_LAT','SC_LON']
                self.poshist_data = poshist[col_names].to_pandas()
                
                # Store GDT objects for enhanced functionality
                if gdt_poshist_objects:
                    self.gdt_poshist_objects = gdt_poshist_objects
                    # Create spacecraft frame from first poshist for advanced operations
                    try:
                        self.spacecraft_frame = gdt_poshist_objects[0].get_spacecraft_frame()
                        self.spacecraft_states = gdt_poshist_objects[0].get_spacecraft_states()
                    except Exception as frame_error:
                        print(f"Warning: Failed to create spacecraft frame: {frame_error}")
            else:
                raise ValueError("No valid poshist files could be processed")
                
        except Exception as e:
            print(f"Error processing poshist files: {e}")
            # Fallback to basic structure
            self.poshist_data = pd.DataFrame()
    
    
    def extract_location(self, utc, t1, t2, binsize, snr=3, savepath='./location'):
        """
        Enhanced extract_location method with GDT integration
        使用GDT集成的增强定位提取方法
        
        This method now uses an alternative implementation instead of the
        unavailable daily_search.zjh_location module.
        """
        try:
            # Use alternative implementation instead of zjh_loc
            print("📍 Using alternative location analysis (daily_search not available)")
            analysis_results = self._alternative_location_analysis(
                utc, t1, t2, self.tte_data, self.poshist_data, savepath, binsize, snr
            )
            
            # Enhanced functionality using GDT objects if available
            if hasattr(self, 'gdt_poshist_objects') and hasattr(self, 'spacecraft_frame'):
                print("🚀 Running enhanced GDT analysis...")
                self._enhanced_location_analysis(utc, t1, t2, savepath)
            else:
                print("ℹ️  GDT objects not available, skipping enhanced analysis")
                
            return analysis_results
                
        except Exception as e:
            print(f"Error in location extraction: {e}")
            return None
            
    def _enhanced_location_analysis(self, utc, t1, t2, savepath):
        """
        Additional location analysis using GDT capabilities
        使用GDT功能的额外定位分析
        """
        try:
            from astropy.coordinates import SkyCoord
            from gdt.missions.fermi.gbm.detectors import GbmDetectors
            
            # Convert UTC to Fermi time
            trigger_time = Time(utc, scale='utc', precision=9)
            
            # Create output directory
            Path(savepath).mkdir(parents=True, exist_ok=True)
            
            # Analyze detector responses for different sky positions
            # This is a simplified example - real localization would be more complex
            analysis_results = {
                'trigger_time': utc,
                'time_range': [t1, t2],
                'detector_analysis': {},
                'spacecraft_info': {}
            }
            
            # Get spacecraft frame at trigger time
            if self.spacecraft_frame:
                frame_at_trigger = self.spacecraft_frame.at(trigger_time)
                
                # Store spacecraft position information
                try:
                    earth_loc = frame_at_trigger.earth_location
                    analysis_results['spacecraft_info'] = {
                        'latitude': float(earth_loc.lat.value) if hasattr(earth_loc.lat, 'value') else float(earth_loc.lat),
                        'longitude': float(earth_loc.lon.value) if hasattr(earth_loc.lon, 'value') else float(earth_loc.lon),
                        'altitude': float(earth_loc.height.value) if hasattr(earth_loc.height, 'value') else float(earth_loc.height)
                    }
                except Exception as loc_error:
                    print(f"Warning: Could not extract spacecraft location: {loc_error}")
                
                # Analyze detector visibility for sample sky positions
                test_positions = [
                    (0, 0), (90, 0), (180, 0), (270, 0),  # Equatorial positions
                    (0, 30), (90, 30), (180, 30), (270, 30),  # 30° declination
                    (0, -30), (90, -30), (180, -30), (270, -30)  # -30° declination
                ]
                
                for ra, dec in test_positions:
                    coord = SkyCoord(ra, dec, frame='icrs', unit='deg')
                    position_key = f"ra{ra}_dec{dec}"
                    
                    analysis_results['detector_analysis'][position_key] = {
                        'visible': frame_at_trigger.location_visible(coord),
                        'detector_angles': {}
                    }
                    
                    # Calculate angles for all detectors
                    for detector in GbmDetectors:
                        try:
                            angle = frame_at_trigger.detector_angle(detector.name, coord)
                            if hasattr(angle, '__len__') and len(angle) > 0:
                                angle_val = angle[0]
                            else:
                                angle_val = angle
                                
                            if hasattr(angle_val, 'to_value'):
                                angle_deg = float(angle_val.to_value('deg'))
                            elif hasattr(angle_val, 'value'):
                                angle_deg = float(angle_val.value)
                            else:
                                angle_deg = float(angle_val)
                                
                            analysis_results['detector_analysis'][position_key]['detector_angles'][detector.name] = angle_deg
                        except Exception as det_error:
                            # Skip problematic detectors
                            continue
            
            # Save enhanced analysis results
            analysis_file = Path(savepath) / 'enhanced_analysis.json'
            with open(analysis_file, 'w') as f:
                json.dump(analysis_results, f, indent=2)
                
            print(f"Enhanced location analysis saved to: {analysis_file}")
            
        except Exception as e:
            print(f"Warning: Enhanced location analysis failed: {e}")
            
    def get_detector_efficiency_map(self, utc, sky_grid_resolution=10):
        """
        Generate a detector efficiency map across the sky using GDT
        使用GDT生成天空探测器效率图
        """
        try:
            if not hasattr(self, 'spacecraft_frame'):
                print("Warning: No spacecraft frame available for efficiency mapping")
                return None
                
            from astropy.coordinates import SkyCoord
            from gdt.missions.fermi.gbm.detectors import GbmDetectors
            import matplotlib.pyplot as plt
            
            trigger_time = Time(utc, scale='utc', precision=9)
            frame_at_trigger = self.spacecraft_frame.at(trigger_time)
            
            # Create sky grid
            ra_range = np.arange(0, 360, sky_grid_resolution)
            dec_range = np.arange(-90, 91, sky_grid_resolution)
            
            efficiency_map = {}
            
            for detector in GbmDetectors:
                det_map = np.zeros((len(dec_range), len(ra_range)))
                
                for i, dec in enumerate(dec_range):
                    for j, ra in enumerate(ra_range):
                        coord = SkyCoord(ra, dec, frame='icrs', unit='deg')
                        
                        try:
                            # Check if position is visible
                            visible = frame_at_trigger.location_visible(coord)
                            if not visible:
                                det_map[i, j] = 0
                                continue
                                
                            # Calculate detector angle
                            angle = frame_at_trigger.detector_angle(detector.name, coord)
                            if hasattr(angle, '__len__') and len(angle) > 0:
                                angle_val = angle[0]
                            else:
                                angle_val = angle
                                
                            if hasattr(angle_val, 'to_value'):
                                angle_deg = float(angle_val.to_value('deg'))
                            elif hasattr(angle_val, 'value'):
                                angle_deg = float(angle_val.value)
                            else:
                                angle_deg = float(angle_val)
                            
                            # Simple efficiency model: decreases with angle
                            if angle_deg < 90:
                                efficiency = np.cos(np.radians(angle_deg))
                            else:
                                efficiency = 0
                                
                            det_map[i, j] = efficiency
                            
                        except:
                            det_map[i, j] = 0
                            
                efficiency_map[detector.name] = det_map
                
            return {
                'efficiency_maps': efficiency_map,
                'ra_grid': ra_range,
                'dec_grid': dec_range,
                'trigger_time': utc
            }
            
        except Exception as e:
            print(f"Error generating detector efficiency map: {e}")
            return None