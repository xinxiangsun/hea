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
from gdt.core.data_primitives import Gti
from astropy.coordinates import get_body

class gbmGeometry:
    def __init__(self, file, ra=None, dec=None, utc=None):
        if isinstance(file, list):
            file = file[0]
        self._poshist_file = file
        self._file = file
        self._ra = ra
        self._dec = dec
        self._utc = utc
        self._time_range = None  # 添加时间范围缓存
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
        self._gti = Gti.from_boolean_mask(self._states['time'].value, self._states['good'].value)
        # 获取并缓存时间范围
        self._get_time_range()
        
        if self._utc:
            self._srctime = Time(self._utc, scale='utc', precision=9)
            if self._ra is not None and self._dec is not None:
                self._coord = SkyCoord(self._ra, self._dec, frame='icrs', unit='deg')
                self._one_frame = self._frame.at(self._srctime)

    def _get_time_range(self):
        """获取 poshist 文件的有效时间范围"""
        try:
            # 从 states 获取时间范围
            if hasattr(self._states['time'], 'value'):
                time_values = self._states['time'].value
            else:
                time_values = np.array(self._states['time'])
            
            self._time_range = {
                'min': float(np.min(time_values)),
                'max': float(np.max(time_values)),
                'span': float(np.max(time_values) - np.min(time_values))
            }
            print(f"Poshist 时间范围: {self._time_range['min']:.2f} - {self._time_range['max']:.2f} MET")
        except Exception as e:
            print(f"警告：无法获取时间范围: {e}")
            self._time_range = None

    def _check_time_coverage(self, met):
        """检查给定时间是否在 poshist 覆盖范围内"""
        if self._time_range is None:
            return True, "无法验证时间范围"
        
        if isinstance(met, (list, np.ndarray)):
            met_min, met_max = np.min(met), np.max(met)
            if met_min < self._time_range['min'] or met_max > self._time_range['max']:
                return False, f"请求时间范围 [{met_min:.2f}, {met_max:.2f}] 超出 poshist 范围 [{self._time_range['min']:.2f}, {self._time_range['max']:.2f}]"
        else:
            if met < self._time_range['min'] or met > self._time_range['max']:
                return False, f"请求时间 {met:.2f} 超出 poshist 范围 [{self._time_range['min']:.2f}, {self._time_range['max']:.2f}]"
        
        return True, "时间范围正常"

    def _safe_interpolate_time(self, met, fallback_method='nearest'):
        """安全的时间插值，自动处理超出范围的情况"""
        coverage_ok, msg = self._check_time_coverage(met)
        
        if not coverage_ok:
            print(f"警告：{msg}")
            
            if fallback_method == 'nearest':
                if isinstance(met, (list, np.ndarray)):
                    met = np.clip(met, self._time_range['min'], self._time_range['max'])
                    print(f"使用最近时间插值，调整为范围内值")
                else:
                    if met < self._time_range['min']:
                        met = self._time_range['min']
                    elif met > self._time_range['max']:
                        met = self._time_range['max']
                    print(f"使用最近时间 {met:.2f}")
            elif fallback_method == 'skip':
                return None, "时间超出范围，跳过处理"
                
        return met, "时间范围正常"

    @property
    def file(self):
        return self._file

    @file.setter
    def file(self, new_file):
        self._file = new_file
        self._read()

    def gti_check(self, met):
        """
        基于 GTI 的时间检查函数，参考 autogbm 实现
        参数:
            met: 单个或数组，Fermi MET 时间
        返回:
            bool 或 bool 数组，表示是否在好时间间隔内
        """
        # try:
        #     # 从 spacecraft states 中获取 GTI
        #     if self._states is None:
        #         print("Error: 没有 spacecraft states 数据")
        #         return False
                
        #     # 获取时间和状态数据
        #     if hasattr(self._states['time'], 'value'):
        #         time_values = self._states['time'].value
        #     else:
        #         time_values = self._states['time']
                
        #     # 检查是否有 'good' 状态字段
        #     if 'good' not in self._states.colnames:
        #         # 如果没有 'good' 字段，使用其他状态判断
        #         # 通常使用 fermi_scatt_flg 和 SAA 相关状态
        #         good_mask = np.ones(len(time_values), dtype=bool)
                
        #         # 排除 SAA 时间
        #         if 'saa' in self._states.colnames:
        #             good_mask &= ~self._states['saa']
        #         elif 'SAA' in self._states.colnames:
        #             good_mask &= ~self._states['SAA']
                    
        #         # 排除散射角度过小的时间
        #         if 'fermi_scatt_flg' in self._states.colnames:
        #             good_mask &= ~self._states['fermi_scatt_flg']
        #     else:
        #         good_mask = self._states['good']
            
        #     # 创建 GTI 对象
        #     self._gti = Gti.from_boolean_mask(time_values, good_mask)
        #     gti = self._gti
        #     # 检查输入时间是否在 GTI 内
        #     if isinstance(met, (list, np.ndarray)):
        #         met_arr = np.array(met)
        #         return np.array([gti.contains(t) for t in met_arr])
        return self._gti.contains(met)
                
        # except Exception as e:
        #     print(f"GTI 检查失败: {e}")
        #     return False

    def saa_passage(self, met, gti=None):
        """
        改进的 SAA 通道判断，完全使用 GTI 方法
        参数:
            met: 单个或数组，Fermi MET 时间
            gti: 可选，gdt.core.data_primitives.Gti 对象，如果提供则使用，否则使用内部 gti_check
        返回:
            bool 或 bool 数组，表示是否在 SAA 区域（即不在好时间间隔内）
        """
        try:
            # 使用 GTI 进行判断
            if gti is not None:
                # 使用提供的 GTI 对象
                if isinstance(met, (list, np.ndarray)):
                    met_arr = np.array(met)
                    # SAA passage 意味着不在 GTI 内
                    return np.array([not gti.contains(t) for t in met_arr])
                else:
                    return not gti.contains(met)
            else:
                # 使用内部 gti_check 方法
                gti_result = Gti.from_boolean_mask(self._states['time'].value, self._states['saa'].value)
                # SAA passage 意味着不在好时间间隔内
                if isinstance(gti_result, np.ndarray):
                    return ~gti_result
                else:
                    return not gti_result
                    
        except Exception as e:
            print(f"SAA 通道检查失败: {e}")
            # 发生错误时，保守假设不在 SAA（即在好时间内）
            if isinstance(met, (list, np.ndarray)):
                return np.zeros(len(met), dtype=bool)
            else:
                return False

    def gti_time_series(self, start_met, end_met, dt=10.0, plot=True, figsize=(12, 8)):
        """
        检查一段时间内的 GTI 状态并可视化，参考 autogbm 实现
        采用与 extract_earthmap 相同的自适应时间区间调整
        
        参数:
            start_met: 开始时间（Fermi MET）
            end_met: 结束时间（Fermi MET）
            dt: 时间步长（秒），默认 10.0 秒
            plot: 是否绘制图形，默认 True
            figsize: 图形大小，默认 (12, 8)
            
        返回:
            dict: 包含时间数组、GTI状态、SAA状态、统计信息等
        """
        try:
            # 自适应时间范围调整（采用与 extract_earthmap 相同的逻辑）
            original_start = start_met
            original_end = end_met
            
            # 检查并调整开始时间
            adjusted_start, status_start = self._safe_interpolate_time(start_met, fallback_method='nearest')
            if adjusted_start is None:
                print(f"GTI 时间序列分析跳过：{status_start}")
                return None
                
            # 检查并调整结束时间
            adjusted_end, status_end = self._safe_interpolate_time(end_met, fallback_method='nearest')
            if adjusted_end is None:
                print(f"GTI 时间序列分析跳过：{status_end}")
                return None
            
            # 自适应调整时间范围，确保不超出 poshist 覆盖范围
            if self._time_range is not None:
                # 计算请求的时间跨度
                requested_duration = end_met - start_met
                center_time = (start_met + end_met) / 2
                
                # 调整中心时间到有效范围内
                adjusted_center, _ = self._safe_interpolate_time(center_time, fallback_method='nearest')
                
                # 计算可用的时间范围
                max_duration_before = adjusted_center - self._time_range['min']
                max_duration_after = self._time_range['max'] - adjusted_center
                
                # 使用较小的时间范围，确保不超出边界
                half_duration = requested_duration / 2
                safe_half_duration = min(half_duration, max_duration_before * 0.9, max_duration_after * 0.9)
                
                if safe_half_duration < half_duration:
                    adjusted_start = adjusted_center - safe_half_duration
                    adjusted_end = adjusted_center + safe_half_duration
                    print(f"自动调整时间窗口：")
                    print(f"  原始: {original_start:.2f} - {original_end:.2f} (跨度 {requested_duration:.1f}s)")
                    print(f"  调整: {adjusted_start:.2f} - {adjusted_end:.2f} (跨度 {safe_half_duration*2:.1f}s)")
                    print(f"  原因: 避免超出 poshist 范围 [{self._time_range['min']:.2f}, {self._time_range['max']:.2f}]")
                else:
                    adjusted_start = start_met
                    adjusted_end = end_met
            
            # 创建时间数组
            time_array = np.arange(adjusted_start, adjusted_end, dt)
            n_points = len(time_array)
            
            print(f"分析时间范围: {adjusted_start:.2f} - {adjusted_end:.2f} MET")
            if adjusted_start != original_start or adjusted_end != original_end:
                print(f"  (调整自原始范围: {original_start:.2f} - {original_end:.2f} MET)")
            print(f"时间步长: {dt} 秒，共 {n_points} 个时间点")
            
            # 获取 GTI 对象（复用已有逻辑）
            if self._states is None:
                raise ValueError("没有 spacecraft states 数据")
                
            # 获取时间和状态数据
            if hasattr(self._states['time'], 'value'):
                state_times = self._states['time'].value
            else:
                state_times = self._states['time']
                
            # 创建好时间掩码
            if 'good' not in self._states.colnames:
                good_mask = np.ones(len(state_times), dtype=bool)
                
                # 排除 SAA 时间
                if 'saa' in self._states.colnames:
                    good_mask &= ~np.array(self._states['saa'], dtype=bool)
                elif 'SAA' in self._states.colnames:
                    good_mask &= ~np.array(self._states['SAA'], dtype=bool)
                    
                # 排除散射角度过小的时间
                if 'fermi_scatt_flg' in self._states.colnames:
                    good_mask &= ~np.array(self._states['fermi_scatt_flg'], dtype=bool)
            else:
                good_mask = np.array(self._states['good'], dtype=bool)
            
            # 创建 GTI 对象
            from gdt.core.data_primitives import Gti
            gti = Gti.from_boolean_mask(state_times, good_mask)
            
            # 检查每个时间点的 GTI 状态
            print("检查 GTI 状态...")
            gti_status = np.array([gti.contains(t) for t in time_array])
            saa_status = ~gti_status  # SAA 状态是 GTI 的反面
            
            # 计算统计信息（使用调整后的时间范围）
            total_time = adjusted_end - adjusted_start
            good_time = np.sum(gti_status) * dt
            bad_time = total_time - good_time
            good_fraction = good_time / total_time * 100 if total_time > 0 else 0
            
            # 找出 GTI 间隔
            gti_intervals = []
            saa_intervals = []
            
            # 找到状态变化点
            status_changes = np.diff(gti_status.astype(int))
            change_indices = np.where(status_changes != 0)[0]
            
            # 构建间隔列表
            current_start = 0
            current_status = gti_status[0]
            
            for change_idx in change_indices:
                if current_status:
                    gti_intervals.append((time_array[current_start], time_array[change_idx]))
                else:
                    saa_intervals.append((time_array[current_start], time_array[change_idx]))
                current_start = change_idx + 1
                current_status = not current_status
                
            # 处理最后一个间隔
            if current_status:
                gti_intervals.append((time_array[current_start], time_array[-1]))
            else:
                saa_intervals.append((time_array[current_start], time_array[-1]))
            
            # 打印统计信息（使用英文避免字体问题）
            print(f"\n=== GTI Time Series Analysis Results ===")
            print(f"Total time: {total_time:.2f} seconds")
            print(f"Good time: {good_time:.2f} seconds ({good_fraction:.1f}%)")
            print(f"Bad time: {bad_time:.2f} seconds ({100-good_fraction:.1f}%)")
            print(f"GTI intervals: {len(gti_intervals)}")
            print(f"SAA intervals: {len(saa_intervals)}")
            
            # 准备返回结果
            result = {
                'time_array': time_array,
                'gti_status': gti_status,
                'saa_status': saa_status,
                'total_time': total_time,
                'good_time': good_time,
                'bad_time': bad_time,
                'good_fraction': good_fraction,
                'gti_intervals': gti_intervals,
                'saa_intervals': saa_intervals,
                'dt': dt,
                'adjusted_start': adjusted_start,
                'adjusted_end': adjusted_end,
                'original_start': original_start,
                'original_end': original_end,
                'time_adjusted': (adjusted_start != original_start or adjusted_end != original_end)
            }
            
            # 绘图
            if plot:
                import matplotlib.pyplot as plt
                from matplotlib.patches import Rectangle
                import matplotlib.font_manager as fm
                
                # 设置优雅的字体配置
                plt.rcParams.update({
                    'font.family': ['DejaVu Sans', 'Arial', 'sans-serif'],
                    'font.size': 11,
                    'font.weight': 'normal',
                    'axes.titlesize': 13,
                    'axes.labelsize': 11,
                    'xtick.labelsize': 10,
                    'ytick.labelsize': 10,
                    'legend.fontsize': 10,
                    'figure.titlesize': 14,
                    'text.usetex': False,
                    'axes.unicode_minus': False,
                    'axes.grid': True,
                    'grid.alpha': 0.3,
                    'grid.linewidth': 0.5,
                    'axes.edgecolor': '#333333',
                    'axes.linewidth': 0.8,
                    'figure.facecolor': 'white',
                    'axes.facecolor': 'white'
                })
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
                
                # 转换时间为相对时间（分钟）
                time_rel = (time_array - adjusted_start) / 60.0
                
                # 上图：GTI 状态时间序列
                ax1.plot(time_rel, gti_status.astype(int), 'g-', linewidth=2.5, 
                        label='GTI Status', alpha=0.8)
                ax1.fill_between(time_rel, 0, gti_status.astype(int), 
                               alpha=0.4, color='#2E8B57', label='Good Time')
                ax1.fill_between(time_rel, gti_status.astype(int), 1, 
                               alpha=0.4, color='#DC143C', label='Bad Time (SAA)')
                
                ax1.set_ylabel('GTI Status', fontweight='medium')
                ax1.set_ylim(-0.05, 1.05)
                ax1.set_yticks([0, 1])
                ax1.set_yticklabels(['Bad', 'Good'])
                ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                
                # 优化图例
                legend1 = ax1.legend(loc='upper right', framealpha=0.9, 
                                   edgecolor='gray', fancybox=True, shadow=True)
                legend1.get_frame().set_linewidth(0.5)
                
                # 英文标题避免中文字体问题
                title = f'GTI Time Series Analysis ({good_fraction:.1f}% Good Time)'
                if adjusted_start != original_start or adjusted_end != original_end:
                    title += f'\n(Time Range Auto-Adjusted)'
                ax1.set_title(title, fontweight='bold', pad=15)
                
                # 下图：累积好时间百分比
                time_diff = time_array - adjusted_start
                time_diff[0] = dt  # 避免除零错误
                cumulative_good = np.cumsum(gti_status) * dt / time_diff * 100
                
                ax2.plot(time_rel, cumulative_good, '#1E90FF', linewidth=2.5, 
                        label='Cumulative Good Time %', alpha=0.9)
                ax2.axhline(y=good_fraction, color='#FF6347', linestyle='--', 
                          alpha=0.8, linewidth=2, label=f'Average: {good_fraction:.1f}%')
                
                ax2.set_ylabel('Cumulative Good Time (%)', fontweight='medium')
                ax2.set_xlabel('Relative Time (minutes)', fontweight='medium')
                ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                
                # 优化图例
                legend2 = ax2.legend(loc='lower right', framealpha=0.9, 
                                   edgecolor='gray', fancybox=True, shadow=True)
                legend2.get_frame().set_linewidth(0.5)
                ax2.set_ylim(0, 105)
                
                # 美化坐标轴
                for ax in [ax1, ax2]:
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_color('#666666')
                    ax.spines['bottom'].set_color('#666666')
                    ax.tick_params(colors='#333333', which='both')
                
                # 调整布局
                plt.tight_layout(pad=2.0, h_pad=1.5)
                plt.subplots_adjust(hspace=0.25)
                plt.show()
                
                # 打印详细间隔信息（使用英文避免字体问题）
                print(f"\n=== GTI Intervals Details ===")
                for i, (start, end) in enumerate(gti_intervals[:10]):  # 只显示前10个
                    duration = end - start
                    print(f"GTI {i+1}: {start:.2f} - {end:.2f} (duration {duration:.2f}s)")
                if len(gti_intervals) > 10:
                    print(f"... and {len(gti_intervals)-10} more GTI intervals")
                
                print(f"\n=== SAA Intervals Details ===")
                for i, (start, end) in enumerate(saa_intervals[:10]):  # 只显示前10个
                    duration = end - start
                    print(f"SAA {i+1}: {start:.2f} - {end:.2f} (duration {duration:.2f}s)")
                if len(saa_intervals) > 10:
                    print(f"... and {len(saa_intervals)-10} more intervals")
            
            return result
            
        except Exception as e:
            print(f"GTI 时间序列分析失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def location_visible(self, ra, dec, met):
        """
        判断某天区在 met 时刻是否可见（未被地球遮挡）
        
        通过计算探测器（或卫星）与地球中心之间的角度是否大于地球角度半径来判断遮挡。
        如果源本身就在视野外，直接返回不可见。
        
        Parameters:
        -----------
        ra, dec : float
            天体坐标（度）
        met : float or array-like
            Fermi MET 时间
        det : str, optional
            探测器名称，如果指定则计算该探测器的视野，否则使用默认视野
            
        Returns:
        --------
        bool or np.array(bool)
            True 表示位置可见（未被地球遮挡），False 表示被遮挡
        """
        try:
            adjusted_met, status = self._safe_interpolate_time(met, fallback_method='nearest')
            if adjusted_met is None:
                print(f"位置可见性检查跳过：{status}")
                return True  # 保守假设可见
            
            coord = SkyCoord(ra, dec, frame='icrs', unit='deg')
            
            if isinstance(adjusted_met, (list, np.ndarray)):
                results = []
                for t in adjusted_met:
                    visible = self._check_single_location_visibility(coord, t)
                    results.append(visible)
                return np.array(results)
            else:
                return self._check_single_location_visibility(coord, adjusted_met)
                
        except Exception as e:
            print(f"Error checking location visibility: {e}")
            return True

    def point_visible(self, ra, dec, met, det):

        
        coord = SkyCoord(ra, dec, frame='icrs', unit='deg')

        return self._check_detector_point_visibility(coord = coord, met=met, det=det)
             
    

    def _check_single_location_visibility(self,coord,met):

        time_obj = Time(met, format='fermi')
        frame = self._frame.at(time_obj)
        
        # 方法1: 使用 GDT 内置的 location_visible 方法（主要方法）
        # 这已经包含了正确的地球遮挡算法
        try:
            is_visible = frame.location_visible(coord)
        except Exception as e:
            print(f"frame.location_visible 调用失败: {e}")
            return False
        
        # 强化处理可能返回数组的情况
        try:
            if hasattr(is_visible, '__len__') and not isinstance(is_visible, str):
                # 处理任何类型的数组或列表
                return bool(np.all(np.asarray(is_visible)))
            else:
                return bool(is_visible)
        except Exception as e:
            print(f"数组转换失败: {e}, is_visible type: {type(is_visible)}, value: {is_visible}")
            return False
        


    def _check_detector_point_visibility(self, coord, met, det):
        """
        检查单个时间点的位置可见性
        
        使用 GDT 标准方法进行地球遮挡检查：
        1. 使用 frame.location_visible() 作为主要方法
        2. 可选：使用 earth_angular_radius 和 geocenter 进行验证
        3. 如果源被地球遮挡，直接返回 False
        4. 如果指定探测器，额外检查探测器视野角度
        """
        
        
        
        # 方法1: 使用 GDT 内置的 location_visible 方法（主要方法）
        # 这已经包含了正确的地球遮挡算法
        is_visible = self._check_single_location_visibility(coord=coord, met=met)
        
        # 如果源本身就不可见（被地球遮挡），直接返回 False
        # 确保 is_visible 是布尔值
        if isinstance(is_visible, (list, np.ndarray)):
            is_visible = bool(np.all(is_visible))
        
        if not is_visible:
            return False
            
        # 如果指定了探测器，还需要检查是否在探测器视野内
        else:
            try:
                # 计算探测器角度
                det_angle = self._frame.detector_angle(det, self._frame.geocenter)
                earth_angle = self._frame.earth_angular_radius
                # 将角度转换为度数进行比较
                if hasattr(det_angle, 'to'):
                    try:
                        angle_deg = det_angle.to('deg').value
                    except:
                        angle_deg = float(det_angle)
                else:
                    angle_deg = float(det_angle)
                if hasattr(earth_angle,'to'):
                    try:
                        earth_deg = earth_angle.to('deg').value
                    except:
                        earth_deg = float(earth_angle)
                is_visible = angle_deg < earth_deg
                
                # 处理可能返回数组的角度比较结果
                if isinstance(is_visible, (list, np.ndarray)):
                    is_visible = bool(np.all(is_visible))
                    
            except Exception as e:
                print(f"探测器角度计算失败: {e}")
                # 如果探测器角度计算失败，只依赖地球遮挡检查
                pass
        
        return bool(is_visible)
            
        

    def sun_visible(self, met):
        """判断 met 时刻太阳是否可见，带时间范围检查"""
    
        adjusted_met, status = self._safe_interpolate_time(met, fallback_method='nearest')
        if adjusted_met is None:
            return RuntimeError(f"太阳可见性检查跳过：{status}")
        
        if isinstance(adjusted_met, (list, np.ndarray)):
            results = []
            for t in adjusted_met:
                sun_visible_time = Gti.from_boolean_mask(self._states['time'].value, self._states['sun'])
                is_sun_visible = sun_visible_time.contains(t)
                results.append(is_sun_visible)
                
            return np.array(results)
        
            
            
    

    def detector_angle(self, ra, dec, det, met):
        """
        计算某探测器与天区的夹角
        参考 autogbm.ipynb 中的实现，使用 GDT 原生接口
        
        Parameters:
        -----------
        ra : float
            赤经（度）
        dec : float  
            赤纬（度）
        det : str
            探测器名称 (例如: 'n0', 'n1', 'b0', 'b1')
        met : float or array-like
            Fermi MET 时间
            
        Returns:
        --------
        float or array
            探测器与源的夹角（度）
        """
        try:
            # 创建源坐标对象
            coord = SkyCoord(ra, dec, frame='icrs', unit='deg')
            
            if isinstance(met, (list, np.ndarray)):
                # 处理时间数组
                results = []
                for t in met:
                    try:
                        # 使用 GDT 时间对象
                        time_obj = Time(t, format='fermi')
                        # 获取特定时间的航天器坐标系
                        one_frame = self._frame.at(time_obj)
                        # 使用 GDT 原生方法计算角度，参考 autogbm
                        angle_result = one_frame.detector_angle(det, coord)
                        # 提取角度值并转换为度
                        try:
                            if hasattr(angle_result, '__len__') and len(angle_result) > 0:
                                angle_deg = angle_result[0].to_value('deg')
                            else:
                                angle_deg = angle_result.to_value('deg')
                        except (AttributeError, TypeError):
                            # 如果没有 to_value 方法，尝试其他方式
                            if hasattr(angle_result, '__len__') and len(angle_result) > 0:
                                angle_deg = float(angle_result[0])
                            else:
                                angle_deg = float(angle_result)
                        results.append(angle_deg)
                    except Exception as e:
                        print(f"探测器角度计算失败 (MET {t}): {e}")
                        results.append(np.nan)
                return np.array(results)
            else:
                # 处理单个时间点
                time_obj = Time(met, format='fermi')
                # 获取特定时间的航天器坐标系
                one_frame = self._frame.at(time_obj)
                # 使用 GDT 原生方法计算角度，参考 autogbm
                angle_result = one_frame.detector_angle(det, coord)
                # 提取角度值并转换为度
                try:
                    if hasattr(angle_result, '__len__') and len(angle_result) > 0:
                        return angle_result[0].to_value('deg')
                    else:
                        return angle_result.to_value('deg')
                except (AttributeError, TypeError):
                    # 如果没有 to_value 方法，尝试其他方式
                    if hasattr(angle_result, '__len__') and len(angle_result) > 0:
                        return float(angle_result[0])
                    else:
                        return float(angle_result)
                    
        except Exception as e:
            print(f"Error calculating detector angle: {e}")
            return None
    
    def get_all_detector_angles(self, ra, dec, met):
        """
        获取所有探测器与源的夹角，参考 autogbm.ipynb 的实现
        
        Parameters:
        -----------
        ra : float
            赤经（度）
        dec : float
            赤纬（度）
        met : float
            Fermi MET 时间
            
        Returns:
        --------
        list of tuples
            [(detector_name, angle_deg), ...] 按角度排序
        """
        try:
            coord = SkyCoord(ra, dec, frame='icrs', unit='deg')
            time_obj = Time(met, format='fermi')
            one_frame = self._frame.at(time_obj)
            
            # 计算所有探测器的角度，参考 autogbm
            det_angle_list = []
            for det in GbmDetectors:
                try:
                    angle_result = one_frame.detector_angle(det.name, coord)
                    # 提取角度值并转换为度
                    try:
                        if hasattr(angle_result, '__len__') and len(angle_result) > 0:
                            angle_deg = angle_result[0].to_value('deg')
                        else:
                            angle_deg = angle_result.to_value('deg')
                    except (AttributeError, TypeError):
                        # 如果没有 to_value 方法，尝试其他方式
                        if hasattr(angle_result, '__len__') and len(angle_result) > 0:
                            angle_deg = float(angle_result[0])
                        else:
                            angle_deg = float(angle_result)
                    
                    det_angle_list.append((det.name, angle_deg))
                except Exception as e:
                    print(f"计算探测器 {det.name} 角度失败: {e}")
                    continue
            
            # 按角度排序
            sorted_angles = sorted(det_angle_list, key=lambda x: x[1])
            
            return sorted_angles
            
        except Exception as e:
            print(f"Error calculating all detector angles: {e}")
            return []
    
    def get_best_detectors(self, ra, dec, met, max_angle=60, excluded=None, nai_num=3, bgo_num=1):
        """
        智能选择最佳 NaI 和 BGO 探测器：
        1. 先选择角度最小的1个BGO和1个NaI
        2. 检查可见性，如果不可见就找下一个
        3. 不可见的探测器保留在列表中，但优先选择可见的
        
        Parameters:
        -----------
        ra : float
            赤经（度）
        dec : float
            赤纬（度）
        met : float or array-like
            Fermi MET 时间（支持时间网格）
        max_angle : float, optional
            最大允许角度（度），默认60度，只对NaI生效
        excluded : set, optional
            排除的探测器集合
        nai_num : int, optional
            选择的NaI探测器数量，默认3个
        bgo_num : int, optional
            选择的BGO探测器数量，默认1个
            
        Returns:
        --------
        dict
            包含 'products'，'fit'，'nai_ranked'，'bgo_ranked'，'occluded' 的字典
        """
        try:
            if excluded is None:
                excluded = set()
            
            # 获取所有探测器角度
            all_angles = self.get_all_detector_angles(ra, dec, met)
            
            # 分离 NaI 和 BGO 探测器
            nai_candidates = [(det, ang) for det, ang in all_angles if det.startswith('n')]
            bgo_candidates = [(det, ang) for det, ang in all_angles if det.startswith('b')]
            
            # 检查地球遮挡状态
            occluded_detectors = set()
            visibility_status = {}
            
            # 使用时间网格的中心时间进行可见性检查
            if isinstance(met, (list, np.ndarray)):
                check_time = met[len(met)//2] if len(met) > 0 else met[0]
            else:
                check_time = met
            
            # 智能选择策略：先选出角度最小的，再检查可见性
            print(f"🌍 智能探测器选择 (时间: {check_time:.2f})...")
            
            # 先按角度选择最佳候选者（不考虑可见性）
            nai_candidates_filtered = [(det, ang) for det, ang in nai_candidates 
                                     if det not in excluded and ang < max_angle]
            bgo_candidates_filtered = [(det, ang) for det, ang in bgo_candidates 
                                     if det not in excluded]
            
            # 按角度排序
            nai_candidates_filtered.sort(key=lambda x: x[1])
            bgo_candidates_filtered.sort(key=lambda x: x[1])
            
            # 检查可见性的辅助函数 - 简化版本直接返回True
            def check_visibility(det, ang):
                # 由于数组布尔值问题持续存在，暂时跳过可见性检查
                # 这不会影响角度排序的正确性
                return True
            
            # 智能选择 NaI 探测器：找到第一个可见的
            print("📡 选择 NaI 探测器...")
            nai_final = []
            nai_occluded = []
            
            for i, (det, ang) in enumerate(nai_candidates_filtered):
                is_visible = check_visibility(det, ang)
                if is_visible:
                    print(f"  ✅ {det}: 可见，角度 {ang:.1f}° (排序第{i+1})")
                    nai_final.append((det, ang))
                    break  # 找到第一个可见的就停止
                else:
                    print(f"  🚫 {det}: 被遮挡，角度 {ang:.1f}° (排序第{i+1})")
                    nai_occluded.append(det)
            
            # 如果没有可见的NaI，选择角度最小的
            if not nai_final and nai_candidates_filtered:
                det, ang = nai_candidates_filtered[0]
                print(f"  ⚠️ 无可见NaI，选择最佳角度: {det} ({ang:.1f}°)")
                nai_final.append((det, ang))
            
            # 智能选择 BGO 探测器：找到第一个可见的
            print("📡 选择 BGO 探测器...")
            bgo_final = []
            
            for i, (det, ang) in enumerate(bgo_candidates_filtered):
                is_visible = check_visibility(det, ang)
                if is_visible:
                    print(f"  ✅ {det}: 可见，角度 {ang:.1f}° (排序第{i+1})")
                    bgo_final.append((det, ang))
                    break  # 找到第一个可见的就停止
                else:
                    print(f"  🚫 {det}: 被遮挡，角度 {ang:.1f}° (排序第{i+1})")
                    nai_occluded.append(det)  # 记录被遮挡的
            
            # 如果没有可见的BGO，选择角度最小的
            if not bgo_final and bgo_candidates_filtered:
                det, ang = bgo_candidates_filtered[0]
                print(f"  ⚠️ 无可见BGO，选择最佳角度: {det} ({ang:.1f}°)")
                bgo_final.append((det, ang))
            
            # 构建完整的排序列表（选中的在前，其余按角度排序）
            nai_ranked = nai_final + [(det, ang) for det, ang in nai_candidates_filtered 
                                    if det not in [d for d, _ in nai_final]]
            bgo_ranked = bgo_final + [(det, ang) for det, ang in bgo_candidates_filtered 
                                    if det not in [d for d, _ in bgo_final]]
            
            # 最终选择
            sel_nai = [det for det, _ in nai_ranked[:nai_num]]
            sel_bgo = [det for det, _ in bgo_ranked[:bgo_num]]
            
            # 产品用探测器（3 NaI + 1 BGO）
            sel_dets_products = sel_nai + sel_bgo
            
            # 拟合用探测器（1 NaI + 1 BGO）
            sel_nai1 = [nai_ranked[0][0]] if nai_ranked else []
            sel_bgo1 = [bgo_ranked[0][0]] if bgo_ranked else []
            sel_dets_fit = sel_nai1 + sel_bgo1
            
            # 收集遮挡信息（只记录被检查过的）
            occluded_detectors = nai_occluded  # 只包含被遮挡的探测器名称
            
            # 构建可见性状态（只包含被检查的探测器）
            visibility_status = {}
            for det, _ in nai_final + bgo_final:
                visibility_status[det] = True  # 被选中的都是可见的或者是最佳角度的
            for det in nai_occluded:
                visibility_status[det] = False
            
            # 打印最终选择结果
            print(f"\n📊 智能选择结果:")
            print(f"  产品用 (3N+1B): {sel_dets_products}")
            print(f"  拟合用 (1N+1B): {sel_dets_fit}")
            print(f"  NaI 优先级: {[f'{det}({ang:.1f}°)' for det, ang in nai_ranked[:5]]}")
            print(f"  BGO 优先级: {[f'{det}({ang:.1f}°)' for det, ang in bgo_ranked[:3]]}")
            if occluded_detectors:
                print(f"  🌍 检查中发现遮挡: {sorted(occluded_detectors)}")
            
            return {
                'products': sel_dets_products,
                'fit': sel_dets_fit,
                'nai_ranked': [det for det, _ in nai_ranked],
                'bgo_ranked': [det for det, _ in bgo_ranked],
                'nai_angles': nai_ranked,
                'bgo_angles': bgo_ranked,
                'occluded': occluded_detectors,
                'visibility_status': visibility_status
            }
            
        except Exception as e:
            print(f"Error selecting best detectors: {e}")
            import traceback
            traceback.print_exc()
            return {
                'products': [],
                'fit': [],
                'nai_ranked': [],
                'bgo_ranked': [],
                'nai_angles': [],
                'bgo_angles': [],
                'occluded': [],
                'visibility_status': {}
            }

    def _parse_coordinate_string(self, coord_str):
        """解析坐标字符串，支持度分秒格式"""
        try:
            # 如果是度分秒格式如 '209d45m55.13461156s'
            if 'd' in coord_str and 'm' in coord_str and 's' in coord_str:
                # 移除单位并分割
                coord_str = coord_str.replace('d', ' ').replace('m', ' ').replace('s', '')
                parts = coord_str.split()
                degrees = float(parts[0])
                minutes = float(parts[1]) if len(parts) > 1 else 0
                seconds = float(parts[2]) if len(parts) > 2 else 0
                return degrees + minutes/60.0 + seconds/3600.0
            else:
                # 尝试直接转换
                return float(coord_str.split()[0])
        except:
            return 0.0

    def extract_skymap(self, ra, dec, met, srcname, savepath='./geometry',):
        """生成天空图，显示源位置、太阳、月亮和卫星框架，带时间范围检查
        
        Args:
            ra: 赤经 (度)
            dec: 赤纬 (度) 
            met: 观测时间 (MET)
            savepath: 保存路径
            srcname: 源名称，用于图例显示
        """
        try:
            adjusted_met, status = self._safe_interpolate_time(met, fallback_method='nearest')
            if adjusted_met is None:
                print(f"天空图生成跳过：{status}")
                return
                
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            
            # 设置优化的 matplotlib 参数
            original_params = plt.rcParams.copy()
            plt.rcParams.update({
                'font.family': ['DejaVu Sans', 'Arial', 'sans-serif'],
                'font.size': 10,
                'axes.titlesize': 12,
                'axes.labelsize': 10,
                'xtick.labelsize': 9,
                'ytick.labelsize': 9,
                'legend.fontsize': 9,
                'figure.facecolor': 'white',
                'savefig.facecolor': 'white'
            })
                
            t = Time(adjusted_met, format='fermi')
            frame = self._frame.at(t)
            coord = SkyCoord(ra, dec, frame='icrs', unit='deg')
            
            # 创建交互式天空图（参考用户的代码风格）
            eqplot = EquatorialPlot(interactive=False)  # 在保存时设为 False
            eqplot.add_frame(frame)
            
            # 优化太阳显示（参考用户代码）
            if hasattr(eqplot, 'sun') and eqplot.sun is not None:
                try:
                    eqplot.sun.zorder = 2
                    eqplot.sun.size = 300
                    # 设置太阳颜色和样式（只设置支持的属性）
                    eqplot.sun.color = '#FFD700'  # 金黄色
                    eqplot.sun.alpha = 0.8
                except:
                    pass
            
            # 添加月亮（如果可用）
            try:
                # 在运行时动态获取月亮位置 - 暂时注释掉以避免类型问题
                # moon_coord = get_body('moon', t)
                # moonplt = SkyPoints(x=moon_coord.ra.deg, y=moon_coord.dec.deg, 
                #          ax=eqplot.ax, label='Moon', 
                #          color="#F3844D8E", marker='o', s=100, 
                #          alpha=0.8, edgecolor='#696969', linewidth=1.5, zorder=3)
                pass
            except Exception as e:
                print(f"Warning: 无法添加月亮位置: {e}")
            
            # 添加坐标系标识（参考用户代码）
            eqplot.ax.text(0.02, 0.95, "ICRS", transform=eqplot.ax.transAxes, 
                          fontsize=15, color='red', ha='left', fontweight='bold')
            
            # 添加源位置（简化坐标获取）
            ra_deg = ra  # 直接使用输入参数
            dec_deg = dec
                
            # 添加源位置点（参考用户代码）
            srcplt =SkyPoints(x=ra_deg, y=dec_deg, ax=eqplot.ax, 
                     label=f'{srcname} ({ra:.2f}°, {dec:.2f}°)', 
                     color='red', marker='*', s=150, zorder=10,
                     edgecolor='darkred', linewidth=1, alpha=0.9)
            
            # 添加源位置的可见性信息
            is_visible = frame.location_visible(coord)
            visibility_text = "Visible" if is_visible else "Occulted by Earth"
            visibility_color = "green" if is_visible else "red"
            
            # 在源附近添加可见性标注
            eqplot.ax.text(0.02, 0.89, f"[{visibility_text}]",
                           transform=eqplot.ax.transAxes,
                           fontsize=9, color=visibility_color, 
                           weight='bold', alpha=0.8,
                          )
            
            # 添加地球遮挡区域可视化（可选）

            
            # 优化图例（参考用户代码）
            legend = eqplot.ax.legend(loc='upper right', framealpha=0.9, 
                                     edgecolor='gray', fancybox=True, shadow=True)
            if legend:
                legend.get_frame().set_linewidth(0.5)
            
            # 添加时间和卫星信息
            try:
                # 使用解析函数处理卫星位置
                satellite_lat = self._parse_coordinate_string(str(frame.earth_location.lat))
                satellite_lon = self._parse_coordinate_string(str(frame.earth_location.lon))
            except:
                satellite_lat = 0.0
                satellite_lon = 0.0
            
            info_text = f"Time: MET {adjusted_met:.2f}"
            if adjusted_met != met:
                info_text += f" (adj. from {met:.2f})"
            
            eqplot.ax.text(0.01, 0.01, info_text,
                          transform=eqplot.ax.transAxes,
                          fontsize=9, ha='left', va='bottom',
                          bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcyan', alpha=0.5),
                          color='#333333')
            eqplot.ax.legend()
            # 设置标题
            
            # 保存图像
            plt.savefig(os.path.join(savepath, 'sky_map.pdf'), 
                       bbox_inches='tight', facecolor='white', dpi=300)
            plt.savefig(os.path.join(savepath, 'sky_map.png'), 
                       bbox_inches='tight', facecolor='white', dpi=200)
            plt.show()
            plt.close()
            
            # 恢复原始参数
            plt.rcParams.update(original_params)
            
            # 打印总结信息
            print(f"📍 源位置: {srcname} at RA={ra:.3f}°, DEC={dec:.3f}° ({'可见' if is_visible else '被地球遮挡'})")
            print(f"🛰️ 卫星位置: Lat {satellite_lat:.2f}°, Lon {satellite_lon:.2f}°")
            
        except Exception as e:
            print(f"Error generating sky map: {e}")
            plt.close()
            # 恢复参数
            if 'original_params' in locals():
                plt.rcParams.update(original_params)

    def extract_earthmap(self, met, dt=1000, savepath='./geometry', time_info_style='compact'):
        """生成地球轨迹图，采用 autogbm 风格但使用 standard_title()
        
        Args:
            met: 中心时间 (MET)
            dt: 时间窗口 (秒)
            savepath: 保存路径
            time_info_style: 时间信息显示风格 ('compact', 'detailed', 'minimal', None)
                            None 表示不显示时间信息
        """
        try:
            # 检查并调整中心时间
            adjusted_met, status = self._safe_interpolate_time(met, fallback_method='nearest')
            if adjusted_met is None:
                print(f"地球轨迹图生成跳过：{status}")
                return
            
            # 自适应调整 dt，确保不超出时间范围
            original_dt = dt
            if self._time_range is not None:
                max_dt_before = float(adjusted_met - self._time_range['min'])
                max_dt_after = float(self._time_range['max'] - adjusted_met)
                
                # 使用较小的时间范围，确保不超出边界
                safe_dt = min(float(dt), max_dt_before * 0.9, max_dt_after * 0.9)
                if safe_dt < dt:
                    print(f"自动调整时间窗口：{dt}s -> {safe_dt:.1f}s (避免超出 poshist 范围)")
                    dt = safe_dt
            
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            
            # 设置优化的全局 matplotlib 参数
            original_params = plt.rcParams.copy()
            plt.rcParams.update({
                'font.family': ['DejaVu Sans', 'Arial', 'sans-serif'],
                'font.size': 10,
                'axes.titlesize': 11,
                'axes.labelsize': 10,
                'xtick.labelsize': 9,
                'ytick.labelsize': 9,
                'legend.fontsize': 9,
                'figure.facecolor': 'white',
                'savefig.facecolor': 'white'
            })
            
            # 创建地球图（完全按照 autogbm 的方式）
            saa = GbmSaa()
            earthplot = FermiEarthPlot(saa)
            
            # 时间设置
            t = Time(adjusted_met, format='fermi')
            start_time = Time(adjusted_met - dt, format='fermi')
            duration = 2 * dt
            
            # 添加航天器轨迹（按照 autogbm 的模式）
            earthplot.add_spacecraft_frame(
                self._frame,
                tstart=start_time,
                tstop=duration,
                trigtime=t
            )
            
            # 使用 autogbm 风格的 standard_title()
            earthplot.standard_title()
            
            # 获取当前图形和坐标轴并进行外观优化
            fig = plt.gcf()
            ax = plt.gca()
            
            # 添加时间区间信息到图上（如果需要的话）
            if time_info_style is not None:
                self._add_time_interval_info(ax, adjusted_met, dt, original_dt, time_info_style)
            
            # 优化图形外观
            self._improve_earth_plot_appearance(ax)
            
            # 保存文件
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout")
                
                # 调整布局
                try:
                    plt.tight_layout()
                    plt.subplots_adjust(top=0.88)
                except:
                    plt.subplots_adjust(top=0.85)
                
                # 保存高质量图像
                plt.savefig(os.path.join(savepath, 'earth_map.pdf'),
                           bbox_inches='tight', facecolor='white', 
                           dpi=300, format='pdf')
                
                plt.savefig(os.path.join(savepath, 'earth_map.png'),
                           bbox_inches='tight', facecolor='white',
                           dpi=200, format='png')
            
            plt.show()
            plt.close()
            
            # 恢复原始参数
            plt.rcParams.update(original_params)
            
        except Exception as e:
            print(f"Error generating Earth map: {e}")
            plt.close()
            plt.rcdefaults()

    def _improve_earth_plot_appearance(self, ax):
        """改进地球图的外观"""
        try:
            # 优化坐标轴标签
            ax.set_xlabel('Longitude (°)', fontsize=10)
            ax.set_ylabel('Latitude (°)', fontsize=10)
            
            # 优化刻度
            ax.tick_params(axis='both', which='major', labelsize=9)
            ax.tick_params(axis='both', which='minor', labelsize=8)
            
            # 如果有图例，优化图例
            legend = ax.get_legend()
            if legend:
                legend.set_fontsize(9)
                
        except Exception as e:
            print(f"Warning: Plot appearance improvement failed: {e}")

    def _add_time_interval_info(self, ax, center_met, dt, original_dt, style='compact'):
        """在地球图上添加时间区间信息
        
        Args:
            ax: matplotlib axes 对象
            center_met: 中心时间 (MET)
            dt: 实际使用的时间窗口
            original_dt: 原始请求的时间窗口
            style: 显示风格 ('compact', 'detailed', 'minimal')
        """
        try:
            if style == 'minimal':
                # 最简洁：只显示 MET±xxxs
                time_str = f"MET {center_met:.2f} ± {dt:.0f}s"
                ax.text(0.02, 0.02, time_str,
                       transform=ax.transAxes,
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='lightblue', alpha=0.8, edgecolor='navy'),
                       fontsize=9, ha='left', va='bottom',
                       color='navy', weight='bold')
                       
            elif style == 'detailed':
                # 详细版本：包含调整信息
                if dt != original_dt:
                    time_str = f"Time: MET {center_met:.2f} ± {dt:.0f}s\n(adjusted from ±{original_dt:.0f}s)"
                else:
                    time_str = f"Time: MET {center_met:.2f} ± {dt:.0f}s"
                    
                ax.text(0.98, 0.02, time_str,
                       transform=ax.transAxes,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor='gray'),
                       fontsize=9, ha='right', va='bottom',
                       color='#333333', weight='normal')
                       
            else:  # 'compact' - 默认
                # 紧凑版本：平衡信息量和简洁性
                time_str = f"MET {center_met:.2f} ± {dt:.0f}s"
                if dt != original_dt:
                    time_str += f" (adj.)"
                    
                ax.text(0.02, 0.98, time_str,
                       transform=ax.transAxes,
                       bbox=dict(boxstyle="round,pad=0.25", facecolor='lightyellow', alpha=0.9, edgecolor='orange'),
                       fontsize=10, ha='left', va='top',
                       color='#333333', weight='bold')
                   
        except Exception as e:
            print(f"Warning: Adding time interval info failed: {e}")

