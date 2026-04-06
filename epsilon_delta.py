import sys

# Pyodide では pyplot より前に WebAgg を明示（静的バックエンドだとウィジェットが「写真」になる）
if sys.platform == "emscripten":
    import matplotlib

    matplotlib.use("webagg")

import numpy as np
import matplotlib.pyplot as plt

if sys.platform == "emscripten":
    plt.ion()
from matplotlib.widgets import Slider, TextBox, Button
import matplotlib.patches as patches
from matplotlib.path import Path
import sympy as sp
import matplotlib.style as mplstyle
from matplotlib import rcParams
import matplotlib.ticker as ticker

class EpsilonDeltaVisualizer:
    def __init__(self, streamlit_mode: bool = False):
        self._streamlit_mode = streamlit_mode
        # Pyodide: 図をページ内のコンテナへ（未設定時は body 直下）
        if sys.platform == "emscripten":
            try:
                from js import document
                el = document.getElementById("pyodide-mpl-target")
                if el is not None:
                    document.pyodideMplTarget = el
            except Exception:
                pass
        # 現代的なスタイル設定
        self.setup_modern_style()
        
        self.fig, self.ax = plt.subplots(figsize=(16, 10))
        plt.subplots_adjust(left=0.06, bottom=0.1, right=0.58, top=0.95)
        
        # 初期値の設定
        self.a = 1.5
        self.epsilon = 0.8
        self.delta = 0.5
        self.b = 0.0  # bの初期値を追加
        self.function_expr = 'x**2'
        
        # 初期値を保存
        self.initial_a = self.a
        self.initial_epsilon = self.epsilon
        self.initial_delta = self.delta
        self.initial_b = self.b
        self.initial_function_expr = self.function_expr
        
        # 拡大縮小用の初期範囲を保存
        self.initial_xlim = (-1, 5)
        self.initial_ylim = (-1, 5)
        
        # グラフの設定
        self.ax.set_xlim(-1, 5)
        self.ax.set_ylim(-1, 5)  # y軸の範囲を-1から5に変更
        
        # x軸とy軸を確実に表示するため、範囲を調整
        self.ax.set_xlim(-1, 5)
        self.ax.set_ylim(-1, 5)
        
        # シンプルなグリッドと軸の設定
        self.ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color=self.colors['grid'], zorder=1)
        self.ax.set_facecolor(self.colors['background'])
        # マス目を正方形にする（データ座標で縦横1単位が同じ長さ）
        self.ax.set_aspect("equal", adjustable="box")
        
        # メイン軸の強調
        self.ax.axhline(y=0, color=self.colors['primary'], linestyle='-', alpha=0.8, linewidth=2, zorder=5)
        self.ax.axvline(x=0, color=self.colors['primary'], linestyle='-', alpha=0.8, linewidth=2, zorder=5)
        
        # 軸のスタイリング（シンプル）
        # グラフの境界線を表示
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_color(self.colors['primary'])
        self.ax.spines['left'].set_linewidth(1.5)
        self.ax.spines['bottom'].set_color(self.colors['primary'])
        self.ax.spines['bottom'].set_linewidth(1.5)
        # x軸とy軸に目盛りを表示（初期設定、draw_axesで動的に更新）
        self.ax.tick_params(colors=self.colors['text'], labelsize=10, width=1, length=5, 
                           bottom=True, top=False, left=True, right=False,
                           labelbottom=False, labelleft=False)
        # 目盛りの間隔を設定（初期値は1刻み、draw_axesで動的に更新）
        self.ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        self.ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        # 目盛りのフォーマッターを設定（整数表示）
        self.ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        self.ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        
        # 関数の描画（x軸の範囲を確実にカバー）
        # 表示範囲に応じて動的にxデータを生成するため、初期値のみ設定
        self.x = np.linspace(-1, 5, 1000)
        
        # x軸とy軸の線を保持するための参照
        self.x_axis_line = None
        self.y_axis_line = None
        # x軸とy軸上のカスタムラベルを保持するための参照
        self.x_axis_labels = []
        self.y_axis_labels = []
        # x軸とy軸上の目盛り線を保持するための参照
        self.x_axis_ticks = []
        self.y_axis_ticks = []
        
        # f(x)を描画（x <= aの部分とx > aの部分を別々に）
        # x <= aの部分（f(x)）- aの点を含める
        x_f = self.x[self.x <= self.a]
        if len(x_f) > 0 and x_f[-1] < self.a:
            # aの点を確実に含める
            x_f = np.append(x_f, self.a)
        y_f = self.evaluate_f(x_f)
        self.line_f, = self.ax.plot(x_f, y_f, color=self.colors['primary'], linewidth=1.5, 
                                   label='f(x)', alpha=0.9, solid_capstyle='round',
                                   zorder=8)
        
        # x >= aの部分（f(x)+b）- aの点を含める
        x_f_plus_b = self.x[self.x >= self.a]
        if len(x_f_plus_b) > 0 and x_f_plus_b[0] > self.a:
            # aの点を確実に含める
            x_f_plus_b = np.insert(x_f_plus_b, 0, self.a)
        y_f_plus_b = self.evaluate_f_plus_b(x_f_plus_b)
        self.line_f_plus_b, = self.ax.plot(x_f_plus_b, y_f_plus_b, color=self.colors['primary'], 
                                          linewidth=1.5, label='f(x)+b', alpha=0.9, 
                                          solid_capstyle='round', zorder=8)
        
        # 全体の関数（後方互換性のため）
        self.y = self.evaluate_function(self.x)
        self.line, = self.ax.plot(self.x, self.y, 'k-', alpha=0.0, label='全体')  # 透明度を0にして非表示
        
        # コントロール（Streamlit 版は外部ウィジェットのため図のみ）
        if self._streamlit_mode:
            plt.subplots_adjust(left=0.06, bottom=0.1, right=0.95, top=0.95)
        else:
            self.add_controls()
        
        # 初期描画
        self.update(None)
        
        # 起動時にx軸とy軸を確実に表示
        self.ax.set_xlim(-1, 5)
        self.ax.set_ylim(-1, 5)
        self.ax.axhline(y=0, color=self.colors['primary'], linestyle='-', alpha=0.8, linewidth=2, zorder=5)
        self.ax.axvline(x=0, color=self.colors['primary'], linestyle='-', alpha=0.8, linewidth=2, zorder=5)
    
    def setup_modern_style(self):
        """シンプルで洗練されたスタイルを設定"""
        # ブラウザ Canvas には OS フォントが無いため、存在する DejaVu のみ（日本語は □ になるのを避ける）
        if sys.platform == "emscripten":
            rcParams["font.family"] = "sans-serif"
            rcParams["font.sans-serif"] = ["DejaVu Sans"]
        else:
            rcParams["font.family"] = [
                "DejaVu Sans",
                "Hiragino Sans",
                "Yu Gothic",
                "Meiryo",
                "Takao",
                "IPAexGothic",
                "IPAPGothic",
                "VL PGothic",
                "Noto Sans CJK JP",
            ]
        rcParams['font.size'] = 11
        rcParams['font.weight'] = 'normal'
        rcParams['axes.titlesize'] = 15
        rcParams['axes.labelsize'] = 12
        rcParams['xtick.labelsize'] = 10
        rcParams['ytick.labelsize'] = 10
        rcParams['legend.fontsize'] = 10
        
        # シンプルで洗練されたカラーパレット
        self.colors = {
            'primary': '#2C3E50',      # ダークグレー（メイン）
            'secondary': '#7F8C8D',    # ミディアムグレー
            'accent': '#E74C3C',       # アクセント赤（ε用）
            'accent2': '#3498DB',      # アクセント青（δ用）
            'success': '#27AE60',      # 緑（拡大用）
            'warning': '#F39C12',      # オレンジ（b用）
            'background': '#FFFFFF',    # 白背景
            'background2': '#F8F9FA',   # ライトグレー背景
            'grid': '#E0E0E0',         # グリッド
            'text': '#34495E',         # テキスト
            'border': '#BDC3C7',        # ボーダー
            'glass_white': (1.0, 1.0, 1.0, 0.95),  # RGBA形式
            'shadow': (0.0, 0.0, 0.0, 0.1),
            'shadow_light': (0.0, 0.0, 0.0, 0.05)
        }
    
    def setup_zoom_pan(self):
        """パン機能を設定（拡大縮小はボタンのみ）"""
        # マウスドラッグでのパンのみ有効
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        
        # 初期状態
        self.pan_start = None

    def _refresh_data_after_pan_or_zoom(self):
        """表示範囲変更後に x サンプルと曲線・軸を揃える（パン終了・プログラム用パンと共通）"""
        self.update_x_data()
        x_f = self.x[self.x <= self.a]
        if len(x_f) > 0 and x_f[-1] < self.a:
            x_f = np.append(x_f, self.a)
        y_f = self.evaluate_f(x_f)
        self.line_f.set_xdata(x_f)
        self.line_f.set_ydata(y_f)
        x_f_plus_b = self.x[self.x >= self.a]
        if len(x_f_plus_b) > 0 and x_f_plus_b[0] > self.a:
            x_f_plus_b = np.insert(x_f_plus_b, 0, self.a)
        y_f_plus_b = self.evaluate_f_plus_b(x_f_plus_b)
        self.line_f_plus_b.set_xdata(x_f_plus_b)
        self.line_f_plus_b.set_ydata(y_f_plus_b)
        self.y = self.evaluate_function(self.x)
        self.line.set_xdata(self.x)
        self.line.set_ydata(self.y)
        self.draw_axes()
        self.fig.canvas.draw_idle()

    def pan_by_data(self, dx: float, dy: float) -> None:
        """データ座標で表示範囲を平行移動（Streamlit 等・マウスドラッグの代わり）"""
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
        self.ax.set_ylim(ylim[0] - dy, ylim[1] - dy)
        self._refresh_data_after_pan_or_zoom()

    def zoom_in(self, event):
        """拡大ボタン"""
        # 拡大縮小の中心点を(a, f(a))に設定
        f_a = self.evaluate_function(np.array([self.a]))[0]
        # f_aがNaNの場合は0を使用
        if np.isnan(f_a):
            f_a = 0.0
        x_center = self.a
        y_center = f_a
        
        # 拡大の倍率
        zoom_factor = 1.2
        
        # 現在の範囲を取得
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        # 新しい範囲を計算
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        
        new_x_range = x_range / zoom_factor
        new_y_range = y_range / zoom_factor
        
        # 中心点を基準に範囲を調整
        new_xlim = [x_center - new_x_range/2, x_center + new_x_range/2]
        new_ylim = [y_center - new_y_range/2, y_center + new_y_range/2]
        
        # 範囲を設定
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        
        # 表示範囲に応じてxデータを更新
        self.update_x_data()
        
        # 関数の線を更新
        x_f = self.x[self.x <= self.a]
        if len(x_f) > 0 and x_f[-1] < self.a:
            x_f = np.append(x_f, self.a)
        y_f = self.evaluate_f(x_f)
        self.line_f.set_xdata(x_f)
        self.line_f.set_ydata(y_f)
        
        x_f_plus_b = self.x[self.x >= self.a]
        if len(x_f_plus_b) > 0 and x_f_plus_b[0] > self.a:
            x_f_plus_b = np.insert(x_f_plus_b, 0, self.a)
        y_f_plus_b = self.evaluate_f_plus_b(x_f_plus_b)
        self.line_f_plus_b.set_xdata(x_f_plus_b)
        self.line_f_plus_b.set_ydata(y_f_plus_b)
        
        self.y = self.evaluate_function(self.x)
        self.line.set_xdata(self.x)
        self.line.set_ydata(self.y)
        
        self.ax.set_aspect("equal", adjustable="box")
        self.draw_axes()
        self.fig.canvas.draw_idle()

    def zoom_out(self, event):
        """縮小ボタン"""
        # 拡大縮小の中心点を(a, f(a))に設定
        f_a = self.evaluate_function(np.array([self.a]))[0]
        # f_aがNaNの場合は0を使用
        if np.isnan(f_a):
            f_a = 0.0
        x_center = self.a
        y_center = f_a
        
        # 縮小の倍率
        zoom_factor = 1/1.2
        
        # 現在の範囲を取得
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        # 新しい範囲を計算
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        
        new_x_range = x_range / zoom_factor
        new_y_range = y_range / zoom_factor
        
        # 中心点を基準に範囲を調整
        new_xlim = [x_center - new_x_range/2, x_center + new_x_range/2]
        new_ylim = [y_center - new_y_range/2, y_center + new_y_range/2]
        
        # 範囲を設定
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        
        # 表示範囲に応じてxデータを更新
        self.update_x_data()
        
        # 関数の線を更新
        x_f = self.x[self.x <= self.a]
        if len(x_f) > 0 and x_f[-1] < self.a:
            x_f = np.append(x_f, self.a)
        y_f = self.evaluate_f(x_f)
        self.line_f.set_xdata(x_f)
        self.line_f.set_ydata(y_f)
        
        x_f_plus_b = self.x[self.x >= self.a]
        if len(x_f_plus_b) > 0 and x_f_plus_b[0] > self.a:
            x_f_plus_b = np.insert(x_f_plus_b, 0, self.a)
        y_f_plus_b = self.evaluate_f_plus_b(x_f_plus_b)
        self.line_f_plus_b.set_xdata(x_f_plus_b)
        self.line_f_plus_b.set_ydata(y_f_plus_b)
        
        self.y = self.evaluate_function(self.x)
        self.line.set_xdata(self.x)
        self.line.set_ydata(self.y)
        
        self.ax.set_aspect("equal", adjustable="box")
        self.draw_axes()
        self.fig.canvas.draw_idle()

    def on_press(self, event):
        """マウスボタン押下"""
        if event.inaxes != self.ax:
            return
        if event.button == 1:  # 左クリック
            self.pan_start = (event.xdata, event.ydata)
            
    def on_motion(self, event):
        """マウス移動（ドラッグ中）"""
        if self.pan_start is None or event.inaxes != self.ax:
            return
            
        if event.xdata is None or event.ydata is None:
            return
            
        # パンの距離を計算
        dx = event.xdata - self.pan_start[0]
        dy = event.ydata - self.pan_start[1]
        
        # 現在の範囲を取得
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        # 新しい範囲を設定
        new_xlim = [xlim[0] - dx, xlim[1] - dx]
        new_ylim = [ylim[0] - dy, ylim[1] - dy]
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        
        # 表示範囲に応じてxデータを更新（パン中は頻繁に更新しない）
        # パン終了時に更新するため、ここではスキップ
        
        # グラフを更新
        self.fig.canvas.draw_idle()
        
    def on_release(self, event):
        """マウスボタン離す"""
        if self.pan_start is not None:
            self._refresh_data_after_pan_or_zoom()
        self.pan_start = None
        
    def evaluate_function(self, x):
        """関数を評価する（f(x) = {f(x) (x≤a), f(x)+b (x>a)}）"""
        try:
            # sympyを使用して関数を評価
            x_sym = sp.Symbol('x')
            expr = sp.sympify(self.function_expr)
            f = sp.lambdify(x_sym, expr, 'numpy')
            result = f(x)
            
            # スカラーの場合は配列に変換
            if np.isscalar(result):
                result = np.array([result])
            else:
                result = np.asarray(result)
            
            # xも配列に変換（スカラーの場合）
            x = np.asarray(x)
            
            # x=aでグラフを分断し、x <= aの場合はf(x)、x > aの場合はf(x) + b
            mask = x > self.a
            if result.ndim == 0:
                result = result.reshape(1)
            if mask.ndim == 0:
                mask = mask.reshape(1)
            
            # サイズが一致することを確認
            if result.shape != x.shape:
                # 形状が一致しない場合は、ブロードキャスト可能か確認
                try:
                    result = np.broadcast_to(result, x.shape)
                except:
                    # ブロードキャストできない場合は、ゼロ配列を返す
                    return np.zeros_like(x)
            
            # bが0に近い場合は0として扱う（数値精度の問題を回避）
            b_value = 0.0 if abs(self.b) < 1e-12 else self.b
            result[mask] += b_value  # f(x) + b for x > a
            
            return result
        except Exception as e:
            # エラーが発生した場合は、ゼロ配列を返す
            x = np.asarray(x)
            return np.zeros_like(x)
    
    def evaluate_f(self, x):
        """f(x)を評価する（x <= aの部分）"""
        try:
            x_sym = sp.Symbol('x')
            expr = sp.sympify(self.function_expr)
            f = sp.lambdify(x_sym, expr, 'numpy')
            return f(x)
        except:
            return np.zeros_like(x)
    
    def get_polygon_center(self, vertices):
        """多角形の頂点リストから中心座標を計算"""
        if not vertices or len(vertices) < 3:
            return None, None
        x_coords = [v[0] for v in vertices]
        y_coords = [v[1] for v in vertices]
        # NaNを除外
        x_coords = [x for x in x_coords if not np.isnan(x)]
        y_coords = [y for y in y_coords if not np.isnan(y)]
        if not x_coords or not y_coords:
            return None, None
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        return center_x, center_y
    
    def add_region_label(self, vertices, label, color='black', fontsize=8, alpha=0.8, 
                         x_offset=0, y_offset=0, position='center'):
        """領域の中心にラベルを追加
        
        position: 'center', 'left', 'right', 'top', 'bottom', 'top_left', 'top_right', 'bottom_left', 'bottom_right'
        """
        if not vertices or len(vertices) < 3:
            return
            
        x_coords = [v[0] for v in vertices if not np.isnan(v[0])]
        y_coords = [v[1] for v in vertices if not np.isnan(v[1])]
        
        if not x_coords or not y_coords:
            return
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        # 位置に応じて座標を調整
        if position == 'left':
            center_x = x_min + (x_max - x_min) * 0.25
        elif position == 'right':
            center_x = x_min + (x_max - x_min) * 0.75
        elif position == 'top':
            center_y = y_min + (y_max - y_min) * 0.75
        elif position == 'bottom':
            center_y = y_min + (y_max - y_min) * 0.25
        elif position == 'top_left':
            center_x = x_min + (x_max - x_min) * 0.25
            center_y = y_min + (y_max - y_min) * 0.75
        elif position == 'top_right':
            center_x = x_min + (x_max - x_min) * 0.75
            center_y = y_min + (y_max - y_min) * 0.75
        elif position == 'bottom_left':
            center_x = x_min + (x_max - x_min) * 0.25
            center_y = y_min + (y_max - y_min) * 0.25
        elif position == 'bottom_right':
            center_x = x_min + (x_max - x_min) * 0.75
            center_y = y_min + (y_max - y_min) * 0.25
        
        # オフセットを適用
        center_x += x_offset
        center_y += y_offset
        
        self.ax.text(center_x, center_y, label, 
                    fontsize=fontsize, fontweight='bold', 
                    color=color, ha='center', va='center',
                    alpha=alpha, zorder=25,
                    bbox=dict(boxstyle='round,pad=0.15', 
                            facecolor='white', 
                            edgecolor='none', 
                            alpha=0.7))
    
    def is_increasing_at_a(self):
        """点aの近傍で関数が増加しているかを判定"""
        # aの近傍での導関数の符号を調べる
        h = 0.001  # 微小な増分
        f_a = self.evaluate_f(np.array([self.a]))[0]
        f_a_plus_h = self.evaluate_f(np.array([self.a + h]))[0]
        f_a_minus_h = self.evaluate_f(np.array([self.a - h]))[0]
        
        # NaN処理
        if np.isnan(f_a) or np.isnan(f_a_plus_h) or np.isnan(f_a_minus_h):
            return True  # デフォルトは増加
        
        # 増加関数か減少関数かを判定
        # 両側の傾きを調べる
        slope_right = (f_a_plus_h - f_a) / h
        slope_left = (f_a - f_a_minus_h) / h
        
        # 平均傾きで判定
        avg_slope = (slope_right + slope_left) / 2
        return avg_slope >= 0
    
    def evaluate_f_plus_b(self, x):
        """f(x)+bを評価する（x >= aの部分）"""
        try:
            x_sym = sp.Symbol('x')
            expr = sp.sympify(self.function_expr)
            f = sp.lambdify(x_sym, expr, 'numpy')
            result = f(x)
            # bが0に近い場合は0として扱う（数値精度の問題を回避）
            b_value = 0.0 if abs(self.b) < 1e-12 else self.b
            
            # xを配列に変換（スカラーの場合）
            x_array = np.asarray(x)
            result_array = np.asarray(result)
            
            # 形状を統一
            if result_array.ndim == 0:
                result_array = result_array.reshape(1)
            if x_array.ndim == 0:
                x_array = x_array.reshape(1)
            
            # ブロードキャスト可能な場合
            if result_array.shape != x_array.shape:
                try:
                    result_array = np.broadcast_to(result_array, x_array.shape)
                except:
                    pass
            
            # x >= aのすべての点にbを加算
            # ただし、b=0の場合は、x=aの点ではbを加算しない（f(a)とf(a)+bを一致させるため）
            if result_array.shape == x_array.shape:
                if abs(b_value) < 1e-12:
                    # b=0の場合、x > aの点のみbを加算（x=aの点では加算しない）
                    mask_gt_a = x_array > self.a
                    result_array = result_array.copy()
                    result_array[mask_gt_a] += b_value
                else:
                    # b≠0の場合、x >= aのすべての点にbを加算
                    mask_ge_a = x_array >= self.a
                    result_array = result_array.copy()
                    result_array[mask_ge_a] += b_value
            else:
                # 形状が一致しない場合は、すべての点にbを加算
                result_array = result_array + b_value
            
            # スカラーの場合はスカラーに戻す
            if np.isscalar(x) and result_array.size == 1:
                return result_array[0]
            return result_array
        except:
            return np.zeros_like(x)
    
    def update_x_data(self):
        """現在の表示範囲に応じてxデータを動的に更新"""
        xlim = self.ax.get_xlim()
        # 表示範囲の1.5倍の範囲でxデータを生成（余裕を持たせる）
        margin = (xlim[1] - xlim[0]) * 0.5
        x_min = xlim[0] - margin
        x_max = xlim[1] + margin
        # 解像度を維持（範囲に応じて点数を調整）
        num_points = max(1000, int((x_max - x_min) * 200))
        new_x = np.linspace(x_min, x_max, num_points)
        # サイズが変わった場合のみ更新（パフォーマンス向上）
        if len(self.x) != len(new_x) or not np.allclose(self.x, new_x, rtol=1e-10):
            self.x = new_x
    
    def draw_axes(self):
        """x軸とy軸を確実に描画"""
        # 既存の軸線を削除
        if self.x_axis_line is not None:
            try:
                self.x_axis_line.remove()
            except:
                pass
        if self.y_axis_line is not None:
            try:
                self.y_axis_line.remove()
            except:
                pass
        
        # 既存のカスタムラベルを削除（x軸とy軸上のラベル）
        if hasattr(self, 'x_axis_labels'):
            for label in self.x_axis_labels:
                try:
                    label.remove()
                except:
                    pass
        if hasattr(self, 'y_axis_labels'):
            for label in self.y_axis_labels:
                try:
                    label.remove()
                except:
                    pass
        # 既存の目盛り線を削除
        if hasattr(self, 'x_axis_ticks'):
            for tick_line in self.x_axis_ticks:
                try:
                    tick_line.remove()
                except:
                    pass
        if hasattr(self, 'y_axis_ticks'):
            for tick_line in self.y_axis_ticks:
                try:
                    tick_line.remove()
                except:
                    pass
        
        # x軸とy軸を再描画
        ylim = self.ax.get_ylim()
        xlim = self.ax.get_xlim()
        
        # 目盛りの基本設定
        # x軸（y=0）とy軸（x=0）にメモリ（目盛り）を表示
        self.ax.tick_params(colors=self.colors['text'], labelsize=10, width=1, length=5, 
                           bottom=True, top=False, left=True, right=False,
                           labelbottom=False, labelleft=False)
        
        # 目盛りの間隔を動的に設定（拡大時に適切な間隔で表示）
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        
        # 表示範囲に応じて目盛りの間隔を決定
        # 範囲が小さい場合は細かく、大きい場合は粗く（0.01刻みまで対応）
        if x_range <= 0.1:
            x_tick_interval = 0.01
        elif x_range <= 0.5:
            x_tick_interval = 0.05
        elif x_range <= 1:
            x_tick_interval = 0.1
        elif x_range <= 5:
            x_tick_interval = 0.5
        elif x_range <= 10:
            x_tick_interval = 1
        elif x_range <= 20:
            x_tick_interval = 2
        else:
            x_tick_interval = max(1, int(x_range / 10))
        
        if y_range <= 0.1:
            y_tick_interval = 0.01
        elif y_range <= 0.5:
            y_tick_interval = 0.05
        elif y_range <= 1:
            y_tick_interval = 0.1
        elif y_range <= 5:
            y_tick_interval = 0.5
        elif y_range <= 10:
            y_tick_interval = 1
        elif y_range <= 20:
            y_tick_interval = 2
        else:
            y_tick_interval = max(1, int(y_range / 10))
        
        # 目盛りの間隔を設定
        self.ax.xaxis.set_major_locator(ticker.MultipleLocator(x_tick_interval))
        self.ax.yaxis.set_major_locator(ticker.MultipleLocator(y_tick_interval))
        # 目盛りのフォーマッターを設定（適切な精度で表示、0.01刻みまで対応）
        if x_tick_interval < 0.1:
            self.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        elif x_tick_interval < 1:
            self.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        else:
            self.ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        if y_tick_interval < 0.1:
            self.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        elif y_tick_interval < 1:
            self.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        else:
            self.ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        
        # x軸（y=0）が表示範囲内にある場合のみ描画
        self.x_axis_labels = []
        self.x_axis_ticks = []  # x軸上の目盛り線を保持
        x_axis_visible = ylim[0] <= 0 <= ylim[1]
        if x_axis_visible:
            self.x_axis_line = self.ax.axhline(y=0, color='#1A1A1A', linestyle='-', 
                                             alpha=0.9, linewidth=2.5, zorder=5)
            # 目盛り位置を取得
            x_ticks = self.ax.xaxis.get_majorticklocs()
            # x軸（y=0）上に目盛り（メモリ）を表示
            for tick_x in x_ticks:
                if xlim[0] <= tick_x <= xlim[1]:
                    # x軸上の目盛り線を描画（短い縦線）
                    tick_line = self.ax.plot([tick_x, tick_x], [0, (ylim[1] - ylim[0]) * 0.01], 
                                           color='#1A1A1A', linewidth=1.5, zorder=6)
                    self.x_axis_ticks.append(tick_line[0])
            # x軸（y=0）の下に数値を表示
            # 表示範囲内の目盛りのみ表示
            for tick_x in x_ticks:
                if xlim[0] <= tick_x <= xlim[1]:
                    # x軸（y=0）の少し下にラベルを配置
                    # y座標は0より少し下（データ座標で約0.05下）
                    label_y = 0 - (ylim[1] - ylim[0]) * 0.02
                    # 数値のフォーマット（0.01刻みまで対応）
                    if x_tick_interval < 0.1:
                        label_text = f'{tick_x:.2f}'
                    elif x_tick_interval < 1:
                        label_text = f'{tick_x:.1f}'
                    elif x_tick_interval < 2:
                        label_text = f'{tick_x:.1f}' if tick_x != int(tick_x) else f'{int(tick_x)}'
                    else:
                        label_text = f'{int(tick_x)}'
                    label = self.ax.text(tick_x, label_y, label_text, 
                                       ha='center', va='top',
                                       fontsize=10, color=self.colors['text'],
                                       zorder=15)
                    self.x_axis_labels.append(label)
        else:
            # x軸が表示範囲外の場合、グラフの下端に数字を表示
            self.x_axis_line = None
            x_ticks = self.ax.xaxis.get_majorticklocs()
            for tick_x in x_ticks:
                if xlim[0] <= tick_x <= xlim[1]:
                    # グラフの下端（ylim[0]）にラベルを配置
                    label_y = ylim[0]
                    # 数値のフォーマット（0.01刻みまで対応）
                    if x_tick_interval < 0.1:
                        label_text = f'{tick_x:.2f}'
                    elif x_tick_interval < 1:
                        label_text = f'{tick_x:.1f}'
                    elif x_tick_interval < 2:
                        label_text = f'{tick_x:.1f}' if tick_x != int(tick_x) else f'{int(tick_x)}'
                    else:
                        label_text = f'{int(tick_x)}'
                    label = self.ax.text(tick_x, label_y, label_text, 
                                       ha='center', va='top',
                                       fontsize=10, color=self.colors['text'],
                                       zorder=15)
                    self.x_axis_labels.append(label)
        
        # y軸（x=0）が表示範囲内にある場合のみ描画
        self.y_axis_labels = []
        self.y_axis_ticks = []  # y軸上の目盛り線を保持
        y_axis_visible = xlim[0] <= 0 <= xlim[1]
        if y_axis_visible:
            self.y_axis_line = self.ax.axvline(x=0, color='#1A1A1A', linestyle='-', 
                                              alpha=0.9, linewidth=2.5, zorder=5)
            # y軸（x=0）上に目盛り（メモリ）を表示
            y_ticks = self.ax.yaxis.get_majorticklocs()
            for tick_y in y_ticks:
                if ylim[0] <= tick_y <= ylim[1]:
                    # y軸上の目盛り線を描画（短い横線）
                    tick_line = self.ax.plot([0, (xlim[1] - xlim[0]) * 0.01], [tick_y, tick_y], 
                                           color='#1A1A1A', linewidth=1.5, zorder=6)
                    self.y_axis_ticks.append(tick_line[0])
            # y軸（x=0）の左に数値を表示
            # 目盛り位置を取得
            y_ticks = self.ax.yaxis.get_majorticklocs()
            # 表示範囲内の目盛りのみ表示
            for tick_y in y_ticks:
                if ylim[0] <= tick_y <= ylim[1]:
                    # y軸（x=0）の少し左にラベルを配置
                    # x座標は0より少し左（データ座標で約0.05左）
                    label_x = 0 - (xlim[1] - xlim[0]) * 0.02
                    # 数値のフォーマット（0.01刻みまで対応）
                    if y_tick_interval < 0.1:
                        label_text = f'{tick_y:.2f}'
                    elif y_tick_interval < 1:
                        label_text = f'{tick_y:.1f}'
                    elif y_tick_interval < 2:
                        label_text = f'{tick_y:.1f}' if tick_y != int(tick_y) else f'{int(tick_y)}'
                    else:
                        label_text = f'{int(tick_y)}'
                    label = self.ax.text(label_x, tick_y, label_text, 
                                       ha='right', va='center',
                                       fontsize=10, color=self.colors['text'],
                                       zorder=15)
                    self.y_axis_labels.append(label)
        else:
            # y軸が表示範囲外の場合、グラフの左端に数字を表示
            self.y_axis_line = None
            y_ticks = self.ax.yaxis.get_majorticklocs()
            for tick_y in y_ticks:
                if ylim[0] <= tick_y <= ylim[1]:
                    # グラフの左端（xlim[0]）にラベルを配置
                    label_x = xlim[0]
                    # 数値のフォーマット（0.01刻みまで対応）
                    if y_tick_interval < 0.1:
                        label_text = f'{tick_y:.2f}'
                    elif y_tick_interval < 1:
                        label_text = f'{tick_y:.1f}'
                    elif y_tick_interval < 2:
                        label_text = f'{tick_y:.1f}' if tick_y != int(tick_y) else f'{int(tick_y)}'
                    else:
                        label_text = f'{int(tick_y)}'
                    label = self.ax.text(label_x, tick_y, label_text, 
                                       ha='right', va='center',
                                       fontsize=10, color=self.colors['text'],
                                       zorder=15)
                    self.y_axis_labels.append(label)
    
    def find_intersections(self, target_y, min_x=-3, max_x=3, jump_exclude=False):
        """関数と水平線の交点を求める（jump_exclude=Trueなら不連続点x=aをまたぐ交点は除外）"""
        # 高精度で交点を求めるため、より細かいグリッドを使用
        x = np.linspace(min_x, max_x, 5000)  # 1000から5000に増加
        y = self.evaluate_function(x)
        
        # NaN値を処理（sqrt(x)などでx<0の場合）
        valid_mask = ~np.isnan(y)
        if not np.any(valid_mask):
            return []
        
        # 数値精度を向上させるため、より厳密な条件で交点を検出
        diff = y - target_y
        # NaN値がある箇所では符号変化を検出しない
        diff_sign = np.sign(diff)
        diff_sign[np.isnan(diff)] = 0  # NaN箇所は0に設定
        crossings = np.where(np.diff(diff_sign) != 0)[0]
        intersections = []
        
        for i in crossings:
            if i < len(x) - 1:
                x1, x2 = x[i], x[i+1]
                y1, y2 = y[i], y[i+1]
                
                # NaN値がある区間はスキップ
                if np.isnan(y1) or np.isnan(y2):
                    continue
                
                # x=aをまたぐ区間は除外（jump_exclude=Trueのときのみ）
                if jump_exclude and ((x1 < self.a < x2) or (x2 < self.a < x1)):
                    continue
                
                # より高精度な線形補間
                if abs(y2 - y1) > 1e-10:  # ゼロ除算を避ける
                    t = (target_y - y1) / (y2 - y1)
                    x_intersect = x1 + t * (x2 - x1)
                    intersections.append((x_intersect, target_y))
        
        return intersections
    
    def reset_to_initial(self, event=None):
        """初期状態に戻す"""
        if self._streamlit_mode:
            self.a = self.initial_a
            self.epsilon = self.initial_epsilon
            self.delta = self.initial_delta
            self.b = self.initial_b
            self.function_expr = self.initial_function_expr
        else:
            self.x_slider.set_val(self.initial_a)
            self.eps_slider.set_val(self.initial_epsilon)
            self.delta_slider.set_val(self.initial_delta)
            self.b_slider.set_val(self.initial_b)

            self.a_input.set_val(f"{self.initial_a:.2f}")
            self.eps_input.set_val(f"{self.initial_epsilon:.3f}")
            self.delta_input.set_val(f"{self.initial_delta:.3f}")
            self.b_input.set_val(f"{self.initial_b:.2f}")

            self.function_expr = self.initial_function_expr
            self.func_text.set_val(self.initial_function_expr)

        self.ax.set_xlim(self.initial_xlim)
        self.ax.set_ylim(self.initial_ylim)

        self.update_x_data()
        self.update(None)
        self.draw_axes()
    
    def add_controls(self):
        """シンプルで洗練されたコントロールパネルを追加"""
        _w = sys.platform == "emscripten"
        t_zoom_in, t_zoom_out, t_reset = (
            ("Zoom in", "Zoom out", "Reset") if _w else ("拡大", "縮小", "リセット")
        )
        t_negate = "+/-" if _w else "±"
        t_quad = "x^2" if _w else "x²"
        t_sqrt = "sqrt(x)" if _w else "√x"

        # 関数入力（シンプルなスタイル）
        ax_func = plt.axes([0.62, 0.85, 0.35, 0.04])
        ax_func.set_facecolor(self.colors['background'])
        ax_func.spines['top'].set_visible(False)
        ax_func.spines['right'].set_visible(False)
        ax_func.spines['left'].set_color(self.colors['border'])
        ax_func.spines['left'].set_linewidth(1.5)
        ax_func.spines['bottom'].set_color(self.colors['border'])
        ax_func.spines['bottom'].set_linewidth(1.5)
        self.func_text = TextBox(ax_func, 'f(x) = ', initial=self.function_expr)
        self.func_text.on_submit(self.update_function)
        
        # 関数切り替えボタン（シンプルなデザイン）
        # 一次関数ボタン
        ax_linear = plt.axes([0.62, 0.80, 0.08, 0.04])
        ax_linear.set_facecolor(self.colors['background2'])
        ax_linear.spines['top'].set_visible(False)
        ax_linear.spines['right'].set_visible(False)
        ax_linear.spines['left'].set_color(self.colors['border'])
        ax_linear.spines['left'].set_linewidth(1)
        ax_linear.spines['bottom'].set_color(self.colors['border'])
        ax_linear.spines['bottom'].set_linewidth(1)
        self.linear_button = Button(ax_linear, 'x', 
                                   color=self.colors['background2'], 
                                   hovercolor=self.colors['background'])
        self.linear_button.on_clicked(lambda x: self.set_function('x'))
        
        # 符号反転ボタン（現在の関数をマイナスにする）
        ax_negate = plt.axes([0.71, 0.80, 0.08, 0.04])
        ax_negate.set_facecolor(self.colors['background2'])
        ax_negate.spines['top'].set_visible(False)
        ax_negate.spines['right'].set_visible(False)
        ax_negate.spines['left'].set_color(self.colors['border'])
        ax_negate.spines['left'].set_linewidth(1)
        ax_negate.spines['bottom'].set_color(self.colors['border'])
        ax_negate.spines['bottom'].set_linewidth(1)
        self.negate_button = Button(ax_negate, t_negate, 
                                   color=self.colors['background2'], 
                                   hovercolor=self.colors['background'])
        self.negate_button.on_clicked(self.negate_function)
        
        # 二次関数ボタン
        ax_quad = plt.axes([0.80, 0.80, 0.08, 0.04])
        ax_quad.set_facecolor(self.colors['background2'])
        ax_quad.spines['top'].set_visible(False)
        ax_quad.spines['right'].set_visible(False)
        ax_quad.spines['left'].set_color(self.colors['border'])
        ax_quad.spines['left'].set_linewidth(1)
        ax_quad.spines['bottom'].set_color(self.colors['border'])
        ax_quad.spines['bottom'].set_linewidth(1)
        self.quad_button = Button(ax_quad, t_quad, 
                                 color=self.colors['background2'], 
                                 hovercolor=self.colors['background'])
        self.quad_button.on_clicked(lambda x: self.set_function('x**2'))
        
        # ルート関数ボタン
        ax_sqrt = plt.axes([0.89, 0.80, 0.08, 0.04])
        ax_sqrt.set_facecolor(self.colors['background2'])
        ax_sqrt.spines['top'].set_visible(False)
        ax_sqrt.spines['right'].set_visible(False)
        ax_sqrt.spines['left'].set_color(self.colors['border'])
        ax_sqrt.spines['left'].set_linewidth(1)
        ax_sqrt.spines['bottom'].set_color(self.colors['border'])
        ax_sqrt.spines['bottom'].set_linewidth(1)
        self.sqrt_button = Button(ax_sqrt, t_sqrt, 
                                 color=self.colors['background2'], 
                                 hovercolor=self.colors['background'])
        self.sqrt_button.on_clicked(lambda x: self.set_function('sqrt(x)'))
        
        # シンプルなスライダー群（入力欄付き）
        # Xスライダー
        ax_x = plt.axes([0.62, 0.75, 0.18, 0.03])
        ax_x.set_facecolor(self.colors['background'])
        ax_x.spines['top'].set_visible(False)
        ax_x.spines['right'].set_visible(False)
        ax_x.spines['left'].set_color(self.colors['border'])
        ax_x.spines['left'].set_linewidth(1.5)
        ax_x.spines['bottom'].set_color(self.colors['border'])
        ax_x.spines['bottom'].set_linewidth(1.5)
        self.x_slider = Slider(ax_x, 'a', 0, 3, valinit=self.a, 
                              valfmt='%.2f', valstep=0.01,
                              facecolor=self.colors['primary'], 
                              edgecolor=self.colors['primary'])
        self.x_slider.valtext.set_visible(False)
        self.x_slider.on_changed(self.on_a_slider_changed)
        
        # a入力欄
        ax_a_input = plt.axes([0.82, 0.75, 0.06, 0.03])
        ax_a_input.set_facecolor(self.colors['background'])
        self.a_input = TextBox(ax_a_input, '', initial=f'{self.a:.2f}')
        self.a_input.on_submit(self.on_a_input_submit)
        
        # εスライダー
        ax_eps = plt.axes([0.62, 0.70, 0.18, 0.03])
        ax_eps.set_facecolor(self.colors['background'])
        ax_eps.spines['top'].set_visible(False)
        ax_eps.spines['right'].set_visible(False)
        ax_eps.spines['left'].set_color(self.colors['border'])
        ax_eps.spines['left'].set_linewidth(1.5)
        ax_eps.spines['bottom'].set_color(self.colors['border'])
        ax_eps.spines['bottom'].set_linewidth(1.5)
        self.eps_slider = Slider(ax_eps, 'ε', 0.001, 2, valinit=self.epsilon,
                                valfmt='%.3f', valstep=0.001,
                                facecolor=self.colors['accent'],
                                edgecolor=self.colors['accent'])
        self.eps_slider.valtext.set_visible(False)
        self.eps_slider.on_changed(self.on_eps_slider_changed)
        
        # ε入力欄
        ax_eps_input = plt.axes([0.82, 0.70, 0.06, 0.03])
        ax_eps_input.set_facecolor(self.colors['background'])
        self.eps_input = TextBox(ax_eps_input, '', initial=f'{self.epsilon:.3f}')
        self.eps_input.on_submit(self.on_eps_input_submit)
        
        # δスライダー
        ax_delta = plt.axes([0.62, 0.65, 0.18, 0.03])
        ax_delta.set_facecolor(self.colors['background'])
        ax_delta.spines['top'].set_visible(False)
        ax_delta.spines['right'].set_visible(False)
        ax_delta.spines['left'].set_color(self.colors['border'])
        ax_delta.spines['left'].set_linewidth(1.5)
        ax_delta.spines['bottom'].set_color(self.colors['border'])
        ax_delta.spines['bottom'].set_linewidth(1.5)
        self.delta_slider = Slider(ax_delta, 'δ', 0.001, 2, valinit=self.delta,
                                  valfmt='%.3f', valstep=0.001,
                                  facecolor=self.colors['accent2'],
                                  edgecolor=self.colors['accent2'])
        self.delta_slider.valtext.set_visible(False)
        self.delta_slider.on_changed(self.on_delta_slider_changed)
        
        # δ入力欄
        ax_delta_input = plt.axes([0.82, 0.65, 0.06, 0.03])
        ax_delta_input.set_facecolor(self.colors['background'])
        self.delta_input = TextBox(ax_delta_input, '', initial=f'{self.delta:.3f}')
        self.delta_input.on_submit(self.on_delta_input_submit)
        
        # bスライダー
        ax_b = plt.axes([0.62, 0.60, 0.18, 0.03])
        ax_b.set_facecolor(self.colors['background'])
        ax_b.spines['top'].set_visible(False)
        ax_b.spines['right'].set_visible(False)
        ax_b.spines['left'].set_color(self.colors['border'])
        ax_b.spines['left'].set_linewidth(1.5)
        ax_b.spines['bottom'].set_color(self.colors['border'])
        ax_b.spines['bottom'].set_linewidth(1.5)
        self.b_slider = Slider(ax_b, 'b', -2, 2, valinit=self.b,
                              valfmt='%.2f', valstep=0.01,
                              facecolor=self.colors['secondary'],
                              edgecolor=self.colors['secondary'])
        self.b_slider.valtext.set_visible(False)
        self._b_slider_cid = self.b_slider.on_changed(self.on_b_slider_changed)
        
        # b入力欄
        ax_b_input = plt.axes([0.82, 0.60, 0.06, 0.03])
        ax_b_input.set_facecolor(self.colors['background'])
        self.b_input = TextBox(ax_b_input, '', initial=f'{self.b:.2f}')
        self.b_input.on_submit(self.on_b_input_submit)
        
        # bを0にするボタン
        ax_b_zero = plt.axes([0.62, 0.55, 0.25, 0.03])
        ax_b_zero.set_facecolor(self.colors['background2'])
        ax_b_zero.spines['top'].set_visible(False)
        ax_b_zero.spines['right'].set_visible(False)
        ax_b_zero.spines['left'].set_color(self.colors['border'])
        ax_b_zero.spines['left'].set_linewidth(1)
        ax_b_zero.spines['bottom'].set_color(self.colors['border'])
        ax_b_zero.spines['bottom'].set_linewidth(1)
        self.b_zero_button = Button(ax_b_zero, 'b=0', 
                                    color=self.colors['background2'], 
                                    hovercolor=self.colors['background'])
        self.b_zero_button.on_clicked(self.set_b_to_zero)
        
        # 拡大縮小ボタン
        ax_zoom_in = plt.axes([0.62, 0.50, 0.12, 0.03])
        ax_zoom_in.set_facecolor(self.colors['background2'])
        ax_zoom_in.spines['top'].set_visible(False)
        ax_zoom_in.spines['right'].set_visible(False)
        ax_zoom_in.spines['left'].set_color(self.colors['border'])
        ax_zoom_in.spines['left'].set_linewidth(1)
        ax_zoom_in.spines['bottom'].set_color(self.colors['border'])
        ax_zoom_in.spines['bottom'].set_linewidth(1)
        self.zoom_in_button = Button(ax_zoom_in, t_zoom_in, 
                                   color=self.colors['background2'], 
                                   hovercolor=self.colors['background'])
        self.zoom_in_button.on_clicked(self.zoom_in)
        
        ax_zoom_out = plt.axes([0.75, 0.50, 0.12, 0.03])
        ax_zoom_out.set_facecolor(self.colors['background2'])
        ax_zoom_out.spines['top'].set_visible(False)
        ax_zoom_out.spines['right'].set_visible(False)
        ax_zoom_out.spines['left'].set_color(self.colors['border'])
        ax_zoom_out.spines['left'].set_linewidth(1)
        ax_zoom_out.spines['bottom'].set_color(self.colors['border'])
        ax_zoom_out.spines['bottom'].set_linewidth(1)
        self.zoom_out_button = Button(ax_zoom_out, t_zoom_out, 
                                    color=self.colors['background2'], 
                                    hovercolor=self.colors['background'])
        self.zoom_out_button.on_clicked(self.zoom_out)
        
        # リセットボタン
        ax_reset = plt.axes([0.62, 0.45, 0.25, 0.03])
        ax_reset.set_facecolor(self.colors['background2'])
        ax_reset.spines['top'].set_visible(False)
        ax_reset.spines['right'].set_visible(False)
        ax_reset.spines['left'].set_color(self.colors['border'])
        ax_reset.spines['left'].set_linewidth(1)
        ax_reset.spines['bottom'].set_color(self.colors['border'])
        ax_reset.spines['bottom'].set_linewidth(1)
        self.reset_button = Button(ax_reset, t_reset, 
                                  color=self.colors['background2'], 
                                  hovercolor=self.colors['background'])
        self.reset_button.on_clicked(self.reset_to_initial)
        
        # シンプルなタイトル（関数入力の上に配置）— ブラウザは Canvas 用 CJK フォントが無いので数式表記に切り替え
        _title = r"$\varepsilon$-$\delta$" if _w else "ε-δ論法"
        self.fig.suptitle(_title, fontsize=16, fontweight='bold', 
                         color=self.colors['primary'], y=0.96, x=0.795,
                         bbox=dict(boxstyle="round,pad=0.4", facecolor=self.colors['background'], 
                                 edgecolor=self.colors['border'], linewidth=1.5))
        
        # 拡大縮小機能の有効化
        self.setup_zoom_pan()
    
    def on_a_slider_changed(self, val):
        """aスライダーが変更されたときに入力欄を更新"""
        self.a_input.set_val(f'{val:.2f}')
        self.update(val)
    
    def on_a_input_submit(self, text):
        """a入力欄が変更されたときにスライダーを更新"""
        try:
            val = float(text)
            if 0 <= val <= 3:
                self.x_slider.set_val(val)
            else:
                self.a_input.set_val(f'{self.x_slider.val:.2f}')
        except ValueError:
            self.a_input.set_val(f'{self.x_slider.val:.2f}')
    
    def on_eps_slider_changed(self, val):
        """εスライダーが変更されたときに入力欄を更新"""
        self.eps_input.set_val(f'{val:.3f}')
        self.update(val)
    
    def on_eps_input_submit(self, text):
        """ε入力欄が変更されたときにスライダーを更新"""
        try:
            val = float(text)
            if 0.001 <= val <= 2:
                self.eps_slider.set_val(val)
            else:
                self.eps_input.set_val(f'{self.eps_slider.val:.3f}')
        except ValueError:
            self.eps_input.set_val(f'{self.eps_slider.val:.3f}')
    
    def on_delta_slider_changed(self, val):
        """δスライダーが変更されたときに入力欄を更新"""
        self.delta_input.set_val(f'{val:.3f}')
        self.update(val)
    
    def on_delta_input_submit(self, text):
        """δ入力欄が変更されたときにスライダーを更新"""
        try:
            val = float(text)
            if 0.001 <= val <= 2:
                self.delta_slider.set_val(val)
            else:
                self.delta_input.set_val(f'{self.delta_slider.val:.3f}')
        except ValueError:
            self.delta_input.set_val(f'{self.delta_slider.val:.3f}')
    
    def on_b_slider_changed(self, val):
        """bスライダーが変更されたときに入力欄を更新"""
        self.b_input.set_val(f'{val:.2f}')
        self.update(val)
    
    def on_b_input_submit(self, text):
        """b入力欄が変更されたときにスライダーを更新"""
        try:
            val = float(text)
            if -2 <= val <= 2:
                self.b_slider.set_val(val)
            else:
                self.b_input.set_val(f'{self.b_slider.val:.2f}')
        except ValueError:
            self.b_input.set_val(f'{self.b_slider.val:.2f}')
    
    def set_b_to_zero(self, event=None):
        """bを0に設定"""
        if self._streamlit_mode:
            self.b = 0.0
            self.update(None)
            return
        if hasattr(self, "_b_slider_cid"):
            self.b_slider.disconnect(self._b_slider_cid)
        self.b_slider.set_val(0.0)
        self.b_input.set_val("0.00")
        self._b_slider_cid = self.b_slider.on_changed(self.on_b_slider_changed)
        self.update(None)
    
    def set_function(self, expr):
        """関数を設定（ボタン用）"""
        if not self._streamlit_mode:
            self.func_text.set_val(expr)
        self.update_function(expr)
    
    def negate_function(self, event):
        """現在の関数の符号を反転する"""
        try:
            # 現在の関数式を解析
            x_sym = sp.Symbol('x')
            expr = sp.sympify(self.function_expr)
            # 符号を反転
            negated_expr = -expr
            # 文字列に変換
            negated_str = str(negated_expr)
            if not self._streamlit_mode:
                self.func_text.set_val(negated_str)
            self.update_function(negated_str)
        except:
            # エラーの場合は何もしない
            pass
    
    def update_function(self, text):
        """関数を更新"""
        # 関数式を更新
        old_expr = self.function_expr
        try:
            # 新しい関数式が有効かチェック
            x_sym = sp.Symbol('x')
            expr = sp.sympify(text)
            test_f = sp.lambdify(x_sym, expr, 'numpy')
            test_result = test_f(np.array([0.0]))
            
            # 有効な場合のみ更新
            self.function_expr = text
        except:
            # 無効な場合は元に戻す
            self.function_expr = old_expr
            if not self._streamlit_mode:
                self.func_text.set_val(old_expr)
            return
        
        # 現在の表示範囲に応じてxデータを更新
        self.update_x_data()
        
        # f(x)とf(x)+bの線を更新
        # x <= aの部分（f(x)）- aの点を含める
        x_f = self.x[self.x <= self.a]
        if len(x_f) > 0 and x_f[-1] < self.a:
            # aの点を確実に含める
            x_f = np.append(x_f, self.a)
        y_f = self.evaluate_f(x_f)
        self.line_f.set_xdata(x_f)
        self.line_f.set_ydata(y_f)
        
        # x >= aの部分（f(x)+b）- aの点を含める
        x_f_plus_b = self.x[self.x >= self.a]
        if len(x_f_plus_b) > 0 and x_f_plus_b[0] > self.a:
            # aの点を確実に含める
            x_f_plus_b = np.insert(x_f_plus_b, 0, self.a)
        y_f_plus_b = self.evaluate_f_plus_b(x_f_plus_b)
        self.line_f_plus_b.set_xdata(x_f_plus_b)
        self.line_f_plus_b.set_ydata(y_f_plus_b)
        
        # 全体の関数（後方互換性のため）
        self.y = self.evaluate_function(self.x)
        self.line.set_xdata(self.x)
        self.line.set_ydata(self.y)
        
        # グラフを更新（範囲やbの値は保持される）
        self.update(None)
        self.fig.canvas.draw_idle()
    
    def update(self, val):
        """グラフを更新"""
        if not self._streamlit_mode:
            self.a = self.x_slider.val
            self.epsilon = self.eps_slider.val
            self.delta = self.delta_slider.val
            b_raw = self.b_slider.val
            if abs(b_raw) < 1e-12:
                self.b = 0.0
                if np.signbit(b_raw):
                    if hasattr(self, "_b_slider_cid"):
                        self.b_slider.disconnect(self._b_slider_cid)
                    self.b_slider.set_val(0.0)
                    self.b_input.set_val("0.00")
                    self._b_slider_cid = self.b_slider.on_changed(self.on_b_slider_changed)
            else:
                self.b = b_raw
        else:
            b_raw = self.b
            if abs(b_raw) < 1e-12:
                self.b = 0.0
            else:
                self.b = b_raw
        
        # 現在の表示範囲に応じてxデータを更新（必要に応じて）
        # ただし、update()が呼ばれるたびに更新するとパフォーマンスが悪いので、
        # 表示範囲が大きく変わった場合のみ更新する
        if not hasattr(self, 'x') or len(self.x) == 0:
            # xデータが初期化されていない場合は更新
            self.update_x_data()
        else:
            current_xlim = self.ax.get_xlim()
            x_margin = (current_xlim[1] - current_xlim[0]) * 0.5
            x_min_needed = current_xlim[0] - x_margin
            x_max_needed = current_xlim[1] + x_margin
            
            # 現在のxデータの範囲を確認
            if self.x[0] > x_min_needed or self.x[-1] < x_max_needed:
                # xデータを更新する必要がある
                self.update_x_data()
        
        # 既存の要素をクリア（関数の線とグリッド線以外）
        for artist in self.ax.collections + self.ax.patches + self.ax.texts:
            artist.remove()
        
        # 関数の線以外の線を削除
        lines_to_remove = []
        for line in self.ax.lines:
            if line != self.line and line != self.line_f and line != self.line_f_plus_b:
                lines_to_remove.append(line)
        
        for line in lines_to_remove:
            line.remove()
        
        # 凡例をクリア
        if self.ax.get_legend():
            self.ax.get_legend().remove()
        
        # f(x)とf(x)+bの線を更新
        # x <= aの部分（f(x)）- aの点を含める
        x_f = self.x[self.x <= self.a]
        if len(x_f) > 0 and x_f[-1] < self.a:
            # aの点を確実に含める
            x_f = np.append(x_f, self.a)
        y_f = self.evaluate_f(x_f)
        self.line_f.set_xdata(x_f)
        self.line_f.set_ydata(y_f)
        
        # x >= aの部分（f(x)+b）- aの点を含める
        x_f_plus_b = self.x[self.x >= self.a]
        if len(x_f_plus_b) > 0 and x_f_plus_b[0] > self.a:
            # aの点を確実に含める
            x_f_plus_b = np.insert(x_f_plus_b, 0, self.a)
        y_f_plus_b = self.evaluate_f_plus_b(x_f_plus_b)
        self.line_f_plus_b.set_xdata(x_f_plus_b)
        self.line_f_plus_b.set_ydata(y_f_plus_b)
        
        # 全体の関数（後方互換性のため）
        self.y = self.evaluate_function(self.x)
        self.line.set_xdata(self.x)
        self.line.set_ydata(self.y)
        
        # f(a)の値を高精度で計算
        f_a = self.evaluate_function(np.array([self.a]))[0]
        # f_aがNaNの場合、0を使用（sqrt(x)でa < 0の場合など）
        if np.isnan(f_a):
            f_a = 0.0
        
        # x座標でaからδ離れた点を高精度で設定
        a_minus_delta = self.a - self.delta
        a_plus_delta = self.a + self.delta
        
        # a-δとa+δに対応するy座標y1とy2を高精度で計算
        y1 = self.evaluate_function(np.array([a_minus_delta]))[0]
        y2 = self.evaluate_function(np.array([a_plus_delta]))[0]
        
        # y1またはy2がNaNの場合、f_aを使用（sqrt(x)でx < 0の場合など）
        if np.isnan(y1):
            y1 = f_a
        if np.isnan(y2):
            y2 = f_a
        
        # 数値精度の向上：微小な値でも正確に計算
        if abs(self.delta) < 1e-6:
            # 非常に小さいδの場合、より高精度な計算を使用
            y1_temp = self.evaluate_function(np.array([a_minus_delta]))[0]
            y2_temp = self.evaluate_function(np.array([a_plus_delta]))[0]
            if not np.isnan(y1_temp):
                y1 = y1_temp
            if not np.isnan(y2_temp):
                y2 = y2_temp
        
        # y座標でf(a)からε離れた点を設定
        f_a_plus_epsilon = f_a + self.epsilon
        f_a_minus_epsilon = f_a - self.epsilon
        
        # 関数が増加か減少かを判定
        is_increasing = self.is_increasing_at_a()
        
        # f(a)+εとf(a)-εに対応するx座標x1とx2を高精度で求める
        # 単調増加の場合：
        #   x1: f(x) = f(a)-εを満たすx座標（x < aの範囲）
        #   x2: f(x)+b = f(a)+εを満たすx座標（x >= aの範囲）
        # 単調減少の場合：
        #   x1: f(x) = f(a)+εを満たすx座標（x < aの範囲）
        #   x2: f(x)+b = f(a)-εを満たすx座標（x >= aの範囲）
        
        if is_increasing:
            # 単調増加の場合（既存のロジック）
            target_y_for_x1 = f_a_minus_epsilon
            target_y_for_x2 = f_a_plus_epsilon
        else:
            # 単調減少の場合
            target_y_for_x1 = f_a_plus_epsilon
            target_y_for_x2 = f_a_minus_epsilon
        
        # x1: 関数f(x)と水平線y=target_y_for_x1の交点（ジャンプ除外ON、高精度）
        intersections_for_x1 = self.find_intersections(target_y_for_x1, -3, 3, jump_exclude=True)
        
        # x2: 関数f(x)+bと水平線y=target_y_for_x2の交点
        # f(x)+b = target_y_for_x2 を満たすx座標（x >= aの範囲で）
        x2 = None
        
        # 特殊な関数の場合、解析的にx2を計算
        # ただし、x >= aの範囲でのみ有効
        try:
            x_sym = sp.Symbol('x')
            expr = sp.sympify(self.function_expr)
            # 関数がxまたはx+定数と等しいかチェック (x, x+1, x-1 など)
            # x+cの形式を検出
            diff_from_x = sp.simplify(expr - x_sym)
            if diff_from_x.is_number:
                if is_increasing:
                    # f(x)+b = x+c+b = f(a)+ε = a+c+ε → x = a+ε-b
                    x2_calculated = self.a + self.epsilon - self.b
                else:
                    # f(x)+b = x+c+b = f(a)-ε = a+c-ε → x = a-ε-b
                    x2_calculated = self.a - self.epsilon - self.b
                if x2_calculated >= self.a:
                    x2 = x2_calculated
                else:
                    # bがεを超えた場合、x2 = aに設定
                    x2 = self.a
            # 関数が-x+定数の場合を検出 (-x, -x+1, -x-1 など)
            else:
                diff_from_neg_x = sp.simplify(expr + x_sym)
                if diff_from_neg_x.is_number:
                    if is_increasing:
                        x2_calculated = self.a + self.epsilon - self.b
                    else:
                        # f(x)+b = -x+c+b = f(a)-ε = -a+c-ε → -x = -a-ε-b → x = a+ε+b
                        x2_calculated = self.a + self.epsilon + self.b
                    if x2_calculated >= self.a:
                        x2 = x2_calculated
                    else:
                        x2 = self.a
                # 関数がsqrt(x)+定数の場合を検出 (sqrt(x), sqrt(x)+1, sqrt(x)-1 など)
                else:
                    diff_from_sqrt = sp.simplify(expr - sp.sqrt(x_sym))
                    if diff_from_sqrt.is_number:
                        # f(x)+b = sqrt(x)+c+b = f(a)+ε = sqrt(a)+c+ε
                        # → sqrt(x) = sqrt(a)+ε-b → x = (sqrt(a)+ε-b)²
                        # sqrt(a)を計算（定数部分cを除く）
                        sqrt_a = np.sqrt(self.a) if self.a >= 0 else 0
                        if is_increasing:
                            sqrt_arg = sqrt_a + self.epsilon - self.b
                        else:
                            sqrt_arg = sqrt_a - self.epsilon - self.b
                        if sqrt_arg >= 0:
                            x2_calculated = sqrt_arg ** 2
                            if x2_calculated >= self.a:
                                x2 = x2_calculated
                            else:
                                x2 = self.a
                        else:
                            x2 = self.a
                    else:
                        # 関数が-sqrt(x)+定数の場合を検出 (-sqrt(x), -sqrt(x)+1, -sqrt(x)-1 など)
                        diff_from_neg_sqrt = sp.simplify(expr + sp.sqrt(x_sym))
                        if diff_from_neg_sqrt.is_number:
                            # f(x)+b = -sqrt(x)+c+b = f(a)-ε = -sqrt(a)+c-ε
                            # → -sqrt(x) = -sqrt(a)-ε-b → sqrt(x) = sqrt(a)+ε+b → x = (sqrt(a)+ε+b)²
                            sqrt_a = np.sqrt(self.a) if self.a >= 0 else 0
                            if is_increasing:
                                # 単調増加（実際には-sqrt(x)は単調減少だが念のため）
                                sqrt_arg = sqrt_a + self.epsilon - self.b
                            else:
                                # 単調減少の場合: target_y_for_x2 = f(a) - ε = -sqrt(a) + c - ε
                                # f(x) + b = -sqrt(x) + c + b = -sqrt(a) + c - ε
                                # -sqrt(x) = -sqrt(a) - ε - b
                                # sqrt(x) = sqrt(a) + ε + b
                                sqrt_arg = sqrt_a + self.epsilon + self.b
                            if sqrt_arg >= 0:
                                x2_calculated = sqrt_arg ** 2
                                if x2_calculated >= self.a:
                                    x2 = x2_calculated
                                else:
                                    x2 = self.a
                            else:
                                x2 = self.a
        except:
            pass
        
        # x2がまだ計算されていない場合、数値的に交点を求める
        if x2 is None:
            # その他の関数の場合、数値的に交点を求める
            # x >= aの範囲でf(x)+bとtarget_y_for_x2の交点を求める
            x_candidates_x2 = np.linspace(self.a, max(3, self.a + abs(self.epsilon) + abs(self.b) + 1), 5000)
            y_fx_plus_b_x2 = self.evaluate_f_plus_b(x_candidates_x2)
            diff_x2 = y_fx_plus_b_x2 - target_y_for_x2
            # NaN値を処理
            diff_x2_sign = np.sign(diff_x2)
            diff_x2_sign[np.isnan(diff_x2)] = 0
            crossings_x2 = np.where(np.diff(diff_x2_sign) != 0)[0]
            
            if len(crossings_x2) > 0:
                # 最もaに近い交点を選択
                intersections_x2 = []
                for i in crossings_x2:
                    if i < len(x_candidates_x2) - 1:
                        x1_i, x2_i = x_candidates_x2[i], x_candidates_x2[i+1]
                        y1_i, y2_i = y_fx_plus_b_x2[i], y_fx_plus_b_x2[i+1]
                        # NaN値をスキップ
                        if np.isnan(y1_i) or np.isnan(y2_i):
                            continue
                        if abs(y2_i - y1_i) > 1e-10:
                            t = (target_y_for_x2 - y1_i) / (y2_i - y1_i)
                            x_intersect = x1_i + t * (x2_i - x1_i)
                            if x_intersect >= self.a:  # x >= aの範囲のみ
                                intersections_x2.append(x_intersect)
                
                if intersections_x2:
                    distances_x2 = [abs(x - self.a) for x in intersections_x2]
                    closest_idx_x2 = np.argmin(distances_x2)
                    x2 = intersections_x2[closest_idx_x2]
            
            # 交点が見つからない場合、x2 = aに設定
            if x2 is None:
                x2 = self.a
        
        # 微小なεの場合、より高精度な交点計算
        if abs(self.epsilon) < 1e-6:
            # より細かい範囲で交点を再計算
            intersections_for_x1 = self.find_intersections(target_y_for_x1, self.a - 2*abs(self.delta), self.a + 2*abs(self.delta), jump_exclude=True)
            # x2も再計算
            x_candidates_x2 = np.linspace(self.a, self.a + 2*abs(self.delta), 5000)
            y_fx_plus_b_x2 = self.evaluate_f_plus_b(x_candidates_x2)
            diff_x2 = y_fx_plus_b_x2 - target_y_for_x2
            # NaN値を処理
            diff_x2_sign = np.sign(diff_x2)
            diff_x2_sign[np.isnan(diff_x2)] = 0
            crossings_x2 = np.where(np.diff(diff_x2_sign) != 0)[0]
            
            if len(crossings_x2) > 0:
                intersections_x2 = []
                for i in crossings_x2:
                    if i < len(x_candidates_x2) - 1:
                        x1_i, x2_i = x_candidates_x2[i], x_candidates_x2[i+1]
                        y1_i, y2_i = y_fx_plus_b_x2[i], y_fx_plus_b_x2[i+1]
                        # NaN値をスキップ
                        if np.isnan(y1_i) or np.isnan(y2_i):
                            continue
                        if abs(y2_i - y1_i) > 1e-10:
                            t = (target_y_for_x2 - y1_i) / (y2_i - y1_i)
                            x_intersect = x1_i + t * (x2_i - x1_i)
                            if x_intersect >= self.a:  # x >= aの範囲
                                intersections_x2.append(x_intersect)
                
                if intersections_x2:
                    distances_x2 = [abs(x - self.a) for x in intersections_x2]
                    closest_idx_x2 = np.argmin(distances_x2)
                    x2 = intersections_x2[closest_idx_x2]
            
            # 微小なεの場合でも交点が見つからない場合はx2 = aに設定
            if x2 is None:
                x2 = self.a
        
        # x1を設定（target_y_for_x1を満たすx座標、最もaに近い交点を選択）
        x1 = None
        if intersections_for_x1:
            # 交点からaより左側で最もaに近いものをx1として選択
            x_candidates = [x for x, y in intersections_for_x1 if x < self.a]
            if x_candidates:
                distances_x1 = [abs(x - self.a) for x in x_candidates]
                closest_x1_idx = np.argmin(distances_x1)
                x1 = x_candidates[closest_x1_idx]
            else:
                # aより左側に交点がない場合は、最もaに近いものを選択
                distances_x1 = [abs(x - self.a) for x, y in intersections_for_x1]
                closest_x1_idx = np.argmin(distances_x1)
                x1 = intersections_for_x1[closest_x1_idx][0]
        
        # D6領域の作成（白色四角形）
        # 説明: x=0からx=aまでの範囲で、y=f(a)からy=f(a)+bまでの矩形領域
        # この領域は、不連続点x=aにおける関数の値の差（b）を視覚化するために使用されます
        # (0,f(a))と(0,f(a)+b)と(a,f(a))と(a,f(a)+b)で囲まれた領域
        d6_verts = []
        # 左下: (0, f(a))
        d6_verts.append((0, f_a))
        # 右下: (a, f(a))
        d6_verts.append((self.a, f_a))
        # 右上: (a, f(a)+b) ただしb<0なら(a, f(a))
        if self.b >= 0:
            d6_verts.append((self.a, f_a + self.b))
        else:
            d6_verts.append((self.a, f_a))
        # 左上: (0, f(a)+b) ただしb<0なら(0, f(a))
        if self.b >= 0:
            d6_verts.append((0, f_a + self.b))
        else:
            d6_verts.append((0, f_a))
        # D6領域（白色四角形）を描画（最前面）
        if len(d6_verts) > 2 and abs(self.b) > 1e-6:
            d6_path = Path(d6_verts)
            d6_patch = patches.PathPatch(d6_path, facecolor='white', alpha=0.95, 
                                       edgecolor='none', linewidth=0)
            self.ax.add_patch(d6_patch)
            # D6領域のラベル
            # self.add_region_label(d6_verts, 'D6', color='gray', position='left')

        # D1_1領域の作成（D6の後に描画）
        # 説明: ε近傍の領域（単調増加：下側、単調減少：上側）
        # 単調増加: x=0からx=aまでの範囲で、y=f(a)-εからy=f(a)までの領域
        # 単調減少: x=0からx=aまでの範囲で、y=f(a)からy=f(a)+εまでの領域（x軸基準で反転）
        if x1 is not None and x2 is not None:
            # D1_1領域の頂点を定義
            d1_1_verts = []
            
            x_curve_d1_1 = np.linspace(0, self.a, 500)
            y_curve_d1_1 = self.evaluate_function(x_curve_d1_1)
            
            if is_increasing:
                # 単調増加の場合：f(a)-ε から f(a) の間
                d1_1_verts.append((0, f_a_minus_epsilon))
                valid_mask = ~np.isnan(y_curve_d1_1) & (y_curve_d1_1 >= f_a_minus_epsilon)
                valid_indices = np.where(valid_mask)[0]
                if len(valid_indices) > 0:
                    x_right = x_curve_d1_1[valid_indices[-1]]
                    for i in valid_indices:
                        if not np.isnan(y_curve_d1_1[i]):
                            d1_1_verts.append((x_curve_d1_1[i], y_curve_d1_1[i]))
                    d1_1_verts.append((x_right, f_a_minus_epsilon))
                    d1_1_verts.append((self.a, f_a))
                else:
                    d1_1_verts.append((self.a, f_a_minus_epsilon))
                    d1_1_verts.append((self.a, f_a))
                d1_1_verts.append((0, f_a))
            else:
                # 単調減少の場合：f(a)+ε から f(a) の間（x軸対称で反転）
                d1_1_verts.append((0, f_a_plus_epsilon))
                valid_mask = ~np.isnan(y_curve_d1_1) & (y_curve_d1_1 <= f_a_plus_epsilon)
                valid_indices = np.where(valid_mask)[0]
                if len(valid_indices) > 0:
                    x_right = x_curve_d1_1[valid_indices[-1]]
                    for i in valid_indices:
                        if not np.isnan(y_curve_d1_1[i]):
                            d1_1_verts.append((x_curve_d1_1[i], y_curve_d1_1[i]))
                    d1_1_verts.append((x_right, f_a_plus_epsilon))
                    d1_1_verts.append((self.a, f_a))
                else:
                    d1_1_verts.append((self.a, f_a_plus_epsilon))
                    d1_1_verts.append((self.a, f_a))
                d1_1_verts.append((0, f_a))
            
            # D1_1領域を描画
            if len(d1_1_verts) > 2:
                d1_1_path = Path(d1_1_verts)
                d1_1_patch = patches.PathPatch(d1_1_path, facecolor=self.colors['accent'], 
                                             alpha=0.3, edgecolor='none', 
                                             linewidth=0)
                self.ax.add_patch(d1_1_patch)
                # D1_1領域のラベル
                # self.add_region_label(d1_1_verts, 'D1_1', color=self.colors['accent'], position='top_left')


        # D5領域の作成（D2より手前に描画）
        # 説明: 不連続点x=aの右側で、関数f(x)+bがf(a)と交わる点までの矩形領域
        # この領域は、不連続点x=aの右側における関数の挙動を視覚化するために使用されます
        # y=f(a)とy=f(x)+bの交点(x0, y0)を求める
        # f(x)+b = f(a) となるx0を探す（x > a の範囲で）
        x0 = None
        y0 = None
        
        # sqrt(x)+定数の場合、解析的にx0を計算
        # f(x)+b = f(a) を解く
        # sqrt(x)+c+b = sqrt(a)+c → sqrt(x) = sqrt(a)-b → x = (sqrt(a)-b)²
        try:
            x_sym = sp.Symbol('x')
            expr = sp.sympify(self.function_expr)
            # sqrt(x)+定数の形式を検出
            diff_from_sqrt = sp.simplify(expr - sp.sqrt(x_sym))
            if diff_from_sqrt.is_number:
                # sqrt(a)を計算（定数部分cを除く）
                sqrt_a = np.sqrt(self.a) if self.a >= 0 else 0
                sqrt_arg = sqrt_a - self.b
                if sqrt_arg >= 0:
                    x0_calculated = sqrt_arg ** 2
                    if x0_calculated > self.a:
                        x0 = x0_calculated
                        y0 = f_a
        except:
            pass
        
        # x0がまだ計算されていない場合、数値的に交点を求める
        if x0 is None:
            # bがマイナスの場合、検索範囲を拡大（bに応じて）
            search_max = max(3, self.a + abs(self.b) * 10 + 10)
            x_candidates = np.linspace(self.a, search_max, 2000)
            y_fx_plus_b = self.evaluate_f_plus_b(x_candidates)
            y_fa = f_a
            # 交点を探す（NaN値を処理）
            diff_d5 = y_fx_plus_b - y_fa
            # NaN値を含む差分は処理しない
            valid_mask_d5 = ~np.isnan(diff_d5)
            idx_cross = []
            for i in range(len(diff_d5) - 1):
                if valid_mask_d5[i] and valid_mask_d5[i+1]:
                    if np.signbit(diff_d5[i]) != np.signbit(diff_d5[i+1]):
                        idx_cross.append(i)
            if len(idx_cross) > 0:
                i = idx_cross[0]
                x1_d5, x2_d5 = x_candidates[i], x_candidates[i+1]
                y1_d5, y2_d5 = y_fx_plus_b[i], y_fx_plus_b[i+1]
                # 線形補間で交点を求める（NaN値チェック）
                if not np.isnan(y1_d5) and not np.isnan(y2_d5) and abs(y2_d5 - y1_d5) > 1e-10:
                    t = (y_fa - y1_d5) / (y2_d5 - y1_d5)
                    x0 = x1_d5 + t * (x2_d5 - x1_d5)
                    y0 = y_fa
        # D5領域の描画
        if x0 is not None and y0 is not None:
            d5_verts = [
                (self.a, f_a),      # ⑤
                (self.a, 0),        # ⑥
                (x0, 0),            # ⑦
                (x0, y0)            # ⑧
            ]
            d5_path = Path(d5_verts)
            d5_patch = patches.PathPatch(d5_path, facecolor='white', alpha=0.95, 
                                       edgecolor='none', linewidth=0)
            self.ax.add_patch(d5_patch)
            # D5領域のラベル
            if abs(self.b) > 1e-6:
                # self.add_region_label(d5_verts, 'D5', color='gray', position='right')
                pass

        # D2領域の作成（D5領域の計算の後、x0とy0が必要なため）
        # 説明: ε近傍に対応するx座標の範囲。x=x1からx=x2までの範囲で、y=0からy=f(x)までの領域
        # この領域は、εの値に応じて関数の値がf(a)±εの範囲内に収まるx座標の範囲を視覚化します
        # D2領域を2つに分割：
        # - D2_1: ①、④、⑤、⑥で囲まれた領域（x <= aの部分）
        # - D2_2: ②、③、⑦、⑧で囲まれた領域（x >= aの部分）
        if x1 is not None and x2 is not None:
            # x1とx2の範囲で関数の値を取得
            x_min = min(x1, x2)
            x_max = max(x1, x2)
            
            # D2_1領域: ①、④、⑤、⑥で囲まれた領域（x <= aの部分）
            # ①: (x_min, 0)
            # ④: (x_min, f(x_min)) または (a, f(a))
            # ⑤: (a, f_a)
            # ⑥: (a, 0)
            if x_min < self.a:
                x_curve_d2_1 = np.linspace(x_min, self.a, 1000)
                y_curve_d2_1 = self.evaluate_function(x_curve_d2_1)
                
                d2_1_verts = []
                # ①: 左下 (x_min, 0)
                d2_1_verts.append((x_min, 0))
                # ⑥: (a, 0)
                d2_1_verts.append((self.a, 0))
                # ⑤: (a, f_a)
                d2_1_verts.append((self.a, f_a))
                # ④: 左上 - 曲線上の点（x_minからaまで）（NaNをスキップ）
                for i in range(len(x_curve_d2_1)-1, -1, -1):
                    x, y = x_curve_d2_1[i], y_curve_d2_1[i]
                    # 単調増加の場合: y >= 0、単調減少の場合: y <= 0
                    if not np.isnan(y):
                        if (is_increasing and y >= 0) or (not is_increasing and y <= 0):
                            d2_1_verts.append((x, y))
                
                if len(d2_1_verts) > 2:
                    d2_1_path = Path(d2_1_verts)
                    d2_1_patch = patches.PathPatch(d2_1_path, facecolor='#E74C3C', 
                                               alpha=0.3, edgecolor='none', 
                                               linewidth=0)
                    self.ax.add_patch(d2_1_patch)
                    # D2_1領域のラベル
                    # self.add_region_label(d2_1_verts, 'D2_1', color='#E74C3C', position='bottom_left')
            
            # D2_2領域: ②、③、⑦、⑧とf(x)+b曲線で囲まれた領域（x >= aの部分）
            # ②: (x_max, 0)
            # ③: (x_max, f(x_max)+b) - x_max > aなのでf(x)+bを使用
            # ⑦: (x0, 0) - D5領域の⑦と同じ、bが0以上のときは(a, 0) = ⑥と同じ
            # ⑧: (x0, y0) - D5領域の⑧と同じ、bがプラスのときは(a, f(a)+b)
            if x_max > self.a:
                # y=f(a)+bがy=f(a)とy=f(a)+εの間に存在する場合の処理
                if is_increasing:
                    # 単調増加の場合
                    # bがプラスのとき、⑧の点は(a, f(a)+b)
                    if self.b > 0:
                        x0_d2 = self.a
                        y0_d2 = f_a + self.b
                    elif self.b == 0:
                        x0_d2 = self.a
                        y0_d2 = f_a
                    elif x0 is not None and y0 is not None:
                        x0_d2 = x0
                        y0_d2 = y0
                    else:
                        x0_d2 = self.a
                        y0_d2 = f_a
                else:
                    # 単調減少の場合
                    # bが負のとき、⑧の点は(a, f(a)+b)
                    if self.b < 0:
                        x0_d2 = self.a
                        y0_d2 = f_a + self.b
                    elif self.b == 0:
                        x0_d2 = self.a
                        y0_d2 = f_a
                    elif x0 is not None and y0 is not None:
                        # bが正のとき、f(x)+b = f(a) の交点を使用
                        x0_d2 = x0
                        y0_d2 = y0
                    else:
                        x0_d2 = self.a
                        y0_d2 = f_a
                
                # 関数の値を取得（NaN対応）
                # x_max > aの場合、f(x_max)+bを使用
                f_x_max_plus_b = self.evaluate_f_plus_b(np.array([x_max]))[0]
                if np.isnan(f_x_max_plus_b):
                    f_x_max_plus_b = 0.0
                
                # x0_d2がx_maxより大きい場合、領域はx_maxまでに制限
                # この場合、⑦と⑧はx_maxの位置になる
                if x0_d2 > x_max:
                    x0_d2 = x_max
                    y0_d2 = f_x_max_plus_b
                
                d2_2_verts = []
                
                if is_increasing:
                    # 単調増加の場合: x軸から曲線まで（上向き）
                    # ⑦: (x0_d2, 0)
                    d2_2_verts.append((x0_d2, 0))
                    # ②: 右下 (x_max, 0)
                    d2_2_verts.append((x_max, 0))
                    # ③: 右上 (x_max, f(x_max)+b)
                    d2_2_verts.append((x_max, f_x_max_plus_b))
                    # ③から⑧までf(x)+b曲線上の点（x_maxからx0_d2まで逆順）
                    if x_max > x0_d2:
                        x_curve_d2_2 = np.linspace(x0_d2, x_max, 1000)
                        y_curve_d2_2 = self.evaluate_f_plus_b(x_curve_d2_2)
                        for i in range(len(x_curve_d2_2)-1, -1, -1):
                            x, y = x_curve_d2_2[i], y_curve_d2_2[i]
                            if not np.isnan(y) and y >= 0:
                                d2_2_verts.append((x, y))
                    # ⑧: (x0_d2, y0_d2)
                    if x0_d2 < x_max:
                        d2_2_verts.append((x0_d2, y0_d2))
                else:
                    # 単調減少の場合: x軸から曲線まで
                    # 単調増加と同じ構造で、bの符号が反転：
                    # - 単調減少でb<=0 → 単調増加でb>=0と対称
                    # - 単調減少でb>0 → 単調増加でb<0と対称
                    
                    # 曲線上の点を取得
                    x_curve_d2_2 = np.linspace(x0_d2, x_max, 1000)
                    y_curve_d2_2 = self.evaluate_f_plus_b(x_curve_d2_2)
                    
                    # 単調増加と同じ構造で描画
                    # ⑦: (x0_d2, 0)
                    d2_2_verts.append((x0_d2, 0))
                    # ②: 右端 (x_max, 0)
                    d2_2_verts.append((x_max, 0))
                    # ③: (x_max, f(x_max)+b)
                    d2_2_verts.append((x_max, f_x_max_plus_b))
                    # ③から⑧までf(x)+b曲線上の点（x_maxからx0_d2まで逆順）
                    if x_max > x0_d2:
                        for i in range(len(x_curve_d2_2)-1, -1, -1):
                            x, y = x_curve_d2_2[i], y_curve_d2_2[i]
                            if not np.isnan(y) and y <= 0:
                                d2_2_verts.append((x, y))
                    # ⑧: (x0_d2, y0_d2)
                    if x0_d2 < x_max:
                        d2_2_verts.append((x0_d2, y0_d2))
                
                if len(d2_2_verts) > 2:
                    d2_2_path = Path(d2_2_verts)
                    d2_2_patch = patches.PathPatch(d2_2_path, facecolor='#E74C3C', 
                                               alpha=0.3, edgecolor='none', 
                                               linewidth=0)
                    self.ax.add_patch(d2_2_patch)
                    # D2_2領域のラベル
                    # self.add_region_label(d2_2_verts, 'D2_2', color='#E74C3C', position='bottom_right')
            
            # D2領域の各点に番号を表記
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            x_range = xlim[1] - xlim[0]
            y_range = ylim[1] - ylim[0]
            
            # D2_1領域の点: ①、④、⑤、⑥ (非表示)
            # if x_min < self.a:
            #     # ①: (x_min, 0)
            #     label_x = x_min - x_range * 0.02
            #     label_y = 0 - y_range * 0.02
            #     self.ax.text(label_x, label_y, '①',
            #                 fontsize=12, fontweight='bold', 
            #                 color='#E74C3C', ha='right', va='top',
            #                 zorder=20,
            #                 bbox=dict(boxstyle='round,pad=0.3', 
            #                         facecolor='white', 
            #                         edgecolor='#E74C3C', 
            #                         alpha=0.9, linewidth=1.5))
            #     
            #     # ④: (x_min, f(x_min)) または (a, f(a))
            #     x_curve_d2_1 = np.linspace(x_min, self.a, 1000)
            #     y_curve_d2_1 = self.evaluate_function(x_curve_d2_1)
            #     if len(y_curve_d2_1) > 0:
            #         y_min_val = y_curve_d2_1[0]
            #         if y_min_val >= 0:
            #             label_x = x_min - x_range * 0.02
            #             label_y = y_min_val + y_range * 0.02
            #             self.ax.text(label_x, label_y, '④',
            #                         fontsize=12, fontweight='bold', 
            #                         color='#E74C3C', ha='right', va='bottom',
            #                         zorder=20,
            #                         bbox=dict(boxstyle='round,pad=0.3', 
            #                                 facecolor='white', 
            #                                 edgecolor='#E74C3C', 
            #                                 alpha=0.9, linewidth=1.5))
            #     
            #     # ⑤: (a, f_a) - D5領域の⑤と同じ
            #     label_x = self.a - x_range * 0.02
            #     label_y = f_a + y_range * 0.02
            #     self.ax.text(label_x, label_y, '⑤',
            #                 fontsize=12, fontweight='bold', 
            #                 color='black', ha='right', va='bottom',
            #                 zorder=20,
            #                 bbox=dict(boxstyle='round,pad=0.3', 
            #                         facecolor='white', 
            #                         edgecolor='black', 
            #                         alpha=0.9, linewidth=1.5))
            #     
            #     # ⑥: (a, 0) - D5領域の⑥と同じ
            #     label_x = self.a - x_range * 0.02
            #     label_y = 0 - y_range * 0.02
            #     self.ax.text(label_x, label_y, '⑥',
            #                 fontsize=12, fontweight='bold', 
            #                 color='black', ha='right', va='top',
            #                 zorder=20,
            #                 bbox=dict(boxstyle='round,pad=0.3', 
            #                         facecolor='white', 
            #                         edgecolor='black', 
            #                         alpha=0.9, linewidth=1.5))
            
            # D2_2領域の点: ②、③、⑦、⑧ (非表示)
            # if x_max > self.a:
            #     # 単調増加/減少に応じてx0_d2, y0_d2を設定
            #     if is_increasing:
            #         # 単調増加の場合
            #         if self.b > 0:
            #             x0_d2 = self.a
            #             y0_d2 = f_a + self.b
            #         elif self.b == 0:
            #             x0_d2 = self.a
            #             y0_d2 = f_a
            #         elif x0 is not None and y0 is not None:
            #             x0_d2 = x0
            #             y0_d2 = y0
            #         else:
            #             x0_d2 = self.a
            #             y0_d2 = f_a
            #     else:
            #         # 単調減少の場合
            #         if self.b < 0:
            #             x0_d2 = self.a
            #             y0_d2 = f_a + self.b
            #         elif self.b == 0:
            #             x0_d2 = self.a
            #             y0_d2 = f_a
            #         elif x0 is not None and y0 is not None:
            #             x0_d2 = x0
            #             y0_d2 = y0
            #         else:
            #             x0_d2 = self.a
            #             y0_d2 = f_a
            #     
            #     # f(x_max)+bを取得（ラベル表示用）
            #     f_x_max_plus_b_label = self.evaluate_f_plus_b(np.array([x_max]))[0]
            #     if np.isnan(f_x_max_plus_b_label):
            #         f_x_max_plus_b_label = 0.0
            #     
            #     # x0_d2がx_maxより大きい場合、ラベル位置をx_maxに制限
            #     if x0_d2 > x_max:
            #         x0_d2 = x_max
            #         y0_d2 = f_x_max_plus_b_label
            #     
            #     # ②: (x_max, 0)
            #     label_x = x_max + x_range * 0.02
            #     label_y = 0 - y_range * 0.02
            #     self.ax.text(label_x, label_y, '②',
            #                 fontsize=12, fontweight='bold', 
            #                 color='#E74C3C', ha='left', va='top',
            #                 zorder=20,
            #                 bbox=dict(boxstyle='round,pad=0.3', 
            #                         facecolor='white', 
            #                         edgecolor='#E74C3C', 
            #                         alpha=0.9, linewidth=1.5))
            #     
            #     # ③: (x_max, f(x_max)+b) - x_max > aなのでf(x)+bを使用
            #     x_curve_d2_2 = np.linspace(self.a, x_max, 1000)
            #     y_curve_d2_2 = self.evaluate_f_plus_b(x_curve_d2_2)
            #     if len(y_curve_d2_2) > 0:
            #         y_max_val = y_curve_d2_2[-1]
            #         if not np.isnan(y_max_val) and y_max_val >= 0:
            #             label_x = x_max + x_range * 0.02
            #             label_y = y_max_val - y_range * 0.02
            #             self.ax.text(label_x, label_y, '③',
            #                         fontsize=12, fontweight='bold', 
            #                         color='#E74C3C', ha='left', va='top',
            #                         zorder=20,
            #                         bbox=dict(boxstyle='round,pad=0.3', 
            #                                 facecolor='white', 
            #                                 edgecolor='#E74C3C', 
            #                                 alpha=0.9, linewidth=1.5))
            #     
            #     # ⑦と⑧: x0_d2 == x_maxの場合は②③と重なるので表示しない
            #     if x0_d2 < x_max:
            #         # ⑦: (x0_d2, 0) - bが0以上のときは(a, 0) = ⑥と同じ
            #         label_x = x0_d2 + x_range * 0.02
            #         if is_increasing:
            #             label_y = 0 - y_range * 0.02
            #             va_7 = 'top'
            #         else:
            #             label_y = 0 + y_range * 0.02
            #             va_7 = 'bottom'
            #         self.ax.text(label_x, label_y, '⑦',
            #                     fontsize=12, fontweight='bold', 
            #                     color='black', ha='left', va=va_7,
            #                     zorder=20,
            #                     bbox=dict(boxstyle='round,pad=0.3', 
            #                             facecolor='white', 
            #                             edgecolor='black', 
            #                             alpha=0.9, linewidth=1.5))
            #     
            #         # ⑧: (x0_d2, y0_d2) - bがプラスのときは(a, f(a)+b)、bが0のときは(a, f_a) = ⑤と同じ
            #         label_x = x0_d2 + x_range * 0.02
            #         if is_increasing:
            #             label_y = y0_d2 + y_range * 0.02
            #             va_8 = 'bottom'
            #         else:
            #             label_y = y0_d2 - y_range * 0.02
            #             va_8 = 'top'
            #         self.ax.text(label_x, label_y, '⑧',
            #                     fontsize=12, fontweight='bold', 
            #                     color='black', ha='left', va=va_8,
            #                     zorder=20,
            #                     bbox=dict(boxstyle='round,pad=0.3', 
            #                             facecolor='white', 
            #                             edgecolor='black', 
            #                             alpha=0.9, linewidth=1.5))

        # D1_2領域の作成（D6の後に描画）
        # 説明: ε近傍の領域（単調増加：上側、単調減少：下側）
        # 単調増加: (0, f(a)), (a, f(a)), (x2, f(a)+ε), (0, f(a)+ε)と⑧の点で囲まれた領域
        # 単調減少: (0, f(a)-ε), (a, f(a)), (x2, f(a)-ε), (0, f(a)-ε)と⑧の点で囲まれた領域（x軸基準で反転）
        if x1 is not None and x2 is not None:
            # D1_2領域の頂点を定義
            d1_2_verts = []
            
            # ⑧の点を取得（bが0以上のときは(a, f_a)、bがマイナスのときは(x0, y0)）
            if self.b >= 0:
                x8 = self.a
                y8 = f_a
            elif x0 is not None and y0 is not None:
                x8 = x0
                y8 = y0
            else:
                x8 = self.a
                y8 = f_a
            
            if is_increasing:
                # 単調増加の場合：f(a) から f(a)+ε の間
                if self.b < 0 and x0 is not None and y0 is not None:
                    d1_2_verts.append((0, f_a))
                    d1_2_verts.append((x0, y0))
                    if x2 > x0:
                        x_curve_fxpb = np.linspace(x0, x2, 100)
                        y_curve_fxpb = self.evaluate_f_plus_b(x_curve_fxpb)
                        for x, y in zip(x_curve_fxpb, y_curve_fxpb):
                            if not np.isnan(y) and (x0 <= x <= x2) and (y >= y0) and (y <= f_a_plus_epsilon):
                                d1_2_verts.append((x, y))
                    d1_2_verts.append((x2, f_a_plus_epsilon))
                    d1_2_verts.append((0, f_a_plus_epsilon))
                else:
                    d1_2_verts.append((0, f_a))
                    d1_2_verts.append((self.a, f_a))
                    if x2 > self.a:
                        x_start = self.a
                        x_candidates_start = np.linspace(self.a, x2, 1000)
                        y_candidates_start = self.evaluate_f_plus_b(x_candidates_start)
                        diff_start = y_candidates_start - f_a
                        diff_start_sign = np.sign(diff_start)
                        diff_start_sign[np.isnan(diff_start)] = 0
                        crossings_start = np.where(np.diff(diff_start_sign) != 0)[0]
                        if len(crossings_start) > 0:
                            for i in crossings_start:
                                if i < len(x_candidates_start) - 1:
                                    x1_i, x2_i = x_candidates_start[i], x_candidates_start[i+1]
                                    y1_i, y2_i = y_candidates_start[i], y_candidates_start[i+1]
                                    if not np.isnan(y1_i) and not np.isnan(y2_i) and abs(y2_i - y1_i) > 1e-10:
                                        t = (f_a - y1_i) / (y2_i - y1_i)
                                        x_intersect = x1_i + t * (x2_i - x1_i)
                                        if x_intersect >= self.a and x_intersect <= x2:
                                            x_start = x_intersect
                                            break
                        
                        x_curve_fxpb = np.linspace(x_start, x2, 100)
                        y_curve_fxpb = self.evaluate_f_plus_b(x_curve_fxpb)
                        for x, y in zip(x_curve_fxpb, y_curve_fxpb):
                            if not np.isnan(y) and (x_start <= x <= x2) and (y >= f_a) and (y <= f_a_plus_epsilon):
                                d1_2_verts.append((x, y))
                    d1_2_verts.append((x2, f_a_plus_epsilon))
                    d1_2_verts.append((0, f_a_plus_epsilon))
            else:
                # 単調減少の場合：f(a)-ε から f(a) の間（x軸基準で反転）
                # b > 0のとき：x0を使用（単調増加のb<0と対称）
                # b <= 0のとき：x=aから開始（単調増加のb>=0と対称）
                if self.b > 0 and x0 is not None and y0 is not None:
                    # b > 0のとき：x0から開始
                    d1_2_verts.append((0, f_a))
                    d1_2_verts.append((x0, y0))
                    if x2 > x0:
                        x_curve_fxpb = np.linspace(x0, x2, 100)
                        y_curve_fxpb = self.evaluate_f_plus_b(x_curve_fxpb)
                        for x, y in zip(x_curve_fxpb, y_curve_fxpb):
                            if not np.isnan(y) and (x0 <= x <= x2) and (y <= y0) and (y >= f_a_minus_epsilon):
                                d1_2_verts.append((x, y))
                    d1_2_verts.append((x2, f_a_minus_epsilon))
                    d1_2_verts.append((0, f_a_minus_epsilon))
                else:
                    # b <= 0のとき：x=aから開始（単調増加のb>=0と対称構造）
                    d1_2_verts.append((0, f_a))
                    d1_2_verts.append((self.a, f_a))
                    if x2 > self.a:
                        x_start = self.a
                        x_candidates_start = np.linspace(self.a, x2, 1000)
                        y_candidates_start = self.evaluate_f_plus_b(x_candidates_start)
                        diff_start = y_candidates_start - f_a
                        diff_start_sign = np.sign(diff_start)
                        diff_start_sign[np.isnan(diff_start)] = 0
                        crossings_start = np.where(np.diff(diff_start_sign) != 0)[0]
                        if len(crossings_start) > 0:
                            for i in crossings_start:
                                if i < len(x_candidates_start) - 1:
                                    x1_i, x2_i = x_candidates_start[i], x_candidates_start[i+1]
                                    y1_i, y2_i = y_candidates_start[i], y_candidates_start[i+1]
                                    if not np.isnan(y1_i) and not np.isnan(y2_i) and abs(y2_i - y1_i) > 1e-10:
                                        t = (f_a - y1_i) / (y2_i - y1_i)
                                        x_intersect = x1_i + t * (x2_i - x1_i)
                                        if x_intersect >= self.a and x_intersect <= x2:
                                            x_start = x_intersect
                                            break
                        
                        x_curve_fxpb = np.linspace(x_start, x2, 100)
                        y_curve_fxpb = self.evaluate_f_plus_b(x_curve_fxpb)
                        for x, y in zip(x_curve_fxpb, y_curve_fxpb):
                            if not np.isnan(y) and (x_start <= x <= x2) and (y <= f_a) and (y >= f_a_minus_epsilon):
                                d1_2_verts.append((x, y))
                    d1_2_verts.append((x2, f_a_minus_epsilon))
                    d1_2_verts.append((0, f_a_minus_epsilon))
            
            # D1_2領域を描画
            if len(d1_2_verts) > 2:
                d1_2_path = Path(d1_2_verts)
                d1_2_patch = patches.PathPatch(d1_2_path, facecolor=self.colors['accent'], 
                                              alpha=0.3, edgecolor='none', 
                                              linewidth=0)
                self.ax.add_patch(d1_2_patch)
                # D1_2領域のラベル
                # self.add_region_label(d1_2_verts, 'D1_2', color=self.colors['accent'], position='top_right')

        # D3_1領域の作成（a-δからaまで）- 高精度
        # 説明: δ近傍の左側の領域。x=a-δからx=aまでの範囲で、y=0からy=f(x)までの領域
        # この領域は、δの値に応じてx座標がa-δからaの範囲内にある場合の関数の値を視覚化します
        # 微小なδの場合、より細かいグリッドを使用
        n_points = 1000 if abs(self.delta) < 0.01 else 500
        x_curve_d3_1 = np.linspace(a_minus_delta, self.a, n_points)
        y_curve_d3_1 = self.evaluate_function(x_curve_d3_1)
        d3_1_verts = []
        d3_1_verts.append((a_minus_delta, 0))
        d3_1_verts.append((self.a, 0))
        for i in range(len(x_curve_d3_1)-1, -1, -1):
            x, y = x_curve_d3_1[i], y_curve_d3_1[i]
            # NaNをスキップ、単調増加: y >= 0、単調減少: y <= 0
            if not np.isnan(y):
                if (is_increasing and y >= 0) or (not is_increasing and y <= 0):
                    d3_1_verts.append((x, y))
        if len(d3_1_verts) > 2:
            d3_1_path = Path(d3_1_verts)
            d3_1_patch = patches.PathPatch(d3_1_path, facecolor=self.colors['accent2'], 
                                      alpha=0.4, edgecolor='none', 
                                      linewidth=0)
            self.ax.add_patch(d3_1_patch)
            # D3_1領域のラベル
            # self.add_region_label(d3_1_verts, 'D3_1', color=self.colors['accent2'], position='bottom')

        # D3_2領域の作成（aからa+δまで）- 高精度
        # 説明: δ近傍の右側の領域。x=aからx=a+δまでの範囲で、y=0からy=f(x)+bまでの領域
        # この領域は、δの値に応じてx座標がaからa+δの範囲内にある場合の関数の値を視覚化します
        # 特に、(a,0)と(a,f(a)+b)とy2（(a+δ, y2)）と(a+δ,0)で囲まれた領域
        x_curve_d3_2 = np.linspace(self.a, a_plus_delta, n_points)
        y_curve_d3_2 = self.evaluate_f_plus_b(x_curve_d3_2)  # evaluate_functionではなくevaluate_f_plus_bを使用
        d3_2_verts = []
        d3_2_verts.append((self.a, 0))  # (a, 0) - 左下
        d3_2_verts.append((self.a, f_a + self.b))  # (a, f(a)+b) - 左上
        # f(x)+b曲線に沿って(a+δ, y2)まで（NaNをスキップ）
        for i in range(len(x_curve_d3_2)):
            x, y = x_curve_d3_2[i], y_curve_d3_2[i]
            # 単調増加: y >= 0、単調減少: y <= 0
            if not np.isnan(y):
                if (is_increasing and y >= 0) or (not is_increasing and y <= 0):
                    d3_2_verts.append((x, y))
        d3_2_verts.append((a_plus_delta, 0))  # (a+δ, 0) - 右下
        if len(d3_2_verts) > 2:
            d3_2_path = Path(d3_2_verts)
            d3_2_patch = patches.PathPatch(d3_2_path, facecolor=self.colors['accent2'], 
                                      alpha=0.4, edgecolor='none', 
                                      linewidth=0)
            self.ax.add_patch(d3_2_patch)
            # D3_2領域のラベル
            # self.add_region_label(d3_2_verts, 'D3_2', color=self.colors['accent2'], position='bottom')

        # D4_1, D4_2のy_min, y_maxを定義
        y_min = min(y1, y2)
        y_max = max(y1, y2)

        # D4_1領域の作成
        # 説明: δ近傍の左側における関数の値の範囲
        # 単調増加: y1 から f(a) の間（y1 < f(a)）
        # 単調減少: f(a) から y1 の間（y1 > f(a)）（x軸基準で反転）
        d4_1_verts = []
        
        x_curve_d4_1 = np.linspace(a_minus_delta, self.a, 1000)
        y_curve_d4_1 = self.evaluate_f(x_curve_d4_1)
        
        # y1に最も近い点を見つける
        y_target = y1
        closest_idx = None
        min_diff = float('inf')
        
        for i, (x, y) in enumerate(zip(x_curve_d4_1, y_curve_d4_1)):
            if np.isnan(y):
                continue
            diff = abs(y - y_target)
            if diff < min_diff:
                min_diff = diff
                closest_idx = i
        
        if is_increasing:
            # 単調増加: y1 < f(a) なので、上から下へ
            d4_1_verts.append((0, f_a))
            d4_1_verts.append((self.a, f_a))
            if closest_idx is not None:
                for i in range(len(x_curve_d4_1) - 1, closest_idx - 1, -1):
                    x, y = x_curve_d4_1[i], y_curve_d4_1[i]
                    if not np.isnan(y):
                        d4_1_verts.append((x, y))
                if not np.isnan(y_curve_d4_1[closest_idx]):
                    d4_1_verts.append((x_curve_d4_1[closest_idx], y_curve_d4_1[closest_idx]))
            else:
                d4_1_verts.append((a_minus_delta, y1))
            d4_1_verts.append((0, y1))
        else:
            # 単調減少: y1 > f(a) なので、下から上へ（x軸基準で反転）
            d4_1_verts.append((0, f_a))
            d4_1_verts.append((self.a, f_a))
            if closest_idx is not None:
                for i in range(len(x_curve_d4_1) - 1, closest_idx - 1, -1):
                    x, y = x_curve_d4_1[i], y_curve_d4_1[i]
                    if not np.isnan(y):
                        d4_1_verts.append((x, y))
                if not np.isnan(y_curve_d4_1[closest_idx]):
                    d4_1_verts.append((x_curve_d4_1[closest_idx], y_curve_d4_1[closest_idx]))
            else:
                d4_1_verts.append((a_minus_delta, y1))
            d4_1_verts.append((0, y1))
        
        if len(d4_1_verts) > 2:
            d4_1_path = Path(d4_1_verts)
            d4_1_patch = patches.PathPatch(d4_1_path, facecolor=self.colors['accent2'], 
                                      alpha=0.4, edgecolor='none', 
                                      linewidth=0)
            self.ax.add_patch(d4_1_patch)
            # D4_1領域のラベル
            # self.add_region_label(d4_1_verts, 'D4_1', color=self.colors['accent2'], position='top')

        # D4_2領域の作成
        # 説明: δ近傍の右側における関数の値の範囲
        # 単調増加: f(a)+b から y2 の間（y2 > f(a)+b）
        # 単調減少: y2 から f(a)+b の間（y2 < f(a)+b）（x軸基準で反転）
        d4_2_verts = []
        
        # f(a)+bとy2の値を取得
        f_a_plus_b = f_a + self.b
        
        x_curve_d4_2 = np.linspace(self.a, a_plus_delta, 1000)
        y_curve_d4_2 = self.evaluate_f_plus_b(x_curve_d4_2)
        
        # y2に最も近い点を見つける
        y_target = y2
        closest_idx = None
        min_diff = float('inf')
        
        for i, (x, y) in enumerate(zip(x_curve_d4_2, y_curve_d4_2)):
            if np.isnan(y):
                continue
            diff = abs(y - y_target)
            if diff < min_diff:
                min_diff = diff
                closest_idx = i
        
        if is_increasing:
            # 単調増加: y2 > f(a)+b なので、下から上へ
            d4_2_verts.append((0, f_a_plus_b))
            d4_2_verts.append((self.a, f_a_plus_b))
            if closest_idx is not None:
                for i in range(closest_idx + 1):
                    x, y = x_curve_d4_2[i], y_curve_d4_2[i]
                    if not np.isnan(y) and ((f_a_plus_b <= y <= y2) or (y2 <= y <= f_a_plus_b)):
                        d4_2_verts.append((x, y))
                if closest_idx < len(x_curve_d4_2) and not np.isnan(y_curve_d4_2[closest_idx]):
                    d4_2_verts.append((x_curve_d4_2[closest_idx], y_curve_d4_2[closest_idx]))
            else:
                d4_2_verts.append((a_plus_delta, y2))
            d4_2_verts.append((0, y2))
        else:
            # 単調減少: y2 < f(a)+b なので、上から下へ（x軸基準で反転）
            d4_2_verts.append((0, f_a_plus_b))
            d4_2_verts.append((self.a, f_a_plus_b))
            if closest_idx is not None:
                for i in range(closest_idx + 1):
                    x, y = x_curve_d4_2[i], y_curve_d4_2[i]
                    if not np.isnan(y) and ((f_a_plus_b <= y <= y2) or (y2 <= y <= f_a_plus_b)):
                        d4_2_verts.append((x, y))
                if closest_idx < len(x_curve_d4_2) and not np.isnan(y_curve_d4_2[closest_idx]):
                    d4_2_verts.append((x_curve_d4_2[closest_idx], y_curve_d4_2[closest_idx]))
            else:
                d4_2_verts.append((a_plus_delta, y2))
            d4_2_verts.append((0, y2))
        
        if len(d4_2_verts) > 2:
            d4_2_path = Path(d4_2_verts)
            d4_2_patch = patches.PathPatch(d4_2_path, facecolor=self.colors['accent2'], 
                                      alpha=0.4, edgecolor='none', 
                                      linewidth=0)
            self.ax.add_patch(d4_2_patch)
            # D4_2領域のラベル
            # self.add_region_label(d4_2_verts, 'D4_2', color=self.colors['accent2'], position='top')

        # ε近傍の描画
        # self.ax.axhspan(f_a_minus_epsilon, f_a_plus_epsilon, color='pink', alpha=0.3)
        
        # δ近傍の帯状領域
        # self.ax.axvspan(self.a - self.delta, self.a + self.delta, color='purple', alpha=0.3)
        
        # プレミアム中心点の描画
        self.ax.plot(self.a, f_a, 'o', color=self.colors['primary'], markersize=5, 
                    label=f'({self.a:.2f}, {f_a:.2f})', zorder=12)
        
        # プレミアムy1とy2の点を描画 (非表示)
        # self.ax.plot(a_minus_delta, y1, 'o', color=self.colors['accent2'], markersize=8, 
        #             markeredgecolor='white', markeredgewidth=2,
        #             label=f'y1({a_minus_delta:.2f}, {y1:.2f})', zorder=11)
        # # y1の点の近くに文字ラベルを追加
        ylim = self.ax.get_ylim()
        y_range = ylim[1] - ylim[0]
        # self.ax.text(a_minus_delta, y1 + y_range * 0.03, 'y1', 
        #             fontsize=11, fontweight='bold', color=self.colors['accent2'],
        #             ha='center', va='bottom', zorder=15,
        #             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
        #                     edgecolor='none', alpha=0.8))
        
        # self.ax.plot(a_plus_delta, y2, 'o', color=self.colors['accent2'], markersize=8, 
        #             markeredgecolor='white', markeredgewidth=2,
        #             label=f'y2({a_plus_delta:.2f}, {y2:.2f})', zorder=11)
        # # y2の点の近くに文字ラベルを追加
        # self.ax.text(a_plus_delta, y2 + y_range * 0.03, 'y2', 
        #             fontsize=11, fontweight='bold', color=self.colors['accent2'],
        #             ha='center', va='bottom', zorder=15,
        #             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
        #                     edgecolor='none', alpha=0.8))
        
        # プレミアム垂線 (非表示)
        # self.ax.plot([a_minus_delta, a_minus_delta], [0, y1], '--', 
        #             color=self.colors['accent2'], alpha=0.9, linewidth=2.5, zorder=6)
        # self.ax.plot([a_plus_delta, a_plus_delta], [0, y2], '--', 
        #             color=self.colors['accent2'], alpha=0.9, linewidth=2.5, zorder=6)
        
        # プレミアムx1とx2の点を描画 (非表示)
        # x_rangeを事前に計算（x1とx2の両方で使用）
        xlim = self.ax.get_xlim()
        x_range = xlim[1] - xlim[0]
        
        # if x1 is not None:
        #     self.ax.plot(x1, target_y_for_x1, 'o', color=self.colors['accent'], 
        #                 markersize=8, markeredgecolor='white', markeredgewidth=2,
        #                 label=f'x1({x1:.2f}, {target_y_for_x1:.2f})', zorder=11)
        #     # x1の点の近くに文字ラベルを追加
        #     self.ax.text(x1 + x_range * 0.02, target_y_for_x1, 'x1', 
        #                 fontsize=11, fontweight='bold', color=self.colors['accent'],
        #                 ha='left', va='center', zorder=15,
        #                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
        #                         edgecolor='none', alpha=0.8))
        #     # x1からy軸への垂線
        #     self.ax.plot([x1, x1], [0, target_y_for_x1], '--', 
        #                 color=self.colors['accent'], alpha=0.9, linewidth=2.5, zorder=6)
        
        # if x2 is not None:
        #     # x2の点を描画
        #     self.ax.plot(x2, target_y_for_x2, 'o', color=self.colors['accent'], 
        #                 markersize=8, markeredgecolor='white', markeredgewidth=2,
        #                 label=f'x2({x2:.2f}, {target_y_for_x2:.2f})', zorder=11)
        #     # x2の点の近くに文字ラベルを追加
        #     self.ax.text(x2 + x_range * 0.02, target_y_for_x2, 'x2', 
        #                 fontsize=11, fontweight='bold', color=self.colors['accent'],
        #                 ha='left', va='center', zorder=15,
        #                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
        #                         edgecolor='none', alpha=0.8))
        #     # x2からy軸への垂線
        #     self.ax.plot([x2, x2], [0, target_y_for_x2], '--', 
        #                 color=self.colors['accent'], alpha=0.9, linewidth=2.5, zorder=6)
        
        # 水平線の描画
        # self.ax.axhline(y=f_a_plus_epsilon, color='red', linestyle='--', alpha=0.7, label=f'y = f(a) + ε = {f_a_plus_epsilon:.2f}')
        # self.ax.axhline(y=f_a_minus_epsilon, color='red', linestyle='--', alpha=0.7, label=f'y = f(a) - ε = {f_a_minus_epsilon:.2f}')
        
        # y=y1とy=y2の水平線を描画
        # self.ax.axhline(y=y1, color='blue', linestyle='--', alpha=0.7, label=f'y = y1 = {y1:.2f}')
        # self.ax.axhline(y=y2, color='blue', linestyle='--', alpha=0.7, label=f'y = y2 = {y2:.2f}')
        
        # 軸を再描画（update内で確実に表示）
        self.draw_axes()

        self.ax.set_aspect("equal", adjustable="box")

        # 原点Oを表示
        self.ax.text(0, 0, 'O', fontsize=14, fontweight='bold', 
                    color=self.colors['primary'], ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='none', alpha=0.8), zorder=15)
        
        # グラフの更新
        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    visualizer = EpsilonDeltaVisualizer()
    if sys.platform == "emscripten":
        # 参照を保持（GC 対策）＋ 非ブロッキング表示でブラウザのイベントループに委ねる
        try:
            import js

            js.window.epsilonDeltaVisualizer = visualizer
        except Exception:
            pass
        plt.show(block=False)
        visualizer.fig.canvas.draw_idle()
    else:
        plt.show()