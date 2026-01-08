#!/usr/bin/env python3
"""
Echtzeit-Visualisierung für Sturzerkennung
Zeigt Thermal-Bild und Erkennungsstatus
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from fall_detector import FallDetector, SERIAL_PORT, BAUD_RATE, MODEL_PATH
from pathlib import Path
import threading
import queue

# ========== CUSTOM THERMAL COLORMAP ==========
# Klassisches Thermal-Farbschema: Schwarz → Blau → Rot → Gelb → Weiß
thermal_colors = [
    (0.0, 0.0, 0.0),      # Schwarz (kalt)
    (0.0, 0.0, 0.5),      # Dunkelblau
    (0.0, 0.0, 1.0),      # Blau
    (0.5, 0.0, 0.5),      # Lila
    (1.0, 0.0, 0.0),      # Rot
    (1.0, 0.5, 0.0),      # Orange
    (1.0, 1.0, 0.0),      # Gelb
    (1.0, 1.0, 1.0),      # Weiß (heiß)
]
thermal_cmap = LinearSegmentedColormap.from_list("thermal", thermal_colors, N=256)


class RealtimeVisualizer:
    def __init__(self, detector: FallDetector):
        self.detector = detector
        self.data_queue = queue.Queue(maxsize=10)
        
        # Aktueller Zustand
        self.current_frame = np.zeros((24, 32))
        self.current_prediction = 0
        self.current_confidence = 0.0
        self.current_ambient = 25.0
        self.frame_count = 0
        
        # Setup Plot
        self.setup_plot()
    
    def setup_plot(self):
        """Erstellt das Matplotlib-Fenster"""
        plt.style.use('dark_background')
        
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.patch.set_facecolor('#1a1a2e')
        
        # Grid: 2 Spalten
        gs = self.fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[3, 1],
                                    hspace=0.3, wspace=0.2)
        
        # Haupt-Thermal-Bild
        self.ax_thermal = self.fig.add_subplot(gs[0, 0])
        self.ax_thermal.set_title("MLX90640 Thermal Image (32×24)", 
                                   fontsize=14, fontweight='bold', color='white')
        self.ax_thermal.set_xticks([])
        self.ax_thermal.set_yticks([])
        
        self.thermal_img = self.ax_thermal.imshow(
            self.current_frame,
            cmap=thermal_cmap,
            vmin=20, vmax=40,
            interpolation='bicubic',
            aspect='auto'
        )
        
        # Colorbar
        cbar = self.fig.colorbar(self.thermal_img, ax=self.ax_thermal, 
                                  orientation='horizontal', pad=0.08, shrink=0.8)
        cbar.set_label('Temperatur (°C)', color='white')
        cbar.ax.xaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax, 'xticklabels'), color='white')
        
        # Status-Panel
        self.ax_status = self.fig.add_subplot(gs[0, 1])
        self.ax_status.set_xlim(0, 1)
        self.ax_status.set_ylim(0, 1)
        self.ax_status.set_xticks([])
        self.ax_status.set_yticks([])
        self.ax_status.set_facecolor('#1a1a2e')
        for spine in self.ax_status.spines.values():
            spine.set_visible(False)
        
        # Status-Texte
        self.status_text = self.ax_status.text(
            0.5, 0.75, "NORMAL",
            ha='center', va='center',
            fontsize=32, fontweight='bold',
            color='#00ff88',
            transform=self.ax_status.transAxes
        )
        
        self.confidence_text = self.ax_status.text(
            0.5, 0.55, "Confidence: 0%",
            ha='center', va='center',
            fontsize=14,
            color='white',
            transform=self.ax_status.transAxes
        )
        
        self.ambient_text = self.ax_status.text(
            0.5, 0.35, "Ambient: --°C",
            ha='center', va='center',
            fontsize=12,
            color='#888888',
            transform=self.ax_status.transAxes
        )
        
        self.frame_text = self.ax_status.text(
            0.5, 0.15, "Frame: 0",
            ha='center', va='center',
            fontsize=10,
            color='#666666',
            transform=self.ax_status.transAxes
        )
        
        # Histogramm
        self.ax_hist = self.fig.add_subplot(gs[1, :])
        self.ax_hist.set_title("Temperatur-Verteilung", fontsize=10, color='white')
        self.ax_hist.set_xlabel("Temperatur (°C)", fontsize=9, color='white')
        self.ax_hist.set_ylabel("Pixel", fontsize=9, color='white')
        self.ax_hist.tick_params(colors='white')
        self.ax_hist.set_facecolor('#1a1a2e')
        
        # Initiales Histogramm
        self.hist_bars = self.ax_hist.bar(
            range(20), [0]*20,
            color='#4a90d9', edgecolor='none', alpha=0.7
        )
        
        self.fig.suptitle("🔥 Echtzeit-Sturzerkennung", 
                          fontsize=16, fontweight='bold', color='white', y=0.98)
        
        plt.tight_layout()
    
    def update_callback(self, frame, prediction, confidence, ambient):
        """Callback vom Detector - Thread-safe"""
        try:
            self.data_queue.put_nowait({
                'frame': frame,
                'prediction': prediction,
                'confidence': confidence,
                'ambient': ambient
            })
        except queue.Full:
            pass  # Überspringe wenn Queue voll
    
    def animation_update(self, frame_num):
        """Update-Funktion für Animation"""
        # Hole neueste Daten
        while not self.data_queue.empty():
            try:
                data = self.data_queue.get_nowait()
                self.current_frame = data['frame']
                self.current_prediction = data['prediction']
                self.current_confidence = data['confidence']
                self.current_ambient = data['ambient']
                self.frame_count += 1
            except queue.Empty:
                break
        
        # Update Thermal-Bild
        self.thermal_img.set_array(self.current_frame)
        
        # Auto-Scale basierend auf aktuellen Daten
        vmin = max(15, self.current_frame.min() - 2)
        vmax = min(45, self.current_frame.max() + 2)
        self.thermal_img.set_clim(vmin, vmax)
        
        # Update Status
        if self.current_prediction == 1:
            self.status_text.set_text("⚠️ STURZ!")
            self.status_text.set_color('#ff4444')
            self.ax_status.set_facecolor('#3a1a1a')
        else:
            self.status_text.set_text("✓ NORMAL")
            self.status_text.set_color('#00ff88')
            self.ax_status.set_facecolor('#1a1a2e')
        
        self.confidence_text.set_text(f"Confidence: {self.current_confidence:.1%}")
        self.ambient_text.set_text(f"Ambient: {self.current_ambient:.1f}°C")
        self.frame_text.set_text(f"Frame: {self.frame_count}")
        
        # Update Histogramm
        if self.current_frame.any():
            hist, bins = np.histogram(self.current_frame.flatten(), bins=20, range=(15, 45))
            for bar, h in zip(self.hist_bars, hist):
                bar.set_height(h)
            self.ax_hist.set_ylim(0, max(hist) * 1.1 + 1)
        
        return [self.thermal_img, self.status_text, self.confidence_text, 
                self.ambient_text, self.frame_text] + list(self.hist_bars)
    
    def run(self):
        """Startet Visualisierung und Detector"""
        # Detector in separatem Thread
        detector_thread = threading.Thread(
            target=self.detector.run,
            kwargs={'callback': self.update_callback},
            daemon=True
        )
        detector_thread.start()
        
        # Animation starten
        self.anim = FuncAnimation(
            self.fig,
            self.animation_update,
            interval=50,  # 20 FPS
            blit=False,
            cache_frame_data=False
        )
        
        plt.show()


# ========== MAIN ==========

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Echtzeit-Visualisierung Sturzerkennung")
    parser.add_argument("--port", default=SERIAL_PORT, help="Serial Port")
    parser.add_argument("--baud", type=int, default=BAUD_RATE, help="Baudrate")
    parser.add_argument("--model", default=str(MODEL_PATH), help="Pfad zum Modell")
    
    args = parser.parse_args()
    
    print("="*50)
    print("  Sturzerkennung - Echtzeit-Visualisierung")
    print("="*50)
    print(f"  Port:  {args.port}")
    print(f"  Baud:  {args.baud}")
    print(f"  Model: {args.model}")
    print("="*50 + "\n")
    
    detector = FallDetector(
        model_path=Path(args.model),
        port=args.port,
        baudrate=args.baud
    )
    
    viz = RealtimeVisualizer(detector)
    viz.run()
