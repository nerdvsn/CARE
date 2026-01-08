import time
import board
import busio
import numpy as np
import matplotlib.pyplot as plt
import adafruit_mlx90640

class ThermalSensor:
    def __init__(self):
        self.i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
        self.mlx = adafruit_mlx90640.MLX90640(self.i2c)
        self.mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ
        self.frame = np.zeros((24*32,))
        
    def get_frame(self):
        """Holt einen Frame vom Sensor"""
        try:
            self.mlx.getFrame(self.frame)
            frame_2d = np.reshape(self.frame, (24, 32))
            return frame_2d
        except Exception as e:
            print(f"Error reading frame: {e}")
            return None
    
    def get_ambient_temperature(self):
        """Gibt die Umgebungstemperatur zurück"""
        try:
            # Der MLX90640 hat keine direkte Ambient-Temperatur-Funktion
            # Wir nehmen den Durchschnitt der Randpixel
            frame = self.get_frame()
            if frame is not None:
                edge_pixels = np.concatenate([
                    frame[0, :],    # Erste Zeile
                    frame[-1, :],   # Letzte Zeile  
                    frame[:, 0],    # Erste Spalte
                    frame[:, -1]    # Letzte Spalte
                ])
                return np.median(edge_pixels)
            return 25.0  # Fallback
        except:
            return 25.0

def test_sensor():
    """Testet den Sensor und zeigt Live-Daten"""
    sensor = ThermalSensor()
    
    print("Thermal Sensor Test - Drücke Ctrl+C zum Beenden")
    
    try:
        while True:
            frame = sensor.get_frame()
            ambient = sensor.get_ambient_temperature()
            
            if frame is not None:
                print(f"Ambient: {ambient:.1f}°C | Frame Min: {np.min(frame):.1f}°C | Max: {np.max(frame):.1f}°C")
                
                # Zeige eine einfache ASCII-Repräsentation
                small_frame = frame[::2, ::2]  # Downsample für ASCII
                for row in small_frame:
                    line = "".join(["*" if temp > 30 else "." for temp in row])
                    print(line)
                print("-" * 40)
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nTest beendet")

if __name__ == "__main__":
    test_sensor()