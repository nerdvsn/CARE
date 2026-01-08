import machine
import time
import json
from machine import I2C, Pin

# I2C für MLX90640
i2c = I2C(0, scl=Pin(5), sda=Pin(4), freq=400000)

# UART für Laptop
uart = machine.UART(0, baudrate=115200, tx=Pin(0), rx=Pin(1))

def check_mlx90640():
    """Prüft ob MLX90640 angeschlossen ist"""
    devices = i2c.scan()
    print("Gefundene I2C Geräte:", [hex(addr) for addr in devices])
    
    mlx_addr = 0x33  # Standardadresse MLX90640
    if mlx_addr in devices:
        print(f"✅ MLX90640 gefunden bei {hex(mlx_addr)}")
        return True
    else:
        print(f"❌ MLX90640 nicht gefunden!")
        print("   Bitte Verkabelung prüfen: SDA->GP4, SCL->GP5")
        return False

# Test MLX90640
if check_mlx90640():
    print("MLX90640 bereit für Datenerfassung")
    
    # Simulierte Daten (später echte MLX90640 Daten)
    while True:
        # Simulierte Temperaturmatrix (32x24 = 768 Werte)
        temp_data = [25.0 + (i % 32) * 0.1 for i in range(768)]
        ambient_temp = 23.5
        
        # Daten als JSON für Laptop
        data_packet = {
            "temperature": temp_data,
            "at": ambient_temp
        }
        
        # Über UART senden
        json_str = json.dumps(data_packet)
        uart.write(json_str + '\n')
        print(f"📤 Gesendet: {len(temp_data)} Temperaturwerte")
        
        time.sleep(0.5)