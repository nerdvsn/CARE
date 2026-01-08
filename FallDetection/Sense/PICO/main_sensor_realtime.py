import machine
import time
import json
import struct
from machine import I2C, Pin

# ==================== KONFIGURATION ====================
# I2C für MLX90640
I2C_SCL_PIN = 5  # GPIO 5, Pin 7
I2C_SDA_PIN = 4  # GPIO 4, Pin 6

# UART für Kommunikation mit Laptop
UART_TX_PIN = 0  # GPIO 0, Pin 1
UART_RX_PIN = 1  # GPIO 1, Pin 2
UART_BAUDRATE = 115200 #921600  # WICHTIG: Muss mit Laptop übereinstimmen!

# MLX90640 I2C Adresse
MLX90640_ADDRESS = 0x33

# ==================== INITIALISIERUNG ====================
print("=== MLX90640 mit Raspberry Pi Pico W ===")
print(f"Baudrate: {UART_BAUDRATE}")

# I2C initialisieren
try:
    i2c = I2C(0, scl=Pin(I2C_SCL_PIN), sda=Pin(I2C_SDA_PIN), freq=400000)
    print(f"I2C initialisiert: SCL=GP{I2C_SCL_PIN}, SDA=GP{I2C_SDA_PIN}")
except Exception as e:
    print(f"Fehler bei I2C Initialisierung: {e}")
    i2c = None

# UART initialisieren
try:
    uart = machine.UART(0, baudrate=UART_BAUDRATE, tx=Pin(UART_TX_PIN), rx=Pin(UART_RX_PIN))
    print(f"UART initialisiert: TX=GP{UART_TX_PIN}, RX=GP{UART_RX_PIN}")
    # Kleine Pause für UART Stabilisierung
    time.sleep(0.5)
except Exception as e:
    print(f"Fehler bei UART Initialisierung: {e}")
    uart = None

# LED für Status
try:
    led = machine.Pin("LED", machine.Pin.OUT)
    print("LED initialisiert")
except:
    try:
        led = machine.Pin(25, machine.Pin.OUT)
        print("LED auf Pin 25 initialisiert")
    except:
        led = None
        print("Keine LED gefunden")

# ==================== MLX90640 FUNKTIONEN ====================
def check_mlx90640():
    """Prüft ob MLX90640 angeschlossen ist"""
    if i2c is None:
        return False
    
    try:
        devices = i2c.scan()
        print(f"Gefundene I2C Geräte: {[hex(addr) for addr in devices]}")
        
        if MLX90640_ADDRESS in devices:
            print(f"✅ MLX90640 gefunden bei {hex(MLX90640_ADDRESS)}")
            return True
        else:
            print(f"❌ MLX90640 NICHT gefunden bei {hex(MLX90640_ADDRESS)}")
            print("   Bitte Verkabelung prüfen:")
            print(f"   MLX90640 VIN  → Pico 3.3V (Pin 36)")
            print(f"   MLX90640 GND  → Pico GND (Pin 38)")
            print(f"   MLX90640 SCL  → Pico GPIO {I2C_SCL_PIN} (Pin 7)")
            print(f"   MLX90640 SDA  → Pico GPIO {I2C_SDA_PIN} (Pin 6)")
            return False
    except Exception as e:
        print(f"Fehler bei I2C Scan: {e}")
        return False

def read_mlx90640_dummy():
    """Simulierte MLX90640 Daten für Test"""
    # Erstelle 32x24 Temperaturmatrix (768 Werte)
    base_temp = 25.0
    temp_data = []
    
    for row in range(24):  # 24 Zeilen
        for col in range(32):  # 32 Spalten
            # Simuliere ein Muster
            value = base_temp + (col * 0.2) + (row * 0.1) + (time.time() % 5) * 0.1
            temp_data.append(round(value, 2))
    
    ambient_temp = 23.5 + (time.time() % 10) * 0.1
    
    return temp_data, round(ambient_temp, 2)

def read_mlx90640_register(register, length):
    """Liest Register vom MLX90640"""
    if i2c is None:
        return None
    
    try:
        data = i2c.readfrom_mem(MLX90640_ADDRESS, register, length)
        return data
    except Exception as e:
        print(f"Fehler beim Lesen Register {register}: {e}")
        return None

def send_data_over_uart(temperature_data, ambient_temp):
    """Sendet Daten im korrekten Format an Laptop"""
    if uart is None:
        print("UART nicht verfügbar")
        return False
    
    try:
        # Format muss exakt sein für preprocess_temperature_data() auf Laptop:
        # {"temperature": [768 Werte], "at": ambient_temp}
        data_dict = {
            "temperature": temperature_data,
            "at": ambient_temp
        }
        
        # Konvertiere zu String und sende
        json_str = json.dumps(data_dict)
        uart.write(json_str + '\n')
        
        return True
        
    except Exception as e:
        print(f"Fehler beim Senden über UART: {e}")
        return False

# ==================== HAUPTPROGRAMM ====================
def main():
    print("\n" + "="*50)
    print("Starte Hauptprogramm...")
    
    # Prüfe MLX90640
    mlx_connected = check_mlx90640()
    
    if not mlx_connected:
        print("\n⚠  MLX90640 nicht gefunden! Sende Testdaten...")
        use_dummy_data = True
    else:
        print("\n✅ MLX90640 verbunden. Starte echte Datenerfassung...")
        use_dummy_data = False
        
        # Hier würdest du die echte MLX90640 Bibliothek laden
        # import mlx90640
        # sensor = mlx90640.MLX90640(i2c)
    
    counter = 0
    last_send_time = time.time()
    
    print("\nSende Daten an Laptop...")
    print("Format: {'temperature': [768 Werte], 'at': ambient_temp}")
    print(f"Baudrate: {UART_BAUDRATE}")
    print("Drücke RESET zum Stoppen")
    print("-" * 50)
    
    try:
        while True:
            # LED blinken lassen
            if led:
                led.toggle()
            
            # Temperaturdaten lesen
            if use_dummy_data:
                temperature_data, ambient_temp = read_mlx90640_dummy()
            else:
                # Hier echte MLX90640 Daten lesen
                # temperature_frame, ambient_temp = sensor.getFrame()
                # temperature_data = temperature_frame.tolist()
                temperature_data, ambient_temp = read_mlx90640_dummy()  # Vorläufig
            
            # Daten über UART senden
            if send_data_over_uart(temperature_data, ambient_temp):
                counter += 1
                
                # Status alle 10 Pakete anzeigen
                if counter % 10 == 0:
                    current_time = time.time()
                    elapsed = current_time - last_send_time
                    fps = 10 / elapsed if elapsed > 0 else 0
                    
                    print(f"Paket {counter}: {len(temperature_data)} Werte, "
                          f"Ambient={ambient_temp}°C, "
                          f"FPS={fps:.1f}")
                    
                    last_send_time = current_time
            
            # Wartezeit für gewünschte Framerate (z.B. 8 FPS = 0.125s)
            time.sleep(0.125)
            
    except KeyboardInterrupt:
        print("\nProgramm durch Benutzer gestoppt")
    except Exception as e:
        print(f"\nFehler im Hauptprogramm: {e}")

# ==================== START ====================
if __name__ == "__main__":
    # Kurze Pause für USB Initialisierung
    time.sleep(1.0)
    
    # Starte Hauptprogramm
    main()
    
    # Programmende
    if led:
        led.off()
    print("\n=== Programm beendet ===")