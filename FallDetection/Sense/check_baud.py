import serial
import serial.tools.list_ports

# WICHTIG: Ersetze das hier durch deinen Port, falls anders
PORT = '/dev/ttyUSB0'

# Die Geschwindigkeiten, die wir testen wollen (Standard bis High-Speed)
RATES_TO_TEST = [
    115200, 
    460800, 
    921600, 
    1000000, # 1 Mbit
    1500000, # 1.5 Mbit
    2000000, # 2 Mbit (Das Ziel!)
    3000000, # 3 Mbit
    4000000  # 4 Mbit
]

print(f"--- Prüfe Baudraten für {PORT} ---")

for baud in RATES_TO_TEST:
    try:
        # Versuchen, den Port zu öffnen
        ser = serial.Serial(PORT, baud, timeout=1)
        
        # Sicherheits-Check: Hat der Treiber die Rate wirklich gesetzt?
        settings = ser.get_settings()
        actual = settings['baudrate']
        
        if actual == baud:
            print(f"{baud:7} Baud: VERFÜGBAR")
        else:
            print(f"{baud:7} Baud: Treiber erzwingt stattdessen {actual}")
            
        ser.close()
    except Exception as e:
        print(f"{baud:7} Baud: NICHT MÖGLICH ({e})")