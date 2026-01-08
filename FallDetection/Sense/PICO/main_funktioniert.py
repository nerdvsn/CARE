import machine
import time
import struct

# Wir nutzen UART 0 auf Pin 0 (TX) und Pin 1 (RX)
# WICHTIG: Baudrate ist jetzt 115200 für Stabilität
uart = machine.UART(0, baudrate=115200, tx=machine.Pin(0), rx=machine.Pin(1))

# LED Setup (optional, falls Pico W eine interne hat, oft "LED" oder Pin 25)
try:
    led = machine.Pin("LED", machine.Pin.OUT)
except:
    try:
        led = machine.Pin(25, machine.Pin.OUT)
    except:
        led = None

print("START: Pico sendet jetzt auf UART0 mit 115200 Baud...")

counter = 0

while True:
    # 1. Wir senden einfachen Text
    msg = f"Datenpaket {counter}"
    print(f"Sende: {msg}") # Ausgabe in der Pico-Konsole (USB)
    
    # Wichtig: Ein '\n' (Zeilenumbruch) am Ende, damit readline() funktioniert
    uart.write(msg + '\n')
    
    # LED blinken lassen als Herzschlag
    if led:
        led.toggle()
        
    counter += 1
    time.sleep(1) # 1 Sekunde warten