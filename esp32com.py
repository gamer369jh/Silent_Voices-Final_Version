import socket

def send_char_to_esp32(char_to_send,esp32_ip, esp32_port=80 ):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((esp32_ip, esp32_port))
    except OSError as e:
        print("Error connecting to the ESP32:", e)
        return

    try:
        sock.send(char_to_send.encode('utf-8'))
        print(f"Sent '{char_to_send}' to ESP32")
    except OSError as e:
        print("Error sending data to the ESP32:", e)
    finally:
        sock.close()
