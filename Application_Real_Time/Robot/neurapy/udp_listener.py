import socket
import json
import time
import sys
if sys.platform == "darwin":
    from neurapy.robot_mac import Robot
else:
    from neurapy.robot import Robot
from neurapy.tictactoe import TicTacToeGame

def udp_listener():
    UDP_IP = "127.0.0.1"
    UDP_PORT = 9001   # Must match DEST_UDP_PORT in your udp-ws-bridge in GUI
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"NeuraPy TicTacToe UDP listener running on {UDP_IP}:{UDP_PORT}...")

    # Initialize the robot and tictactoe game.
    r = Robot() # TODO check if this is correct for robot
    r.set_mode("Automatic")
    time.sleep(1)  # Allow time for the robot to switch modes.
    game = TicTacToeGame(r)

    while True:
        data, addr = sock.recvfrom(1024)
        try:
            # Try to decode as JSON first.
            msg = json.loads(data.decode().strip())
            field_id = msg.get("fieldId")
        except json.JSONDecodeError:
            # If not JSON, assume it's a plain integer string.
            try:
                field_id = int(data.decode().strip())
            except ValueError:
                print("Received invalid message:", data)
                continue

        if field_id is not None:
            print(f"Received command to draw circle in field {field_id}")
            if game.draw_field(field_id):
                print("Circle drawn successfully!")
            else:
                print("Drawing failed for field", field_id)
        time.sleep(0.5)  # Optional: small delay to avoid rapid re-triggering

if __name__ == "__main__":
    udp_listener()