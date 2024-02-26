import win32api as wapi
import time

# This list contains all the keys we want to check
keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\":
    keyList.append(char)

# Additional keys for left and right arrow keys
keyList.extend(["left", "right"])

def key_check():
    keys = []
    for key in keyList:
        if key == "left":
            if wapi.GetAsyncKeyState(0x25):  # VK_LEFT
                keys.append(key)
        elif key == "right":
            if wapi.GetAsyncKeyState(0x27):  # VK_RIGHT
                keys.append(key)
        elif wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys
