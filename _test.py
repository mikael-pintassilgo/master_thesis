import ctypes

try:
    ctypes.windll.user32.LoadCursorW.restype = ctypes.c_void_p
    IDC_ARROW = 32512  # Standard arrow cursor
    hCursor = ctypes.windll.user32.LoadCursorW(0, IDC_ARROW)
    ctypes.windll.user32.SetSystemCursor(hCursor, 32512)  # 32512 = OCR_NORMAL
except Exception as e:
    print(f"Could not reset cursor: {e}")