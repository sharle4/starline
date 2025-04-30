import mss
import numpy as np
import cv2
from window_finder import find_emulator_window

def capture_window(window):
    """Capture le contenu de la fenêtre spécifiée."""
    if not window or not window.isActive :
         window = find_emulator_window()
         if not window : return None, None

    monitor = {
        "top": window.top,
        "left": window.left,
        "width": window.width,
        "height": window.height,
        "mon": 1
    }

    with mss.mss() as sct:
        try:
            sct_img = sct.grab(monitor)
            img = np.array(sct_img)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return img_bgr, monitor
        
        except mss.ScreenShotError as e:
            print(f"Erreur de capture d'écran : {e}")
            new_window = find_emulator_window()
            if new_window and (new_window.left != window.left or new_window.top != window.top):
                 print("La fenêtre semble avoir été déplacée, tentative de recapturer...")
                 window = new_window
            return None, None
        
        except Exception as e:
             print(f"Erreur générale de capture: {e}")
             return None, None


if __name__ == "__main__":
    emulator_win = find_emulator_window()
    if emulator_win:
        while True:
            frame, _ = capture_window(emulator_win)
            if frame is not None:
                cv2.imshow("Capture Emulator", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("Erreur de capture, arrêt.")
                break
        cv2.destroyAllWindows()