import mss
import numpy as np
import cv2
from window_finder import find_emulator_window

TEMPLATE_GRAY = None
W, H = None, None

def load_puck():
    """Charge le template du palet."""
    global TEMPLATE_GRAY, W, H
    
    try:
        template = cv2.imread('puck_template.png', cv2.IMREAD_COLOR)
        if template is None:
            raise FileNotFoundError("Template non trouvé ou invalide.")
        TEMPLATE_GRAY = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        W, H = TEMPLATE_GRAY.shape[::-1]
        print(f"Template chargé : {W}x{H}")
        
    except Exception as e:
        print(f"Erreur chargement template: {e}. Assurez-vous d'avoir un fichier 'puck_template.png'.")
        exit()
        

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

def find_puck(frame):
    """Trouve le palet en utilisant template matching."""
    if frame is None: return None

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(frame_gray, TEMPLATE_GRAY, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    threshold = 0.65
    if max_val >= threshold:
        top_left = max_loc
        center_x = top_left[0] + W // 2
        center_y = top_left[1] + H // 2
        print(f"Palet trouvé à ({center_x}, {center_y}) avec confiance {max_val:.2f}")
        return (center_x, center_y)
    else:
        print(f"Palet non trouvé (Confiance max: {max_val:.2f} < {threshold})")
        return None


if __name__ == "__main__":
    emulator_win = find_emulator_window()
    load_puck()
    if emulator_win:
        while True:
            frame, _ = capture_window(emulator_win)
            
            if frame is not None:
                cv2.imshow("Capture Emulator", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
                puck_pos = find_puck(frame)
                if puck_pos:
                    cv2.circle(frame, puck_pos, 15, (0, 255, 0), 2)
                cv2.imshow("Capture Emulator", frame)
                
            else:
                print("Erreur de capture, arrêt.")
                break
        cv2.destroyAllWindows()