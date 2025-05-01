import mss
import numpy as np
import cv2
import time
from pynput.mouse import Listener, Button
from window_finder import find_emulator_window
from overlay_class import TrajectoryOverlay
from arrow_direction import find_arrow_direction


TEMPLATE_GRAY = None
W, H = None, None
OVERLAY = None
LAST_CLICK_POS = None
CURRENT_WIN = None


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


def on_mouse_click(x, y, button, pressed):
    """Capture les clics de souris et convertit les coordonnées globales en coordonnées relatives à la fenêtre."""
    global LAST_CLICK_POS, CURRENT_WIN
    
    if pressed and button == Button.left:
        if CURRENT_WIN and CURRENT_WIN.isActive:
            relative_x = x - CURRENT_WIN.left
            relative_y = y - CURRENT_WIN.top
            
            if 0 <= relative_x < CURRENT_WIN.width and 0 <= relative_y < CURRENT_WIN.height:
                LAST_CLICK_POS = (relative_x, relative_y)
                print(f"Clic enregistré à: {LAST_CLICK_POS}")
    return True


def find_puck(frame, click_pos=None):
    """Retourne (x,y,r) du palet ou None."""
    if frame is None:
        return None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    bas  = np.array([100, 150, 100])
    haut = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, bas, haut)

    mask = cv2.medianBlur(mask, 5)
    mask = cv2.erode (mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)

    hough = cv2.HoughCircles(mask,
                             cv2.HOUGH_GRADIENT,
                             dp=1.1,
                             minDist=40,
                             param1=120,
                             param2=15,
                             minRadius=18,
                             maxRadius=30)

    if hough is not None:
        hough = np.uint16(np.around(hough[0]))
        if click_pos is not None:
            cx, cy = click_pos
            hough = sorted(hough, key=lambda c: (c[0]-cx)**2 + (c[1]-cy)**2)
        x, y, r = hough[0]
        print(f"Palet trouvé : ({x}, {y}), rayon : {r}")
        return (x, y, r)

    print("Aucun palet trouvé.")
    return None


def toggle_overlay(state=[True]):
    """Active ou désactive l'overlay."""
    state[0] = not state[0]
    OVERLAY.root.attributes("-topmost", state[0])
    if state[0]:
        OVERLAY._make_click_through()   # re-active click-through
    else:
        # enlève le style transparent pour pouvoir interagir
        import win32gui, win32con
        hwnd = OVERLAY.root.winfo_id()
        style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        style &= ~win32con.WS_EX_TRANSPARENT
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, style)
        

def __init__():
    """Initialise l'application."""
    global OVERLAY
    load_puck()
    emulator_win = find_emulator_window()
    
    if emulator_win:
        initial_monitor_info = {
            "top": emulator_win.top, "left": emulator_win.left,
            "width": emulator_win.width, "height": emulator_win.height
        }
        OVERLAY = TrajectoryOverlay(initial_monitor_info)
        return emulator_win
    else:
        print("Fenêtre non trouvée, impossible de lancer l'overlay.")
        exit()

    
if __name__ == "__main__":
    CURRENT_WIN = __init__()
    mouse_listener = Listener(on_click=on_mouse_click)
    mouse_listener.start()

    try:
        while True:

            frame, monitor_info = capture_window(CURRENT_WIN)
            
            if frame is not None:
                debug_frame = frame.copy()
                
                puck = find_puck(frame, LAST_CLICK_POS)
                if puck:
                    x, y, r = puck
                    cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
                    trajectory_start, trajectory_end = find_arrow_direction(debug_frame, puck[:2])
                else:
                    trajectory_start, trajectory_end = None, None
                OVERLAY.update_trajectory(trajectory_start, trajectory_end)
                        
                cv2.imshow("Debug view", debug_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('o'):
                    toggle_overlay()

        
            else:
                print("Erreur de capture, pause.")
                time.sleep(0.1)
                continue
        
    except KeyboardInterrupt:
        print("Arrêt demandé.")
    
    finally:
        if mouse_listener.is_alive():
            mouse_listener.stop()
        if 'overlay' in locals() and OVERLAY:
            OVERLAY.close()
        cv2.destroyAllWindows()
