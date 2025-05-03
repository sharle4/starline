import mss
import numpy as np
import cv2
import time
import pygetwindow as gw
from collections import deque
from screeninfo import get_monitors
from pynput.mouse import Listener, Button
from window_finder import find_emulator_window
from overlay_class import TrajectoryOverlay
from arrow_direction import find_arrow_direction, distance_points


TEMPLATE_GRAY = None
W, H = None, None
OVERLAY = None
LAST_CLICK_POS = None
LAST_RCLICK_POS = None
CURRENT_WIN = None
BASE_PUCK_RADIUS = 25


def get_screen_dimensions():
    """Retourne les dimensions de l'écran principal."""
    for m in get_monitors():
        if m.is_primary:
            return m.width, m.height

def position_all_windows():
    """Position des fenêtres au 4 coins de l'écran."""
    screen_width, screen_height = get_screen_dimensions()
    quadrant_width = screen_width // 2
    quadrant_height = screen_height // 2
    
    emulator_win = find_emulator_window()
    if emulator_win:
        try:
            win = gw.getWindowsWithTitle(emulator_win.title)[0]
            win.moveTo(0, 0)
            win.resizeTo(quadrant_width, quadrant_height)
            
            emulator_win.left = 0
            emulator_win.top = 0
            emulator_win.width = quadrant_width
            emulator_win.height = quadrant_height
            
            global OVERLAY
            if OVERLAY:
                OVERLAY.update_position({
                    "top": 0, "left": 0,
                    "width": quadrant_width, "height": quadrant_height
                })
        except (IndexError, AttributeError) as e:
            print(f"Impossible de repositionner l'émulateur: {e}")
    
    cv2.namedWindow("Debug view", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Debug view", quadrant_width, 0)
    cv2.resizeWindow("Debug view", quadrant_width, quadrant_height)
    
    cv2.namedWindow("Yellow Arrow Mask", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Yellow Arrow Mask", 0, quadrant_height)
    cv2.resizeWindow("Yellow Arrow Mask", quadrant_width, quadrant_height)
    
    cv2.namedWindow("Debug Candidates", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Debug Candidates", quadrant_width, quadrant_height)
    cv2.resizeWindow("Debug Candidates", quadrant_width, quadrant_height)


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
    global LAST_CLICK_POS, LAST_RCLICK_POS, CURRENT_WIN
    
    if pressed and button == Button.left:
        if CURRENT_WIN and CURRENT_WIN.isActive:
            relative_x = x - CURRENT_WIN.left
            relative_y = y - CURRENT_WIN.top
            
            if 0 <= relative_x < CURRENT_WIN.width and 0 <= relative_y < CURRENT_WIN.height:
                if button == Button.left:
                    LAST_CLICK_POS = (relative_x, relative_y)
                    print(f"Clic gauche enregistré à: {LAST_CLICK_POS}")
                elif button == Button.right:
                    LAST_RCLICK_POS = (relative_x, relative_y)
                    print(f"Clic droit enregistré à: {LAST_RCLICK_POS}")
    return True


class CircleTracker:
    """
    Lisse (x,y,r) sur N frames.
    S'il n'y a plus de détection quelques frames, 
    on conserve la dernière valeur lissée.
    """
    def __init__(self, history=5):
        self.hist = deque(maxlen=history)

    def smooth(self, puck_raw):
        if puck_raw is not None:
            self.hist.append(puck_raw)

        if not self.hist:
            return None

        xs, ys, rs = zip(*self.hist)
        return (int(np.mean(xs)),
                int(np.mean(ys)),
                int(np.mean(rs)))

puck_tracker = CircleTracker(history=5)
ball_tracker = CircleTracker(history=5)


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
        #print(f"Palet trouvé : ({x}, {y}), rayon : {r}")
        return puck_tracker.smooth((x, y, r))

    print("Aucun palet trouvé.")
    return None


def find_ball(frame, puck_radius, click_pos=None):
    """Retourne (x,y,r) du ballon ou None."""
    if frame is None:
        return None
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 50, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    mask = cv2.medianBlur(mask, 5)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    min_r = 8
    max_r = int(puck_radius/2)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30:
            continue
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if radius < min_r or radius > max_r:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.7:
            continue

        mask_ball = np.zeros(mask.shape, np.uint8)
        cv2.circle(mask_ball, (int(x), int(y)), int(radius*0.8), 255, -1)
        mean_val = cv2.mean(frame, mask=mask_ball)
        hsv_ball = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v = cv2.mean(hsv_ball[:,:,2], mask=mask_ball)[0]
        if v < 220:
            candidates.append((int(x), int(y), int(radius), distance_points((x, y), click_pos)))

    if candidates:
        x, y, r, d = max(candidates, key=lambda c: c[3])
        return ball_tracker.smooth((x, y, r))

    print("Aucun ballon trouvé.")
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
        

def compute_bounce_trajectory(start, end, width, height, max_bounces=5):
    """Calcule la trajectoire avec rebonds sur les murs de la fenêtre."""
    top_wall = int(height * (217 / 720))
    bottom_wall = int(height * (653 / 720))
    left_wall = int(width * (234 / 1280))
    right_wall = int(width * (982 / 1280))
    
    points = [start]
    x, y = start
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if dx == 0 and dy == 0:
        return points
    vx = dx
    vy = dy
    
    for _ in range(max_bounces):
        t_candidates = []
        if vx > 0:
            t_right = (right_wall - x) / vx
            t_candidates.append((t_right, 'right'))
        elif vx < 0:
            t_left = (left_wall - x) / vx
            t_candidates.append((t_left, 'left'))
            
        if vy > 0:
            t_bottom = (bottom_wall - y) / vy
            t_candidates.append((t_bottom, 'bottom'))
        elif vy < 0:
            t_top = (top_wall - y) / vy
            t_candidates.append((t_top, 'top'))
            
        t_wall, wall = min((t for t in t_candidates if t[0] > 0), default=(None, None))
        if t_wall is None:
            break
        x_new = x + vx * t_wall
        y_new = y + vy * t_wall
        
        x_new = min(max(x_new, left_wall), right_wall)
        y_new = min(max(y_new, top_wall), bottom_wall)
        points.append((int(x_new), int(y_new)))
        x, y = x_new, y_new
        
        if wall == 'left' or wall == 'right':
            vx = -vx
        if wall == 'top' or wall == 'bottom':
            vy = -vy
    return points


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
        
        cv2.namedWindow("Debug view", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Yellow Arrow Mask", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Debug Candidates", cv2.WINDOW_NORMAL)
        position_all_windows()
                               
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
                if monitor_info and (monitor_info["left"] != CURRENT_WIN.left or monitor_info["top"] != CURRENT_WIN.top):
                    position_all_windows()
                
                debug_frame = frame.copy()
                
                puck = find_puck(frame, LAST_CLICK_POS)
                ball = find_ball(frame, BASE_PUCK_RADIUS/1280*monitor_info["width"], LAST_RCLICK_POS)
                if ball:cv2.circle(debug_frame, (ball[0], ball[1]), ball[2], (255, 255, 255), 4)
                
                if puck:
                    x, y, r = puck
                    cv2.circle(debug_frame, (x, y), r, (0, 255, 0), 4)
                    trajectory_start, trajectory_end = find_arrow_direction(debug_frame, puck)
                    if trajectory_start and trajectory_end:
                        width = monitor_info["width"]
                        height = monitor_info["height"]
                        points = compute_bounce_trajectory(trajectory_start, trajectory_end, width, height)
                    else:
                        points = []
                else:
                    points = []
                OVERLAY.update_trajectory(points)
                        
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
