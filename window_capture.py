import mss
import numpy as np
import cv2
import time
import math
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
    
    if pressed:
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
    if frame is None:
        return ball_tracker.smooth(None)
    
    if click_pos is None:
        click_pos = (frame.shape[1] // 2, frame.shape[0] // 2)
    
    x0, y0 = click_pos
    search_radius = int(puck_radius)
    x_min, x_max = max(0, x0 - search_radius), min(frame.shape[1], x0 + search_radius)
    y_min, y_max = max(0, y0 - search_radius), min(frame.shape[0], y0 + search_radius)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.medianBlur(hsv, 5)
    roi = hsv[y_min:y_max, x_min:x_max]
    
    lower_white = np.array([0, 0, 190])
    upper_white = np.array([179, 70, 255])
    mask_white = cv2.inRange(roi, lower_white, upper_white)
    
    
    mask_white = cv2.medianBlur(mask_white, 3)
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel11, iterations=3)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel5, iterations=3)
    
    cv2.imshow("Yellow Arrow Mask", mask_white)
    
    circles = cv2.HoughCircles(
        mask_white,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=150,
        param2=5,
        minRadius=5,
        maxRadius=20
    )
    
    best_ball = None
    if circles is not None:
        circles = np.uint16(np.around(circles[0, :]))
        circles = sorted(circles, key=lambda c: distance_points((c[0], c[1]), (x0, y0)))
        best_ball = circles[0]
        bx, by, br = best_ball[0] + x_min, best_ball[1] + y_min, best_ball[2]
        print(f"Ballon trouvé : ({bx}, {by}), R={br}, distance={distance_points((bx, by), (x0, y0))}")
        return ball_tracker.smooth((bx, by, br))
         
    return ball_tracker.smooth(None)


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


def line_circle_intersection(p1, p2, center, radius):
    """Retourne le point d'intersection entre une ligne et un cercle."""
    if p1 == p2:
        return None
    
    (x1, y1), (x2, y2) = p1, p2
    (cx, cy) = center

    dx = x2 - x1
    dy = y2 - y1

    fx = x1 - cx
    fy = y1 - cy

    a = dx*dx + dy*dy
    b = 2 * (fx*dx + fy*dy)
    c = (fx*fx + fy*fy) - radius*radius

    delta = b*b - 4*a*c
    if delta < 0:
        return None

    sqrt_delta = math.sqrt(delta)
    t1 = (-b - sqrt_delta) / (2*a)
    t2 = (-b + sqrt_delta) / (2*a)

    points = []
    for t in [t1, t2]:
        if 0 <= t <= 1:
            ix = x1 + t*dx
            iy = y1 + t*dy
            points.append((ix, iy))
    if points:
        points.sort(key=lambda pt: (pt[0]-x1)**2 + (pt[1]-y1)**2)
        return points[0]
    return None


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
                    width = monitor_info["width"]
                    height = monitor_info["height"]
                    
                    if ball is not None and trajectory_start and trajectory_end:
                        bx, by, br = ball
                        impact_point = line_circle_intersection(trajectory_start, trajectory_end, (bx, by), br+r)
                        if impact_point:
                            impact_dir = (bx - impact_point[0], by - impact_point[1])
                            norm = np.linalg.norm(impact_dir)
                            if norm > 0:
                                impact_dir = (impact_dir[0] / norm, impact_dir[1] / norm)
                                end_x = int(bx + impact_dir[0] * 2000)
                                end_y = int(by + impact_dir[1] * 2000)
                                trajectory_start_ball = (bx, by)
                                trajectory_end_ball = (end_x, end_y)
                                points = compute_bounce_trajectory(trajectory_start_ball, trajectory_end_ball, width, height)
                            else:
                                points = []
                        else:
                            points = compute_bounce_trajectory(trajectory_start, trajectory_end, width, height)
                            
                    else:
                        if trajectory_start and trajectory_end:
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
