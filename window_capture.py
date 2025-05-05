import mss
import numpy as np
import cv2
import time
import math
import pygetwindow as gw
from collections import deque
from screeninfo import get_monitors
from pynput.mouse import Listener, Button
from pynput.keyboard import Listener as KeyboardListener
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
BALL_TRAJECTORY = True
PUCK_TRAJECTORY = True
FREEZE_TRAJECTORY = False


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
    cv2.moveWindow("Debug view", quadrant_width, quadrant_height)
    cv2.resizeWindow("Debug view", quadrant_width, quadrant_height)
    
    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Mask", 0, quadrant_height)
    cv2.resizeWindow("Mask", quadrant_width, quadrant_height)


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


def on_key_press(key):
    """Gère les pressions de touches."""
    global BALL_TRAJECTORY, PUCK_TRAJECTORY, FREEZE_TRAJECTORY, running
    try:
        if key.char == 'q':
            print("Arrêt demandé.")
            running = False
            
        elif key.char == 'o' or key.char == 't' or key.char == 'O' or key.char == 'T':
            print("Basculer l'affichage des trajectoires.")
            if BALL_TRAJECTORY or PUCK_TRAJECTORY:
                BALL_TRAJECTORY, PUCK_TRAJECTORY = False, False
            else:
                BALL_TRAJECTORY, PUCK_TRAJECTORY = True, True
                
        elif key.char == 'b' or key.char == 'B':
            print("Basculer l'affiche de la trajectoire de la balle.")
            BALL_TRAJECTORY = not BALL_TRAJECTORY
            
        elif key.char == 'p' or key.char == 'P':
            print("Basculer l'affiche de la trajectoire du palet.")
            PUCK_TRAJECTORY = not PUCK_TRAJECTORY
        
        elif key.char == 'f' or key.char == 'F':
            print("Basculer le mode de gel de trajectoires.")
            FREEZE_TRAJECTORY = not FREEZE_TRAJECTORY
            
            
            
    except AttributeError:
        pass
    except Exception as e:
        print(f"Erreur lors de la pression de touche: {e}")


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
aimcircle_tracker = CircleTracker(history=5)

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
                             maxRadius=22)

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
    """Retourne (x,y,r) de la balle ou None."""
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
    
    #cv2.imshow("Mask", mask_white)
    
    circles = cv2.HoughCircles(
        mask_white,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=150,
        param2=5,
        minRadius=9,
        maxRadius=11
    )
    
    best_ball = None
    if circles is not None:
        circles = np.uint16(np.around(circles[0, :]))
        circles = sorted(circles, key=lambda c: distance_points((c[0]+x_min, c[1]+y_min), (x0, y0)))
        best_ball = circles[0]
        bx, by, br = best_ball[0] + x_min, best_ball[1] + y_min, best_ball[2]
        print(f"Ballon trouvé : ({bx}, {by}), R={br}, distance={distance_points((bx, by), (x0, y0))}")
        return ball_tracker.smooth((bx, by, br))
         
    return ball_tracker.smooth(None)


def find_aimcircle(frame, puck, arrow_start, arrow_end):
    """Retourne (x,y,r) du cercle de visée le plus loin du palet ou None."""
    if frame is None or puck is None or arrow_start is None or arrow_end is None:
        return aimcircle_tracker.smooth(None)
    
    x, y, r = puck
    
    dx = arrow_end[0] - arrow_start[0]
    dy = arrow_end[1] - arrow_start[1]
    norm = np.hypot(dx, dy)
    if norm == 0:
        return aimcircle_tracker.smooth(None)
    
    dir_x = -dx / norm
    dir_y = -dy / norm
    
    offset = int(r * 3)
    roi_center = (int(x + dir_x * offset), int(y + dir_y * offset))
    roi_size = int(r + norm/2)
    x_min, x_max = max(0, roi_center[0] - roi_size), min(frame.shape[1], roi_center[0] + roi_size)
    y_min, y_max = max(0, roi_center[1] - roi_size), min(frame.shape[0], roi_center[1] + roi_size)
    roi = frame[y_min:y_max, x_min:x_max]
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.medianBlur(mask, 5)
    
    cv2.imshow("Mask", mask)
    
    circles = cv2.HoughCircles(
        mask,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=6,
        param1=50,
        param2=10,
        minRadius=2,
        maxRadius=4
    )
    
    results = []
    best_circle = None
    
    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        for cx, cy, cr in circles:
            cx, cy, cr = int(cx), int(cy), int(cr)
            x0, y0 = map(int, arrow_start)
            x1, y1 = map(int, arrow_end)
            px, py = cx + x_min, cy + y_min
            num = abs((y1 - y0)*px - (x1 - x0)*py + x1*y0 - y1*x0)
            den = np.hypot(y1 - y0, x1 - x0)
            distance = num / den if den != 0 else float('inf')
            if distance < 4:
                results.append((cx + x_min, cy + y_min, cr))
    
        #best_circle = sorted(results, key=lambda c: distance_points((c[0], c[1]), (x, y)))[-1]
        #print(f"distance au palet: {distance_points((best_circle[0], best_circle[1]), (x, y))}")
        
    if results:
        return results
        #print(f"Cercle de visée trouvé : ({best_circle[0]}, {best_circle[1]}), R={best_circle[2]}")
        return aimcircle_tracker.smooth(best_circle)
    print("Aucun cercle de visée trouvé.")
    return aimcircle_tracker.smooth(None)


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
        cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
        position_all_windows()
                               
        return emulator_win
    else:
        print("Fenêtre non trouvée, impossible de lancer l'overlay.")
        exit()

    
if __name__ == "__main__":
    CURRENT_WIN = __init__()
    mouse_listener = Listener(on_click=on_mouse_click)
    mouse_listener.start()
    keyboard_listener = KeyboardListener(on_press=on_key_press)
    keyboard_listener.start()
    running = True
    print("Lancement de l'application...")
    try:
        while running:

            frame, monitor_info = capture_window(CURRENT_WIN)
            
            if frame is not None:
                if monitor_info and (monitor_info["left"] != CURRENT_WIN.left or monitor_info["top"] != CURRENT_WIN.top):
                    position_all_windows()
                
                debug_frame = frame.copy()
                
                puck = find_puck(frame, LAST_CLICK_POS)
                ball = find_ball(frame, BASE_PUCK_RADIUS/1280*monitor_info["width"], LAST_RCLICK_POS)
                if ball:cv2.circle(debug_frame, (ball[0], ball[1]), ball[2], (255, 255, 255), 4)
                
                trajectories = []
                
                if puck:
                    x, y, r = puck
                    cv2.circle(debug_frame, (x, y), r, (0, 255, 0), 4)
                    trajectory_start, trajectory_end = find_arrow_direction(frame, puck, debug_frame)
                    trajectory_start_points = [trajectory_start]
                    aim_circles = find_aimcircle(frame, puck, trajectory_start, trajectory_end)
                    if aim_circles:
                        for aim_circle in aim_circles:
                            cv2.circle(debug_frame, (aim_circle[0], aim_circle[1]), aim_circle[2], (255, 0, 255), 2)
                            trajectory_start_points.append((aim_circle[0], aim_circle[1]))
                    
                    mean_x = int(np.mean([p[0] for p in trajectory_start_points]))
                    mean_y = int(np.mean([p[1] for p in trajectory_start_points]))
                    refined_trajectory_start = (mean_x, mean_y)
                        
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
                                end_x = int(bx + impact_dir[0] * 500)
                                end_y = int(by + impact_dir[1] * 500)
                                trajectory_start_ball = (bx, by)
                                trajectory_end_ball = (end_x, end_y)
                                ball_points = compute_bounce_trajectory(trajectory_start_ball, trajectory_end_ball, width, height)
                                if BALL_TRAJECTORY:
                                    trajectories.append((ball_points, 'white'))
                            else:
                                ball_points = []
                        else:
                            ball_points = []
                        puck_points = compute_bounce_trajectory(refined_trajectory_start, trajectory_end, width, height)
                        if PUCK_TRAJECTORY:
                            trajectories.append((puck_points, 'red'))
                            
                    else:
                        if trajectory_start and trajectory_end:
                            puck_points = compute_bounce_trajectory(refined_trajectory_start, trajectory_end, width, height)
                            if PUCK_TRAJECTORY:
                                trajectories.append((puck_points, 'red'))
                
                if not FREEZE_TRAJECTORY:
                    OVERLAY.update_trajectory(trajectories)
                        
                cv2.imshow("Debug view", debug_frame)
                key = cv2.waitKey(1) & 0xFF
        
            else:
                print("Erreur de capture, pause.")
                time.sleep(0.1)
                continue
        
    except KeyboardInterrupt:
        print("Arrêt demandé.")
    
    finally:
        if mouse_listener.is_alive():
            mouse_listener.stop()
        if keyboard_listener.is_alive():
            keyboard_listener.stop()
        if 'overlay' in locals() and OVERLAY:
            OVERLAY.close()
        cv2.destroyAllWindows()
