import cv2
import numpy as np

def distance_to_puck(contour, puck_position):
    """ Calcule la distance entre le centre de gravité du contour et la position du palet."""
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return np.sqrt((cx - puck_position[0])**2 + (cy - puck_position[1])**2)
    return float('inf')


def find_arrow_direction(frame, puck_pos):
    """
    Détecte la flèche de visée près du palet et retourne son point de départ (puck_pos)
    et un point final virtuel définissant la direction.
    Retourne (None, None) si non trouvée.
    """
    if frame is None or puck_pos is None:
        return None, None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([0, 100, 100])
    upper_yellow = np.array([10, 255, 255])
    lower_red = np.array([20, 100, 100])
    upper_red = np.array([50, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    mask = cv2.bitwise_or(yellow_mask, red_mask)
    edges = cv2.Canny(mask, 50, 150)

    # kernel = np.ones((3,3), np.uint8)
    # edges = cv2.dilate(edges, kernel, iterations=1)

    cv2.imshow("Edges", edges)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    potential_arrows = []
    min_arrow_area = 50 
    max_distance_from_puck = 200

    for contour in contours:
        if cv2.contourArea(contour) < min_arrow_area:
            continue
        
        if distance_to_puck(contour, puck_pos) > max_distance_from_puck:
            continue
        
        potential_arrows.append(contour)       
        cv2.drawContours(frame, [contour], -1, (255, 0, 255), 2) # Rose

    if not potential_arrows:
        print("Aucun contour d'intérêt près du palet.")
        return None, None

    arrow_contour = min(potential_arrows, key=lambda c: distance_to_puck(c, puck_pos))
    cv2.drawContours(frame, [arrow_contour], -1, (0, 255, 255), 2) # Cyan

    farthest_point = None
    max_dist_sq = -1

    for point in arrow_contour.reshape(-1, 2):
        dist_sq = (point[0] - puck_pos[0])**2 + (point[1] - puck_pos[1])**2
        if dist_sq > max_dist_sq:
            max_dist_sq = dist_sq
            farthest_point = tuple(point)

    if farthest_point:
        dx = farthest_point[0] - puck_pos[0]
        dy = farthest_point[1] - puck_pos[1]
        dist = np.sqrt(dx**2 + dy**2)

        if dist < 1:
            return None, None

        ux = dx / dist
        uy = dy / dist

        trajectory_length = 2000
        end_x = int(puck_pos[0] + ux * trajectory_length)
        end_y = int(puck_pos[1] + uy * trajectory_length)

        cv2.circle(frame, farthest_point, 5, (0, 255, 0), -1) # Point vert à la pointe

        return puck_pos, (end_x, end_y)

    return None, None