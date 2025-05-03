import cv2
import numpy as np

def distance_points(p1, p2):
    """ Calcule la distance euclidienne entre deux points (tuples). """
    if p1 is None or p2 is None:
        return float('inf')
    try:
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    except (TypeError, IndexError):
        print(f"Erreur de calcul de distance avec p1={p1}, p2={p2}")
        return float('inf')



def find_arrow_direction(frame, puck_pos):
    """
    Détecte la flèche de visée près du palet et retourne son point de départ (puck_pos)
    et un point final virtuel définissant la direction.
    Retourne (None, None) si non trouvée.
    """
    if frame is None or puck_pos is None:
        print("Erreur: Frame ou puck_pos manquant.")
        return None, None
    
    debug_frame = frame.copy()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) 
    
    canny_thresh1 = 30
    canny_thresh2 = 100
    edges = cv2.Canny(blurred, canny_thresh1, canny_thresh2)
    cv2.namedWindow("Grayscale Edges", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Debug Candidates", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Grayscale Edges", 320, 240)
    cv2.resizeWindow("Debug Candidates", 320, 240)
    cv2.imshow("Grayscale Edges", edges)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    potential_arrows = []
    min_arrow_area = 30       # Augmenter si nécessaire (dépend de la taille min de la flèche)
    max_arrow_area = 10000    # Limite supérieure pour éviter les très grands objets
    max_distance_from_puck = 150 # Distance max entre centre du contour et palet (ajuster)
    min_aspect_ratio_elongation = 1.2 

    if not contours:
        print("Aucun contour trouvé par Canny.")
        pass

    for contour in contours:
        area = cv2.contourArea(contour)
        
        if not (min_arrow_area < area < max_arrow_area):
            continue
        
        M = cv2.moments(contour)
        if M["m00"] == 0: continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centroid = (cx, cy)
        dist_to_puck = distance_points(centroid, puck_pos)

        if dist_to_puck > max_distance_from_puck:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        inv_aspect_ratio = float(h) / w if w > 0 else 0
        
        if max(aspect_ratio, inv_aspect_ratio) < min_aspect_ratio_elongation:
            continue
        
        potential_arrows.append({'contour': contour, 'centroid': centroid, 'area': area, 'dist': dist_to_puck})
        cv2.drawContours(debug_frame, [contour], -1, (255, 0, 255), 1)
    
    cv2.imshow("Debug Candidates", debug_frame)
    
    if not potential_arrows:
        print("Aucune flèche potentielle après filtrage.")
        return None, None

    best_arrow = max(potential_arrows, key=lambda a: a['area'])
    arrow_contour = best_arrow['contour']

    cv2.drawContours(debug_frame, [arrow_contour], -1, (0, 255, 255), 2)

    farthest_point = None
    max_dist_from_puck_sq = -1
    
    if arrow_contour is None or len(arrow_contour) == 0:
         print("Erreur: arrow_contour est vide.")
         return None, None

    try:
        for point_wrapper in arrow_contour:
             point = tuple(point_wrapper[0])
             dist_sq = (point[0] - puck_pos[0])**2 + (point[1] - puck_pos[1])**2
             if dist_sq > max_dist_from_puck_sq:
                 max_dist_from_puck_sq = dist_sq
                 farthest_point = point
    except Exception as e:
         print(f"Erreur lors de l'itération sur arrow_contour: {e}")
         print(f"arrow_contour: {arrow_contour}") 
         return None, None


    if farthest_point:
        dist_fp = np.sqrt(max_dist_from_puck_sq)
        min_arrow_length = 5
        if dist_fp < min_arrow_length:
            print(f"Point le plus éloigné trop proche: {dist_fp:.1f} pixels")
            return None, None

        dx = farthest_point[0] - puck_pos[0]
        dy = farthest_point[1] - puck_pos[1]

        ux = dx / dist_fp
        uy = dy / dist_fp

        trajectory_length = 2000
        end_x = int(puck_pos[0] + ux * trajectory_length)
        end_y = int(puck_pos[1] + uy * trajectory_length)

        cv2.circle(debug_frame, farthest_point, 5, (0, 255, 0), -1)
        cv2.line(debug_frame, puck_pos, farthest_point, (0, 0, 255), 1)

        cv2.imshow("Debug Candidates", debug_frame)

        return puck_pos, (end_x, end_y)
    else:
        print("Impossible de déterminer le point le plus éloigné.")
        return None, None
