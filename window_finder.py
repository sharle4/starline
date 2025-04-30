import pygetwindow as gw
import time

EMULATOR_WINDOW_TITLE = "MuMu Player 12"

def find_emulator_window():
    """Trouve la fenêtre de l'émulateur et retourne ses informations."""
    try:
        window = gw.getWindowsWithTitle(EMULATOR_WINDOW_TITLE)[0]
        time.sleep(0.5)
        if window:
            print(f"Fenêtre trouvée : {window.title}, Position : ({window.left}, {window.top}), Taille : ({window.width}, {window.height})")

            if window.isMinimized:
                window.restore()
                time.sleep(0.5)
            try:
                window.activate()
            except Exception as e:
                print(f"Ne peut pas activer la fenêtre: {e}") # Certaines fenêtres refusent l'activation programmatique
            
            return window
        
        else:
            print(f"Fenêtre '{EMULATOR_WINDOW_TITLE}' non trouvée.")
            return None
        
    except IndexError:
        print(f"Fenêtre '{EMULATOR_WINDOW_TITLE}' non trouvée.")
        return None
    
    except Exception as e:
        print(f"Erreur en cherchant la fenêtre : {e}")
        return None

if __name__ == "__main__":
    emulator_win = find_emulator_window()
    if emulator_win:
        print("Fenêtre prête.")
    else:
        print("Veuillez lancer l'émulateur avec le jeu.")