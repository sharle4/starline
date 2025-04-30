import tkinter as tk

class TrajectoryOverlay:
    def __init__(self, window_info):
        self.root = tk.Tk()
        self.window_info = window_info

        self.root.geometry(f"{window_info['width']}x{window_info['height']}+{window_info['left']}+{window_info['top']}")
        self.root.overrideredirect(True)
        self.root.wm_attributes("-topmost", True)


        try:
            self.root.wm_attributes("-transparentcolor", "black") 
        except tk.TclError:
            print("La transparence via -transparentcolor n'est peut-être pas supportée.")
            self.root.attributes('-alpha', 0.8)


        self.canvas = tk.Canvas(self.root, bg='black', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.line_id = None
        self._make_click_through()

    def _make_click_through(self):
        try:
            import win32gui
            import win32con
            import win32api

            hwnd = self.root.winfo_id()
            styles = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
            styles |= win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT
            win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, styles)
            win32gui.SetLayeredWindowAttributes(hwnd, win32api.RGB(0, 0, 0), 0, win32con.LWA_COLORKEY)
        except ImportError:
            print("pywin32 non trouvé. La fenêtre overlay ne sera pas 'click-through'.")
        except Exception as e:
            print(f"Erreur lors de la configuration click-through: {e}")


    def update_trajectory(self, start_point, end_point):
        """Met à jour la ligne de trajectoire sur le canvas."""
        if self.line_id:
            self.canvas.delete(self.line_id)

        if start_point and end_point:
            self.line_id = self.canvas.create_line(
                start_point[0], start_point[1],
                end_point[0], end_point[1],
                fill="white", width=2, tags="trajectory"
            )
        else:
             self.line_id = None 

        self.root.update()

    def close(self):
        self.root.destroy()