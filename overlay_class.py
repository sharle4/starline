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

        self.line_ids = []
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


    def update_trajectory(self, trajectories):
        """Met à jour la ligne de trajectoire avec rebonds sur le canvas."""
        for line_id in self.line_ids:
            self.canvas.delete(line_id)

        self.line_ids = []

        for points, color in trajectories:
            if points and len(points) > 1:
                line_id = self.canvas.create_line(
                    *sum(points, ()), fill=color, width=2)
                self.line_ids.append(line_id)

        self.root.update()

    def close(self):
        self.root.destroy()