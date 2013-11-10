import Tkinter as tk

class StructureCanvas(tk.Canvas):
    def __init__(self, *args, **kwargs):
        canvas = tk.Canvas.__init__(self, *args, **kwargs)
        canvas = tk.Canvas(self, width=800, height=500)

        canvas_id = canvas.create_text(10, 10, anchor="nw")

        canvas.itemconfig(canvas_id, text="this is the text")
        canvas.insert(canvas_id, 12, "new ")

if __name__ == "__main__":
    root = tk.Tk()
    StructureCanvas(root).pack(side="top", fill="both", expand=True)
    root.mainloop()
