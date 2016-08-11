import py_landmark_editor_model as m

import sys
#  using argv

import Tkinter
import ttk
#  for listview

from PIL import Image, ImageTk
#  for jpg support
#  required install python-pil.imagetk

WIDTH = 400.0
HEIGHT = 400.0
CIRCLE_R = 2

class Demo_mean_shape():
    def __init__(self, master, mean_landmarks):
        #  display the mean shape in gui window
        window_width = int(WIDTH)
        window_height = int(HEIGHT)
        window = master
        window.title("mean shape")
        canvas = Tkinter.Canvas(window, width=window_width, height=window_height, bg="#000000")
        canvas.pack()
        img = Tkinter.PhotoImage(width=window_width, height=window_height)
        canvas.create_image((window_width/2, window_height/2), image=img, state="normal")
        for part in mean_landmarks:
            circle_left = int(part[0]) - CIRCLE_R/2
            circle_top = int(part[1]) - CIRCLE_R/2
            circle_right = int(part[0]) + CIRCLE_R/2
            circle_buttom = int(part[1]) + CIRCLE_R/2
            canvas.create_oval(circle_left, circle_top, circle_right, circle_buttom, outline="red", fill="green", width=2)

class Demo_selected_shape():
    def __init__(self, master, shapes, shape_ptr):
        window_width = int(WIDTH)
        window_height = int(HEIGHT)

        window2 = master
        window2.title("selected {} shap".format(str(shape_ptr)))
        canvas2 = Tkinter.Canvas(window2, width=window_width, height=window_height, bg="#000000")

        print "\nopening the image: {}\n".format(shapes[shape_ptr]._image_path)
        image = Image.open(shapes[shape_ptr]._image_path)

        #  crop the image by box
        image = image.crop(box=(shapes[shape_ptr].get_left(), shapes[shape_ptr].get_upper(), shapes[shape_ptr].get_right(), shapes[shape_ptr].get_lower()))
        #  scale the image to window-size
        image = image.resize(size=(window_width, window_height))

        photo = ImageTk.PhotoImage(image)
        canvas2.create_image((window_width/2, window_height/2), image=photo, state="normal")

        # keep a reference
        canvas2.image = photo
        canvas2.pack()

        for part in shapes[shape_ptr].get_normalized_landmarks(WIDTH, HEIGHT):
            circle_left = int(part[0]) - CIRCLE_R/2
            circle_top = int(part[1]) - CIRCLE_R/2
            circle_right = int(part[0]) + CIRCLE_R/2
            circle_buttom = int(part[1]) + CIRCLE_R/2
            canvas2.create_oval(circle_left, circle_top, circle_right, circle_buttom, outline="red", fill="green", width=2)

class List_view_all_shapes():
    def __init__(self, master, shapes):
        self.shape_ptr = 0
        self.shapes = shapes

        #  create listview for select
        list_view = master
        list_view.title("list of shape")
        tree_view = ttk.Treeview(list_view, columns=('index','image_path'))
        tree_view['show'] = 'headings'
        tree_view.column('#0', width=0)
        tree_view.column('index', width=100, anchor='center')
        tree_view.column('image_path', width=300, anchor='center')
        tree_view.heading('index', text='index')
        tree_view.heading('image_path', text='image path')

        #  insert data to full tree_view
        for i in range(len(self.shapes)):
            tree_view.insert('' ,i,values=(str(i), self.shapes[i]._image_path))
        tree_view.pack()

        #  bind the event of mouse
        def onListDoubleClick(event):
            selected_item = tree_view.selection()

            #  check the item value is valid
            assert len(tree_view.item(selected_item).get('values')) == 2, "\nError: item invalid ! ! !\n"

            selected_item_index = tree_view.item(selected_item).get('values')[0]
            selected_item_image_path = tree_view.item(selected_item).get('values')[1]
            selected_shape_ptr = int(selected_item_index)

            assert selected_item_image_path == self.shapes[selected_shape_ptr]._image_path, "\nError: image_path not matched ! ! !\n"

            #  store the selected index
            self.shape_ptr = selected_shape_ptr

            self.newWindow = Tkinter.Toplevel(list_view)
            self.app = Demo_selected_shape(self.newWindow, self.shapes, self.shape_ptr)

        tree_view.bind("<Double-1>", onListDoubleClick)

        #  vertical scrollbar
        vbar = ttk.Scrollbar(list_view, orient=Tkinter.VERTICAL, command=tree_view.yview)
        tree_view.configure(yscrollcommand=vbar.set)
        tree_view.grid(row=0, column=0, sticky=Tkinter.NSEW)
        vbar.grid(row=0, column=1, sticky=Tkinter.NS)

def main():
    xml_path = "output300.xml"

    if (len(sys.argv) == 2):
        xml_path = sys.argv[1]

    model = m.Model(xml_path, WIDTH, HEIGHT)

    shapes = model.get_shapes()
    root = Tkinter.Tk()
    app = List_view_all_shapes(root, shapes)


    mean_shape = model.get_mean_shape()
    child = Tkinter.Toplevel()
    appslave = Demo_mean_shape(child, mean_shape)

    root.mainloop()

if __name__ == '__main__':
    main()
