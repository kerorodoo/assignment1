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


class Demo_shapes():
    #  draw shape and image on window
    def __init__(self, master, title, shapes, colors):

        window = master
        window.title(title)
        canvas = Tkinter.Canvas(window, width=int(WIDTH), height=int(HEIGHT), bg="#000000")

        if (shapes[0]._image_path != ''):

            #  print the which image will opening
            sys.stdout.write(
                    "\nopening the image with path: {}\n".format(
                        shapes[0].get_image_path()))

            image = Image.open(shapes[0].get_image_path())

            #  crop the image by box
            image = image.crop(box=(shapes[0].get_left(), shapes[0].get_upper(), shapes[0].get_right(), shapes[0].get_lower()))
            #  scale the image to window-size
            image = image.resize(size=(int(WIDTH), int(HEIGHT)))

            photo = ImageTk.PhotoImage(image)
            canvas.create_image((int(WIDTH)/2, int(HEIGHT)/2), image=photo, state="normal")

            # keep a reference
            canvas.image = photo

        canvas.pack()

        for i in range(len(shapes)):
            for part in shapes[i].get_normalized_landmarks(int(WIDTH), int(HEIGHT)):
                circle_left = int(part[0]) - CIRCLE_R/2
                circle_top = int(part[1]) - CIRCLE_R/2
                circle_right = int(part[0]) + CIRCLE_R/2
                circle_buttom = int(part[1]) + CIRCLE_R/2
                canvas.create_oval(circle_left, circle_top, circle_right, circle_buttom, outline=colors[i], fill=colors[i], width=2*(i+1))

class List_view_all_shapes():
    def __init__(self, master, model):
        self.shape_ptr = 0
        self.shapes = model.get_shapes()

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
            assert (tree_view.item(selected_item).get('values') != str(''), "\nWarring: invalid selected ! ! !\n")

            selected_item_index = tree_view.item(selected_item).get('values')[0]
            selected_item_image_path = tree_view.item(selected_item).get('values')[1]
            selected_shape_ptr = int(selected_item_index)


            #  store the selected index
            self.shape_ptr = selected_shape_ptr

            self.newWindow = Tkinter.Toplevel(list_view)
            self.newWindowTitle = "selected {}-th shape".format(self.shape_ptr)
            self.newWindowShapes = []
            self.newWindowShapes.append(self.shapes[self.shape_ptr])
            self.newWindowShapes.append(model.calcuate_shape_from_mean(self.shapes[self.shape_ptr]))
            self.newWindowColors = ['red', 'green']
            self.app = Demo_shapes(self.newWindow, self.newWindowTitle, self.newWindowShapes, self.newWindowColors)

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

    root = Tkinter.Tk()
    app = List_view_all_shapes(root, model)

    mean_shape = model.get_mean_shape()
    model.calculate_mean_shape_image()
    childWindowShape = [mean_shape]
    childWindow = Tkinter.Toplevel()
    childWindowTitle = "mean shape"
    childWindowColor = ['red']
    appslave = Demo_shapes(childWindow, childWindowTitle, childWindowShape, childWindowColor)


    root.mainloop()

if __name__ == '__main__':
    main()
