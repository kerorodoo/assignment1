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
    def __init__(self, master, model, title, shapes):
        self.colors = ['green', 'yellow', 'red']
        self.model = model
        self.window = master
        self.window.title(title)
        self.canvas = Tkinter.Canvas(
            self.window,
            width=int(WIDTH),
            height=int(HEIGHT),
            bg="#000000" )

        if ( shapes[0].get_image_path() != '' ):

            #  print the which image will opening
            sys.stdout.write(
                "\nopening the image with path: {}\n"
                .format(shapes[0].get_image_path()) )

            image = Image.open( shapes[0].get_image_path() )

            #  crop the image by box
            image = image.crop(
                box=(shapes[0].get_left(),
                    shapes[0].get_upper(),
                    shapes[0].get_right(),
                    shapes[0].get_lower()) )
            #  scale the image to window-size
            image = image.resize( size=(int(WIDTH), int(HEIGHT)) )

            photo = ImageTk.PhotoImage(image)
            self.canvas.create_image(
                (int(WIDTH) / 2, int(HEIGHT) / 2),
                image=photo,
                state="normal" )

            # keep a reference
            self.canvas.image = photo

        self.canvas.pack()

        for shape_idx in range( len(shapes) ):
            for part in ( shapes[shape_idx].
                    get_normalized_landmarks(int(WIDTH), int(HEIGHT)) ):

                #  pick yhe color form color set
                #  accroding landmark status
                circle_color = self.colors[part[2]]

                circle_left = int( part[0] ) - CIRCLE_R / 2
                circle_top = int( part[1] ) - CIRCLE_R / 2
                circle_right = int( part[0] ) + CIRCLE_R / 2
                circle_buttom = int( part[1] ) + CIRCLE_R / 2
                self.canvas.create_oval(
                    circle_left, circle_top,
                    circle_right, circle_buttom,
                    outline=circle_color, fill=circle_color,
                    width=2 * (shape_idx + 1) )

        self.canvas.bind("<ButtonPress-1>", self.on_left_button_press)
        self.canvas.bind("<ButtonPress-3>", self.on_right_button_press)
        #  this event will store shapes into test.xml
        self.window.bind("<s>", self.on_s_press)
        #  this event will re-caculate Average.png
        self.window.bind("<r>", self.on_r_press)

    def on_left_button_press(self, event):
        self.x = int(event.x)
        self.y = int(event.y)
        self.model.add_landmark_to_mean_shape(self.x, self.y)
        self.canvas.create_oval(self.x - CIRCLE_R / 2, self.y - CIRCLE_R / 2,
            self.x + CIRCLE_R / 2, self.y + CIRCLE_R / 2, outline="blue",
            fill="blue", width=2)

    def on_right_button_press(self, event):
        self.x = int(event.x)
        self.y = int(event.y)
        self.model.remove_landmark_from_mean_shape(self.x, self.y)

    def on_s_press(self, event):
        self.model.write_xml_to_file("test.xml")

    def on_r_press(self, event):
        self.model.calculate_mean_shape_image()


class List_view_all_shapes():

    #  the class will represent the tree view,
    #  using the list view

    def __init__(self, master, model):
        self.shape_ptr = 0
        self.model = model
        self.shapes = model.get_shapes()

        #  create listview for select
        self.list_view = master

        #  layout the list view window title and grid
        self.list_view.title("list of shape")
        self.tree_view = ttk.Treeview(
            self.list_view,
            columns=('index', 'image_path'))
        self.tree_view['show'] = 'headings'
        self.tree_view.column('#0', width=0)
        self.tree_view.column('index', width=100, anchor='center')
        self.tree_view.column('image_path', width=300, anchor='center')
        self.tree_view.heading('index', text='index')
        self.tree_view.heading('image_path', text='image path')

        #  insert data to list view
        for shape_idx in range(len(self.shapes)):
            self.tree_view.insert(
                '' ,
                shape_idx,
                values=(str(shape_idx), self.shapes[shape_idx]._image_path))

        self.tree_view.pack()

        #  bind the event of mouse
        self.tree_view.bind("<Double-1>", self.onListDoubleClick)

        #  vertical scrollbar of list view
        vbar = ttk.Scrollbar(
            self.list_view,
            orient=Tkinter.VERTICAL,
            command=self.tree_view.yview)

        self.tree_view.configure(yscrollcommand=vbar.set)
        self.tree_view.grid(row=0, column=0, sticky=Tkinter.NSEW)
        vbar.grid(row=0, column=1, sticky=Tkinter.NS)

    def onListDoubleClick(self, event):
        selected_item = self.tree_view.selection()

        #  check the item value is valid
        sys.stdout.write(
            "\nselected item index: {}\n".format(
            self.tree_view.index(selected_item)) )

        selected_item_index = self.tree_view.item(
            selected_item).get('values')[0]
        selected_item_image_path = self.tree_view.item(
            selected_item).get('values')[1]
        selected_shape_ptr = int(selected_item_index)

        #  store the selected index
        self.shape_ptr = selected_shape_ptr

        self.newWindow = Tkinter.Toplevel(self.list_view)
        self.newWindowTitle = "selected {}-th shape".format(self.shape_ptr)
        self.newWindowShapes = [self.model.estimate_shape_from_mean(
            self.shapes[self.shape_ptr])]
        self.app = Demo_shapes(
            self.newWindow,
            self.model,
            self.newWindowTitle,
            self.newWindowShapes)


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
    appslave = Demo_shapes(
        childWindow,
        model,
        childWindowTitle,
        childWindowShape)

    root.mainloop()

if __name__ == '__main__':
    main()
