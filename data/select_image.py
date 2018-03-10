from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
import os


class SelectImage(object):

    def __init__(self, image_path, result_path_0, result_path_1, width=300, height=300, image_size=200):
        self.image_path = image_path
        self.result_path_0 = result_path_0
        self.result_path_1 = result_path_1
        self.image_size = image_size

        # 所有的图片信息
        self.images_info = self.get_image_info(self.image_path)

        self.gui_root = Tk()
        self._init_gui(width, height)
        self._init_image(self.images_info[0][1])
        self._init_button(self.click_func_0, self.click_func_1)

        # 当前图片计数
        self.count = 0
        pass

    def run(self):
        # 启动主循环
        self.gui_root.mainloop()
        pass

    def _init_gui(self, width, height):
        sw = self.gui_root.winfo_screenwidth()
        sh = self.gui_root.winfo_screenheight()
        self.gui_root.geometry("%dx%d+%d+%d" % (width, height, (sw - width) // 2, (sh - 2 * height) // 2))
        pass

    def _init_image(self, filename):
        im = Image.open(filename)
        im = im.resize((self.image_size, self.image_size))
        global photo
        photo = ImageTk.PhotoImage(im)
        label = Label(self.gui_root, image=photo)
        label.grid(columnspan=2, row=0, ipadx=50, ipady=20)
        pass

    def _init_button(self, click_func_0, click_func_1):
        btn = Button(self.gui_root, text="0", command=click_func_0, width=10, height=1, bg="#678", bd=0, relief="groove")
        btn.grid(column=0, row=1, padx=10, pady=10, sticky="e")
        btn2 = Button(self.gui_root, text="1", command=click_func_1, width=10, height=1, bg="#876", bd=0, relief="groove")
        btn2.grid(column=1, row=1, padx=10, pady=10, sticky="w")
        pass

    # 得到所有的图片
    def get_image_info(self, now_path):
        image_info = []
        image_path_files = os.listdir(now_path)
        for path_file in image_path_files:
            now_path_file = os.path.join(now_path, path_file)
            if os.path.isdir(now_path_file):  # 是目录
                image_info.extend(self.get_image_info(now_path_file))  # 递归
            elif ".jpg" in now_path_file or ".png" in now_path_file:  # 是图片
                image_info.append((os.path.basename(now_path), now_path_file))
            pass
        return image_info

    # 第一类
    def click_func_0(self):
        self.save_image(self.result_path_0)
        pass

    # 第二类
    def click_func_1(self):
        self.save_image(self.result_path_1)
        pass

    # 保存图片并显示下一张
    def save_image(self, result_path, is_save_image=True):
        # OVER
        if self.count >= len(self.images_info):
            messagebox.showinfo("info", "OVER")
            pass

        # 保存
        if is_save_image:
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            src_image_name = self.images_info[self.count][1]
            des_image_file_path = os.path.join(result_path, self.images_info[self.count][0]).replace(" ", "_")
            des_image_path_and_name = "{}_{}{}".format(des_image_file_path, self.count, os.path.splitext(src_image_name)[1])
            Image.open(src_image_name).save(des_image_path_and_name)
            print("save {} ok in {}".format(src_image_name, des_image_path_and_name))
            pass

        try:
            # 下一张
            self.count += 1
            self._init_image(self.images_info[self.count][1])
        except OSError as e:  # 下一张出错
            print(".......... error {}".format(self.images_info[self.count][1]))
            self.save_image(result_path, is_save_image=False)  # 不保存报错的图片，直接显示下一张
            pass
        pass
    pass


if __name__ == '__main__':
    select_image = SelectImage(image_path="../images", result_path_0="../data/data/0",
                               result_path_1="../data/data/1")
    select_image.run()
