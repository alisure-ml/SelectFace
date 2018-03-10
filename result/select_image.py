import os
import shutil


def select_image(result_txt="result.txt", result_ok_path="images_image", result_error_path="images_image_error"):
    result_ok_path_all = result_ok_path + "_all"
    result_error_path_all = result_error_path + "_all"

    with open(result_txt, "r") as f:
        count_ok = 0
        count_error = 0

        all_line = f.readlines()
        for line in all_line:
            image_path, image_file_name, image_type = line.split(",")


            if int(image_type) == 0:
                result_image_path = os.path.join(result_ok_path, image_path)
                result_image_path_all = result_ok_path_all
                count_ok += 1
            else:
                result_image_path = os.path.join(result_error_path, image_path)
                result_image_path_all = result_error_path_all
                count_error += 1
                pass

            if not os.path.exists(result_image_path):
                os.makedirs(result_image_path)

            if not os.path.exists(result_ok_path_all):
                os.makedirs(result_ok_path_all)

            if not os.path.exists(result_error_path_all):
                os.makedirs(result_error_path_all)

            shutil.copy(os.path.join("..",image_file_name), dst=os.path.join(result_image_path, os.path.basename(image_file_name)))

            shutil.copy(os.path.join("..",image_file_name), dst=os.path.join(result_image_path_all, image_path + "_" + os.path.basename(image_file_name)))
            pass
        print("ok is {}, error is {}".format(count_ok, count_error))
    pass

if __name__ == '__main__':
    select_image()
