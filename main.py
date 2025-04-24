import os, sys
import shutil
import os
import glob
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QMessageBox, QListWidgetItem
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QDir

# 引入自动生成的 UI 类
from vehicle_RID import Ui_Form
from login import Ui_widget
from model_manage import Ui_Model_Manger
from register import Ui_Register
from guid import Ui_Guid
from video_rid import Ui_Video
# 导入重识别和检测模块
import test1
from vdete import run_vehicle_detection

class LoginWindow(QWidget, Ui_widget):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # 连接信号
        self.pushButton.clicked.connect(self.handle_user_login)  # 用户登录
        self.pushButton_2.clicked.connect(self.open_register)  # 用户注册
        self.pushButton_3.clicked.connect(self.handle_admin_login)  # 管理员登录

    # -------------------------- 用户功能 --------------------------
    def open_register(self):
        self.hide()
        self.main_window = Register_window()
        self.main_window.show()

    def handle_user_login(self):
        """用户登录验证"""
        username = self.lineEdit.text().strip()
        password = self.lineEdit_2.text().strip()

        if not username or not password:
            QMessageBox.warning(self, "错误", "请输入账号和密码！")
            return

        try:
            with open("user.txt", "r") as f:
                users = [line.strip().split(",", 1) for line in f if line.strip()]
        except FileNotFoundError:
            QMessageBox.warning(self, "错误", "用户未注册！")
            return

        for stored_user, stored_pass in users:
            if stored_user == username and stored_pass == password:
                self.open_main_window()
                return

        QMessageBox.warning(self, "错误", "账号或密码错误！")

    # -------------------------- 管理员功能 --------------------------
    def handle_admin_login(self):
        """管理员登录验证"""
        admin_id = self.lineEdit_3.text().strip()
        password = self.lineEdit_4.text().strip()

        if not admin_id or not password:
            QMessageBox.warning(self, "错误", "请输入管理员账号和密码！")
            return

        try:
            with open("admin.txt", "r") as f:
                admins = [line.strip().split(",", 1) for line in f if line.strip()]
        except FileNotFoundError:
            QMessageBox.critical(self, "错误", "管理员配置文件不存在！")
            return

        for stored_admin, stored_pass in admins:
            if stored_admin == admin_id and stored_pass == password:
                QMessageBox.information(self, "成功", "管理员登录成功！")
                self.open_admin_window()
                return

        QMessageBox.warning(self, "错误", "管理员凭证无效！")

    # -------------------------- 通用功能 --------------------------
    def open_main_window(self):
        """打开导航界面"""
        self.hide()
        self.main_window = GuidWindow()
        self.main_window.show()

    def open_admin_window(self):
        """打开管理员界面"""
        self.hide()
        self.main_window = AdminWindow()
        self.main_window.show()


class GuidWindow(QWidget, Ui_Guid):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.pushButton.clicked.connect(self.open_main_window)  # 图片识别
        self.pushButton_2.clicked.connect(self.open_video_window)  # 视频识别

    def open_main_window(self):
        self.hide()
        self.main_window = MainWindow()
        self.main_window.show()

    def open_video_window(self):
        self.hide()
        self.main_window = VideoWindow()
        self.main_window.show()



class Register_window(QWidget, Ui_Register):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.parent_window = parent  # 保存父窗口引用

        # 连接按钮信号
        self.pushButton.clicked.connect(self.handle_register)
        self.pushButton_2.clicked.connect(self.close_and_return)

    def handle_register(self):
        """处理注册逻辑"""
        username = self.lineEdit.text().strip()
        password = self.lineEdit_2.text().strip()
        confirm_pwd = self.lineEdit_3.text().strip()

        # 输入验证
        if not username or not password or not confirm_pwd:
            QtWidgets.QMessageBox.warning(self, "错误", "所有字段必须填写！")
            return

        if password != confirm_pwd:
            QtWidgets.QMessageBox.warning(self, "错误", "两次输入的密码不一致！")
            return

        # 检查用户是否存在
        try:
            existing_users = []
            try:
                with open("user.txt", "r") as f:
                    existing_users = [line.split(",")[0].strip() for line in f if line.strip()]
            except FileNotFoundError:
                pass

            if username in existing_users:
                QtWidgets.QMessageBox.warning(self, "错误", "用户名已存在！")
                return

            # 写入新用户
            with open("user.txt", "a") as f:
                f.write(f"{username},{password}\n")

            QtWidgets.QMessageBox.information(self, "成功", "注册成功！")
            self.lineEdit.clear()
            self.lineEdit_2.clear()
            self.lineEdit_3.clear()
            self.close_and_return()

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"操作失败: {str(e)}")

    def close_and_return(self):
        """关闭窗口并显示登录界面"""
        self.close()
        self.main_window = LoginWindow()
        self.main_window.show()

class AdminWindow(QWidget, Ui_Model_Manger):
    def __init__(self,parent=None):
        super(AdminWindow, self).__init__(parent)
        self.setupUi(self)
        # 初始化路径
        self.model_dir = "model/"
        self.gallery_dir = "/media/emilia/新加卷/DN ReID/DNDM/gallery"
        self.create_dirs()

        # 初始化信号连接
        self.init_connections()

        # 初始加载数据
        self.refresh_models()
        self.refresh_gallery()

    def create_dirs(self):
        """创建必要的目录"""
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.gallery_dir, exist_ok=True)

    def init_connections(self):
        """连接按钮信号与槽函数"""
        # 模型管理
        self.pushButton.clicked.connect(self.add_model)
        self.pushButton_2.clicked.connect(self.delete_model)
        self.pushButton_3.clicked.connect(self.refresh_models)
        self.pushButton_7.clicked.connect(self.return_to_login)

        # 图库管理
        self.pushButton_4.clicked.connect(self.add_vehicle)
        self.pushButton_5.clicked.connect(self.delete_vehicle)
        self.pushButton_6.clicked.connect(self.refresh_gallery)
        self.pushButton_8.clicked.connect(self.return_to_login)
        # ==================== 模型管理功能 ====================

    def add_model(self):
        """添加模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择模型文件",
            "",
            "Model Files (*.t *.pth *.pt);;All Files (*)"
        )
        if file_path:
            try:
                # 复制文件到模型目录
                dst = os.path.join(self.model_dir, os.path.basename(file_path))
                shutil.copy(file_path, dst)
                self.refresh_models()
                QMessageBox.information(self, "成功", "模型添加成功！")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"添加失败: {str(e)}")

    def delete_model(self):
        """删除选中模型"""
        selected = self.listWidget.currentItem()
        if not selected:
            QMessageBox.warning(self, "警告", "请先选择要删除的模型")
            return

        model_name = selected.text()
        reply = QMessageBox.question(
            self,
            "确认删除",
            f"确定要删除模型 {model_name} 吗？",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                os.remove(os.path.join(self.model_dir, model_name))
                self.refresh_models()
                QMessageBox.information(self, "成功", "模型已删除")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"删除失败: {str(e)}")

    def refresh_models(self):
        """刷新模型列表"""
        self.listWidget.clear()
        models = QDir(self.model_dir).entryList(["*.t", "*.pth", "*.pt"], QDir.Files)
        for model in models:
            QListWidgetItem(model, self.listWidget)

        # ==================== 图库管理功能 ====================

    def add_vehicle(self):
        """添加车辆图片"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择车辆图片",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )
        if files:
            try:
                for file_path in files:
                    dst = os.path.join(self.gallery_dir, os.path.basename(file_path))
                    shutil.copy(file_path, dst)
                self.refresh_gallery()
                QMessageBox.information(self, "成功", f"成功添加 {len(files)} 张图片")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"添加失败: {str(e)}")

    def delete_vehicle(self):
        """删除选中车辆图片"""
        selected = self.listWidget_2.currentItem()
        if not selected:
            QMessageBox.warning(self, "警告", "请先选择要删除的图片")
            return

        img_name = selected.text()
        reply = QMessageBox.question(
            self,
            "确认删除",
            f"确定要删除图片 {img_name} 吗？",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                os.remove(os.path.join(self.gallery_dir, img_name))
                self.refresh_gallery()
                QMessageBox.information(self, "成功", "图片已删除")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"删除失败: {str(e)}")

    def refresh_gallery(self):
        """刷新图库列表"""
        self.listWidget_2.clear()
        images = QDir(self.gallery_dir).entryList(
            ["*.png", "*.jpg", "*.jpeg", "*.bmp"],
            QDir.Files
        )
        for img in images:
            QListWidgetItem(img, self.listWidget_2)

    def return_to_login(self):
        """返回登录界面"""
        self.hide()
        self.main_window = LoginWindow()
        self.main_window.show()

class MainWindow(QWidget, Ui_Form):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        # 绑定上传图片按钮信号
        self.pushButton.clicked.connect(self.upload_image)
        # 绑定开始识别按钮信号（pushButton_2）
        self.pushButton_2.clicked.connect(self.start_recognition)

        self.pushButton_3.clicked.connect(self.return_guid)

        # 保存上传图片路径
        self.image_path = None

        # 初始化模型参数，自动选择 model/ 目录下的 .t 文件
        model_dir = "model/"
        model_files = glob.glob(os.path.join(model_dir, "*.t"))
        if not model_files:
            raise FileNotFoundError(f"No .t model files found in {model_dir}")
        # 选取最新修改的模型文件
        resume_file = sorted(model_files, key=os.path.getctime, reverse=True)[0]
        resume_file = os.path.basename(resume_file)

        self.model_params = test1.load_model(
            model_path=model_dir,
            resume=resume_file,
            dataset="dn348",
            gpu="0",
            img_w=256,
            img_h=256
        )
        # 设置 gallery 文件夹路径
        self.gallery_dir = "/media/emilia/新加卷/DN ReID/DNDM/gallery"

    def return_guid(self):
        self.hide()
        self.main_window = GuidWindow()
        self.main_window.show()

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.image_path = file_path
            pixmap = QPixmap(file_path)
            scaled_pixmap = pixmap.scaled(
                self.label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
            self.label.setPixmap(scaled_pixmap)

    def start_recognition(self):
        # 清空所有结果窗口
        image_labels = [
            self.label_8, self.label_10, self.label_12, self.label_14, self.label_16,
            self.label_18, self.label_20, self.label_22, self.label_24, self.label_26
        ]
        text_labels = [
            self.label_9, self.label_11, self.label_13, self.label_15, self.label_17,
            self.label_19, self.label_21, self.label_23, self.label_25, self.label_27
        ]
        for img_lbl, txt_lbl in zip(image_labels, text_labels):
            img_lbl.clear()
            txt_lbl.setText("")

        if not self.image_path:
            QMessageBox.warning(self, "错误", "请先上传图片！")
            return

        try:
            results = test1.re_identify(
                self.image_path, self.model_params, self.gallery_dir, topk=10
            )
        except Exception as e:
            QMessageBox.critical(self, "错误", "重识别过程中出现异常：" + str(e))
            return

        # 根据结果数量动态展示
        for idx, (img_path, sim) in enumerate(results):
            if idx >= 10:
                break
            pixmap = QPixmap(img_path).scaled(
                image_labels[idx].size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
            image_labels[idx].setPixmap(pixmap)
            text_labels[idx].setText(f"{sim:.2f}")

        if all(sim < 0.7 for _, sim in results):
            # 创建新文件名，按 gallery 中已有图片数量 + 1 命名
            existing_imgs = [f for f in os.listdir(self.gallery_dir)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            new_id = len(existing_imgs) + 1
            ext = os.path.splitext(self.image_path)[-1]  # 获取原文件扩展名
            new_filename = f"{new_id:04d}{ext}"
            save_path = os.path.join(self.gallery_dir, new_filename)

            # 拷贝图片到 gallery 中
            try:
                shutil.copy(self.image_path, save_path)
                QMessageBox.information(
                    self, "已新增图片",
                    f"该图片未匹配到相似车辆（相似度均小于 0.7），已保存到 gallery：{new_filename}"
                )
            except Exception as e:
                QMessageBox.critical(self, "保存失败", f"保存到 gallery 时出错：{str(e)}")

class VideoWindow(QWidget, Ui_Video):
    def __init__(self, parent=None):
        super(VideoWindow, self).__init__(parent)
        self.setupUi(self)

        self.pushButton_4.clicked.connect(self.return_guid)



    def return_guid(self):
        self.hide()
        self.main_window = GuidWindow()
        self.main_window.show()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = LoginWindow()
    window.show()
    sys.exit(app.exec_())
    # video_file = "./video/vehicle.mp4"
    # save_folder = "./vehicle_image"
    # count = run_vehicle_detection(video_file, save_folder)
