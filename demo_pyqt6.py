import sys
import os
import copy
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QFileDialog, QListWidget, QSlider,
                             QComboBox, QCheckBox, QGroupBox, QTextEdit, QLineEdit,
                             QSizePolicy)
from PyQt5.QtCore import Qt
import subprocess
import glob
import numpy as np


try:
    import vtk
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    from vtkmodules.util.numpy_support import numpy_to_vtk
    _HAS_VTK = True
except Exception:
    _HAS_VTK = False

try:
    import open3d as o3d
    _HAS_O3D = True
except Exception:
    _HAS_O3D = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VGGT: Visual Geometry Grounded Transformer")
        self.resize(1600, 900)  # 调整窗口初始尺寸

        main_widget = QWidget()
        # ===== 修改：根水平布局，左控件 + 右点云 =====
        root_layout = QHBoxLayout(main_widget)
        left_container = QWidget()
        left_container.setFixedWidth(420)  # 固定左侧面板宽度
        main_layout = QVBoxLayout(left_container)
        root_layout.addWidget(left_container, stretch=0)

        # 标题区域
        title = QLabel("<h1>VGGT: Visual Geometry Grounded Transformer</h1>")
        main_layout.addWidget(title)

        # 上传区域
        upload_layout = QHBoxLayout()
        self.upload_video_button = QPushButton("上传视频")
        self.upload_images_button = QPushButton("上传图片")
        upload_layout.addWidget(self.upload_video_button)
        upload_layout.addWidget(self.upload_images_button)
        main_layout.addLayout(upload_layout)

        # 预览图廊
        main_layout.addWidget(QLabel("预览图廊："))
        self.gallery_list = QListWidget()
        self.gallery_list.setFixedHeight(120)
        main_layout.addWidget(self.gallery_list)

        # 重建参数设置
        param_group = QGroupBox("重建参数设置")
        param_layout = QVBoxLayout()

        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(0)
        self.conf_slider.setMaximum(1000)  # 修改范围，支持一位小数（0.0 ~ 100.0）
        self.conf_slider.setValue(500)
        self.conf_value_label = QLabel("50.0%")  # 修改默认值显示为一位小数
        self.conf_value_input = QLineEdit("50.0")  # 新增：输入框
        self.conf_value_input.setFixedWidth(50)
        self.conf_value_input.setAlignment(Qt.AlignRight)
        conf_slider_layout = QHBoxLayout()  # 新增：水平布局
        conf_slider_layout.addWidget(self.conf_slider)
        conf_slider_layout.addWidget(self.conf_value_label)
        conf_slider_layout.addWidget(self.conf_value_input)  # 新增：添加输入框
        param_layout.addLayout(conf_slider_layout)

        self.frame_filter_combo = QComboBox()
        self.frame_filter_combo.addItem("All")
        param_layout.addWidget(QLabel("显示特定帧"))
        param_layout.addWidget(self.frame_filter_combo)

        # 复选框设置
        self.show_cam_checkbox = QCheckBox("显示相机")
        self.show_cam_checkbox.setChecked(True)
        self.mask_sky_checkbox = QCheckBox("滤除天空")
        self.mask_black_bg_checkbox = QCheckBox("滤除黑色背景")
        self.mask_white_bg_checkbox = QCheckBox("滤除白色背景")
        param_layout.addWidget(self.show_cam_checkbox)
        param_layout.addWidget(self.mask_sky_checkbox)
        param_layout.addWidget(self.mask_black_bg_checkbox)
        param_layout.addWidget(self.mask_white_bg_checkbox)

        # 使用 BA 复选框
        self.use_ba_checkbox = QCheckBox("使用 BA（Bundle Adjustment）")
        self.use_ba_checkbox.setChecked(False)
        param_layout.addWidget(self.use_ba_checkbox)

        param_group.setLayout(param_layout)
        main_layout.addWidget(param_group)

        #  参数设置
        colmap_param_group = QGroupBox("COLMAP 参数设置")
        colmap_param_layout = QVBoxLayout()

        # 最大重投影误差
        self.max_reproj_error_input = QLineEdit("8.0")  # 默认值
        self.max_reproj_error_input.setFixedWidth(50)
        self.max_reproj_error_input.setAlignment(Qt.AlignRight)
        max_reproj_error_layout = QHBoxLayout()
        max_reproj_error_layout.addWidget(QLabel("最大重投影误差"))
        max_reproj_error_layout.addWidget(self.max_reproj_error_input)
        colmap_param_layout.addLayout(max_reproj_error_layout)

        # 可见性阈值
        self.vis_thresh_input = QLineEdit("0.2")  # 默认值
        self.vis_thresh_input.setFixedWidth(50)
        self.vis_thresh_input.setAlignment(Qt.AlignRight)
        vis_thresh_layout = QHBoxLayout()
        vis_thresh_layout.addWidget(QLabel("可见性阈值"))
        vis_thresh_layout.addWidget(self.vis_thresh_input)
        colmap_param_layout.addLayout(vis_thresh_layout)

        # 最大查询点数
        self.max_query_pts_input = QLineEdit("4096")  # 默认值
        self.max_query_pts_input.setFixedWidth(50)
        self.max_query_pts_input.setAlignment(Qt.AlignRight)
        max_query_pts_layout = QHBoxLayout()
        max_query_pts_layout.addWidget(QLabel("最大查询点数"))
        max_query_pts_layout.addWidget(self.max_query_pts_input)
        colmap_param_layout.addLayout(max_query_pts_layout)

        # 是否启用精细跟踪
        self.fine_tracking_checkbox = QCheckBox("启用精细跟踪")
        self.fine_tracking_checkbox.setChecked(True)  # 默认启用
        colmap_param_layout.addWidget(self.fine_tracking_checkbox)

        # 随机种子
        self.seed_input = QLineEdit("42")  # 默认值
        self.seed_input.setFixedWidth(50)
        self.seed_input.setAlignment(Qt.AlignRight)
        seed_layout = QHBoxLayout()
        seed_layout.addWidget(QLabel("随机种子"))
        seed_layout.addWidget(self.seed_input)
        colmap_param_layout.addLayout(seed_layout)

        colmap_param_group.setLayout(colmap_param_layout)
        main_layout.addWidget(colmap_param_group)

        # 重建和日志区域
        self.reconstruct_button = QPushButton("重建")
        main_layout.addWidget(self.reconstruct_button)

        main_layout.addWidget(QLabel("日志输出："))
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        main_layout.addWidget(self.log_output)

        self.setCentralWidget(main_widget)

        # ===== 修改右侧点云预览：使用 VTK =====
        self.viewer_group = QGroupBox("点云预览 (.ply)")
        viewer_layout = QVBoxLayout()
        if _HAS_VTK:
            self.vtk_widget = QVTKRenderWindowInteractor()
            viewer_layout.addWidget(self.vtk_widget, 1)
            self.vtk_renderer = vtk.vtkRenderer()
            self.vtk_widget.GetRenderWindow().AddRenderer(self.vtk_renderer)
            self.vtk_renderer.SetBackground(0.15, 0.15, 0.18)
            self.vtk_widget.Initialize()
            self.vtk_widget.Start()
            self.viewer = self.vtk_widget
            self._bg_dark = True
            # --- 新增：坐标轴组件 ---
            self._axes_widget = None
            self._init_axes()
        else:
            self.vtk_widget = None
            self.vtk_renderer = None
            viewer_layout.addWidget(QLabel("未安装 VTK，无法显示点云。pip install vtk"))
            self.viewer = None
        # --- 新增按钮/控件：打开点云、刷新、点大小、坐标轴开关 ---
        btn_row = QHBoxLayout()
        self.open_ply_button = QPushButton("打开点云文件")
        self.refresh_ply_button = QPushButton("刷新点云")
        btn_row.addWidget(self.open_ply_button)
        btn_row.addWidget(self.refresh_ply_button)
        viewer_layout.addLayout(btn_row)

        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("点大小"))
        self.point_size_slider = QSlider(Qt.Horizontal)
        self.point_size_slider.setMinimum(1)
        self.point_size_slider.setMaximum(15)
        self.point_size_slider.setValue(3)
        self.point_size_value_label = QLabel("3")
        size_layout.addWidget(self.point_size_slider, 1)
        size_layout.addWidget(self.point_size_value_label)
        viewer_layout.addLayout(size_layout)

        self.show_axes_checkbox = QCheckBox("显示坐标轴")
        self.show_axes_checkbox.setChecked(True)
        viewer_layout.addWidget(self.show_axes_checkbox)
        # 修复：遗漏 setLayout 导致 layout() 为 None
        self.viewer_group.setLayout(viewer_layout)
        root_layout.addWidget(self.viewer_group, stretch=1)
        self.viewer_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        viewer_layout.setContentsMargins(2, 2, 2, 2)

        # 新增：存储上传的文件路径（视频或图片）
        self.uploaded_files = []
        self.current_target_dir = None
        # ===== 新增状态 =====
        self.ply_path = None
        # ===== 状态变量：替换 _viewer_items 为 VTK actor =====
        self._vtk_actor = None  # 替换旧 viewer items

        # 绑定信号槽
        self.upload_video_button.clicked.connect(self.upload_video)
        self.upload_images_button.clicked.connect(self.upload_images)
        self.reconstruct_button.clicked.connect(self.reconstruct)
        self.conf_slider.valueChanged.connect(self.update_conf_value_label)
        self.conf_value_input.editingFinished.connect(self.update_conf_slider_value)
        # ===== 新增信号 =====
        self.refresh_ply_button.clicked.connect(self.refresh_point_cloud)
        self.open_ply_button.clicked.connect(self.open_point_cloud)
        self.point_size_slider.valueChanged.connect(self.update_point_size)
        self.show_axes_checkbox.toggled.connect(self.toggle_axes)

    # --- 新增：初始化坐标轴 ---
    def _init_axes(self):
        if not _HAS_VTK:
            return
        axes = vtk.vtkAxesActor()
        # 优化视觉：调细轴/锥体大小
        axes.SetTotalLength(1.0, 1.0, 1.0)
        axes.SetShaftTypeToCylinder()
        axes.SetCylinderRadius(0.02)
        self._axes_widget = vtk.vtkOrientationMarkerWidget()
        self._axes_widget.SetOrientationMarker(axes)
        self._axes_widget.SetViewport(0.0, 0.0, 0.18, 0.18)
        self._axes_widget.SetInteractor(self.vtk_widget)
        self._axes_widget.EnabledOn()
        self._axes_widget.InteractiveOff()

    # --- 新增：打开点云文件 ---
    def open_point_cloud(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择点云文件",
            "",
            "Point Cloud (*.ply *.pcd *.xyz *.pts);;All Files (*)"
        )
        if not file_path:
            return
        self.ply_path = file_path
        self.log_output.append(f"打开点云文件：{file_path}")
        self.load_point_cloud(file_path)

    # --- 新增：坐标轴开关 ---
    def toggle_axes(self, checked: bool):
        if not _HAS_VTK or self._axes_widget is None:
            return
        if checked:
            self._axes_widget.EnabledOn()
        else:
            self._axes_widget.EnabledOff()
        self.vtk_widget.GetRenderWindow().Render()

    def upload_video(self):
        video_file, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4 *.avi)")
        if video_file:
            self.log_output.append(f"已上传视频：{video_file}")
            self.gallery_list.addItem(os.path.basename(video_file))
            # 保存上传的视频路径
            self.uploaded_files = [video_file]

    def upload_images(self):
        images, _ = QFileDialog.getOpenFileNames(self, "选择图片文件", "", "图片文件 (*.png *.jpg *.jpeg)")
        if images:
            for img in images:
                self.log_output.append(f"已上传图片：{img}")
                self.gallery_list.addItem(os.path.basename(img))
            # 保存上传的图片路径
            self.uploaded_files = images

    def reconstruct(self):
        self.log_output.append("开始重建...")
        if not self.uploaded_files:
            self.log_output.append("没有上传文件。")
            return
        try:
            import cv2, shutil, time, glob
            self.log_output.append("准备临时目录...")
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            temp_dir = os.path.join(os.path.dirname(__file__), f"temp_scene_{timestamp}")
            images_dir = os.path.join(temp_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
            self.current_target_dir = temp_dir
            # 抽帧/复制
            for file in self.uploaded_files:
                if file.lower().endswith((".mp4", ".avi")):
                    cap = cv2.VideoCapture(file)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    step = int(fps) if fps and fps > 0 else 1
                    idx = 0; out_idx = 0
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        idx += 1
                        if idx % step == 0:
                            cv2.imwrite(os.path.join(images_dir, f"{out_idx:06}.png"), frame)
                            out_idx += 1
                    cap.release()
                else:
                    shutil.copy(file, os.path.join(images_dir, os.path.basename(file)))
            image_paths = sorted(glob.glob(os.path.join(images_dir, "*")))
            if not image_paths:
                raise ValueError("未生成任何图像。")
            self.log_output.append(f"图像就绪：{len(image_paths)} 张。")

            # 参数映射
            # 原逻辑将 slider/10 作为阈值，导致阈值范围 0~100 过大，易产生空点云
            # 这里保留显示为百分比( value/10 )，但实际用于重建的阈值改为 value/100 (0~10)
            conf_val = float(self.conf_slider.value()) / 100.0  # 50.0% -> 5.0
            self.log_output.append(f"使用深度置信度阈值: {conf_val:.2f}")
            use_ba = self.use_ba_checkbox.isChecked()  # 代表开启 BA
            max_reproj_error = self.max_reproj_error_input.text()  # 从输入框获取最大重投影误差
            vis_thresh = self.vis_thresh_input.text()  # 从输入框获取可见性阈值
            max_query_pts = self.max_query_pts_input.text()  # 从输入框获取最大查询点数
            fine_tracking = self.fine_tracking_checkbox.isChecked()  # 获取精细跟踪选项
            seed = self.seed_input.text()  # 从输入框获取随机种子

            # 组装命令
            script_path = os.path.join(os.path.dirname(__file__), "demo_colmap.py")
            cmd = [
                sys.executable, script_path,
                "--scene_dir", temp_dir,
                "--seed", seed,
                "--max_reproj_error", max_reproj_error,
                "--vis_thresh", vis_thresh,
                "--max_query_pts", max_query_pts,
            ]
            if fine_tracking:
                cmd.append("--fine_tracking")
            if use_ba:
                cmd.append("--use_ba")
            else:
                cmd += ["--conf_thres_value", f"{conf_val:.2f}"]

            self.log_output.append("执行命令：" + " ".join(cmd))
            self.log_output.append("开始外部重建进程（可能需要较长时间，请等待）...")

            # 启动子进程并实时读取输出
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    self.log_output.append(line)
                    QApplication.processEvents()
            ret = proc.wait()
            if ret != 0:
                self.log_output.append(f"重建进程退出码：{ret}，可能失败。")
                return
            self.log_output.append("外部重建完成。输出目录：{}".format(os.path.join(temp_dir, "sparse")))

            # 更新下拉
            self.frame_filter_combo.clear()
            self.frame_filter_combo.addItem("All")
            for i, p in enumerate(image_paths):
                self.frame_filter_combo.addItem(f"{i}: {os.path.basename(p)}")
            self.log_output.append("完成：COLMAP sparse 已生成。")
            # ===== 新增：自动查找并加载 .ply =====
            if self.current_target_dir:
                ply_files = glob.glob(os.path.join(self.current_target_dir, "**", "*.ply"), recursive=True)
                if ply_files:
                    self.ply_path = ply_files[0]
                    self.log_output.append(f"检测到点云：{self.ply_path}")
                    self.load_point_cloud(self.ply_path)
                else:
                    self.log_output.append("未找到 .ply 点云文件。")
        except Exception as e:
            self.log_output.append(f"重建失败：{e}")

    def update_conf_value_label(self, value):
        conf_value = value / 10.0  # 将滑块值转换为一位小数
        self.conf_value_label.setText(f"{conf_value:.1f}%")
        self.conf_value_input.setText(f"{conf_value:.1f}")  # 同步更新输入框

    def update_conf_slider_value(self):
        try:
            value = float(self.conf_value_input.text()) * 10
            if 0 <= value <= 1000:
                self.conf_slider.setValue(int(value))
            else:
                raise ValueError
        except ValueError:
            self.log_output.append("请输入有效的置信度值（0.0 ~ 100.0）。")
            self.conf_value_input.setText(f"{self.conf_slider.value() / 10.0:.1f}")  # 恢复为滑块当前值

    # ===== 刷新点云 =====
    def refresh_point_cloud(self):
        if not self.ply_path:
            self.log_output.append("暂无可刷新点云。")
            return
        self.load_point_cloud(self.ply_path)

    def update_point_size(self, val: int):
        # 修改：同步标签
        if hasattr(self, "point_size_value_label"):
            self.point_size_value_label.setText(str(val))
        if self._vtk_actor is not None:
            self._vtk_actor.GetProperty().SetPointSize(val)
            if hasattr(self._vtk_actor.GetProperty(), "SetRenderPointsAsSpheres"):
                self._vtk_actor.GetProperty().SetRenderPointsAsSpheres(val >= 3)
            self.vtk_widget.GetRenderWindow().Render()

    def toggle_background(self):
        if not _HAS_VTK:
            return
        if self._bg_dark:
            self.vtk_renderer.SetBackground(0.95, 0.95, 0.97)
        else:
            self.vtk_renderer.SetBackground(0.15, 0.15, 0.18)
        self._bg_dark = not self._bg_dark
        self.vtk_widget.GetRenderWindow().Render()

    # ===== 点云加载=====
    def load_point_cloud(self, path: str):
        if not _HAS_VTK:
            self.log_output.append("当前环境未启用 VTK，无法加载点云。")
            return
        if not os.path.isfile(path):
            self.log_output.append(f"点云文件不存在：{path}")
            return
        try:
            # 优先使用 open3d 读取
            colors = None
            if _HAS_O3D:
                pcd = o3d.io.read_point_cloud(path)
                pts = np.asarray(pcd.points)
                if pts.size == 0:
                    raise ValueError("点云为空。")
                if pcd.has_colors():
                    colors = np.asarray(pcd.colors)
                    if colors.dtype != np.float32 and colors.dtype != np.float64:
                        colors = colors.astype(np.float32)
                    colors = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
            else:
                # 简易 PLY 解析（无颜色）作为兜底
                with open(path, "r") as f:
                    header = []
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        header.append(line.strip())
                        if line.strip() == "end_header":
                            break
                    data = np.loadtxt(f)
                if data.shape[1] >= 3:
                    pts = data[:, :3]
                else:
                    raise ValueError("无法解析点云坐标。")

            # 构建 VTK 点数据
            points = vtk.vtkPoints()
            points.SetData(numpy_to_vtk(pts, deep=True))
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)

            if colors is not None and len(colors) == len(pts):
                vtk_colors = vtk.vtkUnsignedCharArray()
                vtk_colors.SetNumberOfComponents(3)
                vtk_colors.SetName("Colors")
                vtk_colors.SetNumberOfTuples(len(colors))
                # 性能：批量写入
                for i, c in enumerate(colors):
                    vtk_colors.SetTuple3(i, int(c[0]), int(c[1]), int(c[2]))
                polydata.GetPointData().SetScalars(vtk_colors)

            glyph = vtk.vtkVertexGlyphFilter()
            glyph.SetInputData(polydata)
            glyph.Update()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(glyph.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

            # 移除旧 actor
            if self._vtk_actor is not None:
                self.vtk_renderer.RemoveActor(self._vtk_actor)
            self._vtk_actor = actor
            self.vtk_renderer.AddActor(actor)
            self.vtk_renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()

            self.log_output.append(f"点云加载完成：{os.path.basename(path)}，点数 {len(pts)}")
            if colors is not None:
                self.log_output.append("检测到颜色并已应用。")
            else:
                self.log_output.append("未检测到颜色，使用默认灰/白渲染。")
        except Exception as e:
            self.log_output.append(f"点云加载失败：{e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())