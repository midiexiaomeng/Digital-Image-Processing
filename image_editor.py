import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, 
                            QLabel, QWidget, QVBoxLayout, QHBoxLayout,
                            QPushButton, QSlider, QComboBox, QGraphicsView, 
                            QGraphicsScene, QSplitter, QToolBar, QStatusBar,
                            QInputDialog)
from PyQt5.QtGui import QPixmap, QImage, QIcon, QColor, QPen, QPainter
from PyQt5.QtCore import Qt, QSize, QPointF, QRectF

class ImageEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("专业图像处理软件")
        self.setGeometry(100, 100, 1200, 800)
        
        # 初始化UI
        self.initUI()
        
        # 图像数据
        self.image_data = None
        self.original_image = None
        self.history_stack = []
        self.history_index = -1
        
    def initUI(self):
        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 创建工具栏
        self.createToolBar()
        
        # 创建状态栏
        self.statusBar().showMessage("就绪")
        
        # 裁剪相关变量
        self.crop_start = None
        self.crop_end = None
        self.crop_rect = None
        
        # 历史记录限制
        self.max_history = 10
        
        # 主布局
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        
        # 使用QSplitter创建可调整的视图
        splitter = QSplitter(Qt.Horizontal)
        
        # 原图视图
        self.original_view = QGraphicsView()
        self.original_scene = QGraphicsScene()
        self.original_view.setScene(self.original_scene)
        
        # 处理后视图
        self.processed_view = QGraphicsView()
        self.processed_scene = QGraphicsScene()
        self.processed_view.setScene(self.processed_scene)
        
        splitter.addWidget(self.original_view)
        splitter.addWidget(self.processed_view)
        splitter.setSizes([600, 600])
        
        main_layout.addWidget(splitter, 80)
        
        # 控制面板
        control_panel = QVBoxLayout()
        main_layout.addLayout(control_panel, 20)
        
        # 功能按钮
        self.load_btn = QPushButton("加载图像")
        self.load_btn.setIcon(QIcon.fromTheme("document-open"))
        self.load_btn.clicked.connect(self.load_image)
        control_panel.addWidget(self.load_btn)
        
        self.save_btn = QPushButton("保存图像")
        self.save_btn.setIcon(QIcon.fromTheme("document-save"))
        self.save_btn.clicked.connect(self.save_image)
        control_panel.addWidget(self.save_btn)
        
        self.undo_btn = QPushButton("撤回")
        self.undo_btn.setIcon(QIcon.fromTheme("edit-undo"))
        self.undo_btn.clicked.connect(self.undo_action)
        control_panel.addWidget(self.undo_btn)
        
        self.redo_btn = QPushButton("重做")
        self.redo_btn.setIcon(QIcon.fromTheme("edit-redo"))
        self.redo_btn.clicked.connect(self.redo_action)
        control_panel.addWidget(self.redo_btn)
        
        # 功能按钮
        self.grayscale_btn = QPushButton("灰度转换")
        self.grayscale_btn.clicked.connect(lambda: self.apply_processing("灰度转换"))
        control_panel.addWidget(self.grayscale_btn)
        
        self.gaussian_btn = QPushButton("高斯模糊")
        self.gaussian_btn.clicked.connect(lambda: self.apply_processing("高斯模糊"))
        control_panel.addWidget(self.gaussian_btn)
        
        self.edge_btn = QPushButton("边缘检测")
        self.edge_btn.clicked.connect(lambda: self.apply_processing("边缘检测"))
        control_panel.addWidget(self.edge_btn)
        
        # 旋转图像控件
        self.rotate_btn = QPushButton("旋转图像")
        self.rotate_btn.clicked.connect(lambda: self.apply_processing("旋转图像"))
        control_panel.addWidget(self.rotate_btn)
        
        self.rotate_combo = QComboBox()
        self.rotate_combo.addItems(["90度", "180度", "270度"])
        control_panel.addWidget(QLabel("旋转角度:"))
        control_panel.addWidget(self.rotate_combo)
        
        self.crop_btn = QPushButton("裁剪图像")
        self.crop_btn.clicked.connect(self.enable_crop_mode)
        control_panel.addWidget(self.crop_btn)
        
        self.histogram_btn = QPushButton("直方图")
        self.histogram_btn.clicked.connect(lambda: self.apply_processing("直方图"))
        control_panel.addWidget(self.histogram_btn)
        
        self.threshold_btn = QPushButton("阈值处理")
        self.threshold_btn.clicked.connect(lambda: self.apply_processing("阈值处理"))
        control_panel.addWidget(self.threshold_btn)
        
        self.morphology_btn = QPushButton("形态学变换")
        self.morphology_btn.clicked.connect(lambda: self.apply_processing("形态学变换"))
        control_panel.addWidget(self.morphology_btn)
        
        self.flip_btn = QPushButton("翻转图像")
        self.flip_btn.clicked.connect(lambda: self.apply_processing("翻转图像"))
        control_panel.addWidget(self.flip_btn)
        
        self.segmentation_btn = QPushButton("图像分割")
        self.segmentation_btn.clicked.connect(lambda: self.apply_processing("图像分割"))
        control_panel.addWidget(self.segmentation_btn)
        
        self.filter_btn = QPushButton("图像滤波")
        self.filter_btn.clicked.connect(lambda: self.apply_processing("图像滤波"))
        control_panel.addWidget(self.filter_btn)
        
        # 边缘检测阈值
        self.edge_threshold_label = QLabel("边缘检测阈值: 50 (0-100)")
        control_panel.addWidget(self.edge_threshold_label)
        
        self.edge_threshold_slider = QSlider(Qt.Horizontal)
        self.edge_threshold_slider.setRange(0, 100)
        self.edge_threshold_slider.setValue(50)
        self.edge_threshold_slider.valueChanged.connect(
            lambda v: self.edge_threshold_label.setText(f"边缘检测阈值: {v} (0-100)"))
        control_panel.addWidget(self.edge_threshold_slider)

        # 低频滤波参数
        self.low_freq_label = QLabel("低频截止: 50 (0-100)")
        control_panel.addWidget(self.low_freq_label)
        
        self.low_freq_slider = QSlider(Qt.Horizontal)
        self.low_freq_slider.setRange(0, 100)
        self.low_freq_slider.setValue(50)
        self.low_freq_slider.valueChanged.connect(
            lambda v: self.low_freq_label.setText(f"低频截止: {v} (0-100)"))
        control_panel.addWidget(self.low_freq_slider)

        # 高频滤波参数
        self.high_freq_label = QLabel("高频截止: 50 (0-100)")
        control_panel.addWidget(self.high_freq_label)
        
        self.high_freq_slider = QSlider(Qt.Horizontal)
        self.high_freq_slider.setRange(0, 100)
        self.high_freq_slider.setValue(50)
        self.high_freq_slider.valueChanged.connect(
            lambda v: self.high_freq_label.setText(f"高频截止: {v} (0-100)"))
        control_panel.addWidget(self.high_freq_slider)
        
        # 亮度调整
        self.brightness_label = QLabel("亮度: 0 (-100到100)")
        control_panel.addWidget(self.brightness_label)
        
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(
            lambda v: self.brightness_label.setText(f"亮度: {v} (-100到100)"))
        self.brightness_slider.valueChanged.connect(
            lambda: self.apply_processing("亮度调整"))
        control_panel.addWidget(self.brightness_slider)
        
        # 对比度调整
        self.contrast_label = QLabel("对比度: 0 (-100到100)")
        control_panel.addWidget(self.contrast_label)
        
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(-100, 100)
        self.contrast_slider.setValue(0)
        self.contrast_slider.valueChanged.connect(
            lambda v: self.contrast_label.setText(f"对比度: {v} (-100到100)"))
        self.contrast_slider.valueChanged.connect(
            lambda: self.apply_processing("对比度调整"))
        control_panel.addWidget(self.contrast_slider)

        # 翻转方向控制
        self.flip_label = QLabel("翻转方向: 水平 (左:水平, 右:垂直)")
        control_panel.addWidget(self.flip_label)
        
        self.flip_slider = QSlider(Qt.Horizontal)
        self.flip_slider.setRange(0, 100)
        self.flip_slider.setValue(0)
        self.flip_slider.valueChanged.connect(
            lambda v: self.flip_label.setText(
                "翻转方向: 水平" if v <= 50 else "翻转方向: 垂直"))
        control_panel.addWidget(self.flip_slider)
        
        # 形态学变换选项
        self.morphology_combo = QComboBox()
        self.morphology_combo.addItems(["腐蚀", "膨胀", "开运算", "闭运算"])
        control_panel.addWidget(QLabel("形态学操作:"))
        control_panel.addWidget(self.morphology_combo)
        
        self.kernel_size_combo = QComboBox()
        self.kernel_size_combo.addItems(["3x3", "5x5", "7x7"])
        control_panel.addWidget(QLabel("核大小:"))
        control_panel.addWidget(self.kernel_size_combo)
        
        # 应用按钮
        self.apply_btn = QPushButton("应用处理")
        self.apply_btn.setIcon(QIcon.fromTheme("system-run"))
        self.apply_btn.clicked.connect(self.apply_processing)
        control_panel.addWidget(self.apply_btn)
        
        # 添加样式表
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QPushButton {
                padding: 5px;
                min-width: 80px;
                background-color: #e0e0e0;
                border: 1px solid #a0a0a0;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
            QComboBox {
                padding: 3px;
                border: 1px solid #a0a0a0;
                border-radius: 3px;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #c0c0c0;
                margin: 2px 0;
            }
            QSlider::handle:horizontal {
                background: #505050;
                width: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }
        """)
        
        # 添加拉伸使布局更美观
        control_panel.addStretch()
        
        # 连接鼠标事件
        self.original_view.mousePressEvent = self.mouse_press
        self.original_view.mouseMoveEvent = self.mouse_move
        self.original_view.mouseReleaseEvent = self.mouse_release
        
    def createToolBar(self):
        """创建工具栏"""
        toolbar = QToolBar("主工具栏")
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)
        
        # 添加工具栏动作
        load_action = toolbar.addAction(QIcon.fromTheme("document-open"), "加载")
        load_action.triggered.connect(self.load_image)
        
        save_action = toolbar.addAction(QIcon.fromTheme("document-save"), "保存")
        save_action.triggered.connect(self.save_image)
        
        toolbar.addSeparator()
        
        undo_action = toolbar.addAction(QIcon.fromTheme("edit-undo"), "撤回")
        undo_action.triggered.connect(self.undo_action)
        
        redo_action = toolbar.addAction(QIcon.fromTheme("edit-redo"), "重做")
        redo_action.triggered.connect(self.redo_action)
        
        toolbar.addSeparator()
        
        exit_action = toolbar.addAction(QIcon.fromTheme("application-exit"), "退出")
        exit_action.triggered.connect(self.close)

    def enable_crop_mode(self):
        """启用裁剪模式"""
        self.statusBar().showMessage("请在原图上拖动鼠标选择裁剪区域")
        self.crop_start = None
        self.crop_end = None
        self.crop_rect = None

    def mouse_press(self, event):
        """鼠标按下事件"""
        if self.crop_btn.isChecked():
            pos = self.original_view.mapToScene(event.pos())
            self.crop_start = QPointF(pos.x(), pos.y())
            self.crop_end = None
            self.crop_rect = None
            self.update_image_views()

    def mouse_move(self, event):
        """鼠标移动事件"""
        if self.crop_btn.isChecked() and self.crop_start:
            pos = self.original_view.mapToScene(event.pos())
            self.crop_end = QPointF(pos.x(), pos.y())
            
            # 创建虚线矩形
            if self.crop_start and self.crop_end:
                x1, y1 = self.crop_start.x(), self.crop_start.y()
                x2, y2 = self.crop_end.x(), self.crop_end.y()
                self.crop_rect = QRectF(min(x1, x2), min(y1, y2), abs(x2-x1), abs(y2-y1))
            
            self.update_image_views()

    def mouse_release(self, event):
        """鼠标释放事件"""
        if self.crop_btn.isChecked() and self.crop_start:
            pos = self.original_view.mapToScene(event.pos())
            self.crop_end = QPointF(pos.x(), pos.y())
            self.crop_image()

    def draw_crop_rect(self, scene):
        """绘制裁剪矩形"""
        if self.crop_rect:
            pen = QPen(Qt.red, 1, Qt.DashLine)
            scene.addRect(self.crop_rect, pen)

    def update_image_views(self):
        """更新原图和处理后视图"""
        if self.image_data is None:
            return
            
        # 清除场景
        self.original_scene.clear()
        self.processed_scene.clear()
        
        # 显示原图
        if len(self.original_image.shape) == 2:  # 灰度图像
            height, width = self.original_image.shape
            img_bytes = np.ascontiguousarray(self.original_image).data
            q_img = QImage(img_bytes, width, height, width, 
                         QImage.Format_Grayscale8)
        else:  # 彩色图像
            height, width, _ = self.original_image.shape
            img_bytes = np.ascontiguousarray(self.original_image).data
            bytes_per_line = 3 * width
            q_img = QImage(img_bytes, width, height, bytes_per_line, 
                         QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.original_scene.addPixmap(pixmap)
        
        # 绘制裁剪矩形
        self.draw_crop_rect(self.original_scene)
        
        self.original_view.fitInView(self.original_scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        
        # 显示处理后图像
        if len(self.image_data.shape) == 2:  # 灰度图像
            height, width = self.image_data.shape
            img_bytes = np.ascontiguousarray(self.image_data).data
            q_img = QImage(img_bytes, width, height, width, 
                         QImage.Format_Grayscale8)
        else:  # 彩色图像
            height, width, _ = self.image_data.shape
            img_bytes = np.ascontiguousarray(self.image_data).data
            bytes_per_line = 3 * width
            q_img = QImage(img_bytes, width, height, bytes_per_line, 
                         QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.processed_scene.addPixmap(pixmap)
        self.processed_view.fitInView(self.processed_scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def crop_image(self):
        """裁剪图像"""
        if not self.crop_start or not self.crop_end or self.image_data is None:
            return
            
        # 获取裁剪区域坐标
        x1, y1 = self.crop_start.x(), self.crop_start.y()
        x2, y2 = self.crop_end.x(), self.crop_end.y()
        
        # 确保x1,y1是左上角，x2,y2是右下角
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # 转换为整数坐标
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # 裁剪图像
        if len(self.image_data.shape) == 3:  # 彩色图像
            self.image_data = self.image_data[y1:y2, x1:x2, :]
        else:  # 灰度图像
            self.image_data = self.image_data[y1:y2, x1:x2]
            
            # 更新显示
            self.update_image_views()
            
            # 重置裁剪状态
            self.crop_start = None
            self.crop_end = None
            self.crop_rect = None
            self.crop_btn.setChecked(False)
            self.statusBar().showMessage("裁剪完成")

    def flip_image(self):
        """翻转图像"""
        if self.image_data is None:
            return
            
        # 根据滑块值确定翻转方向 (0-50:水平翻转, 51-100:垂直翻转)
        if self.flip_slider.value() <= 50:
            # 水平翻转
            self.image_data = np.fliplr(self.image_data)
            self.statusBar().showMessage("图像已水平翻转")
        else:
            # 垂直翻转
            self.image_data = np.flipud(self.image_data)
            self.statusBar().showMessage("图像已垂直翻转")
        
        # 更新显示
        self.update_image_views()
