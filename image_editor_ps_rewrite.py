import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
from PIL import Image, ImageTk
import math

class ImageProcessor:
    @staticmethod
    def to_grayscale(image):
        """转换为灰度图像"""
        if len(image.shape) == 3:
            return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        return image

    @staticmethod
    def gaussian_blur(image, kernel_size=3):
        """高斯模糊"""
        kernel = ImageProcessor._create_gaussian_kernel(kernel_size)
        return ImageProcessor._apply_kernel(image, kernel)

    @staticmethod
    def _create_gaussian_kernel(size, sigma=1.0):
        """创建高斯核"""
        kernel = np.zeros((size, size))
        center = size // 2
        for i in range(size):
            for j in range(size):
                x, y = i - center, j - center
                kernel[i, j] = math.exp(-(x**2 + y**2)/(2*sigma**2))
        return kernel / np.sum(kernel)

    @staticmethod
    def _apply_kernel(image, kernel):
        """应用卷积核"""
        if len(image.shape) == 3:
            return np.dstack([ImageProcessor._apply_kernel_single_channel(image[:,:,i], kernel) 
                            for i in range(3)])
        return ImageProcessor._apply_kernel_single_channel(image, kernel)

    @staticmethod
    def _apply_kernel_single_channel(channel, kernel):
        """单通道卷积"""
        pad_size = kernel.shape[0] // 2
        padded = np.pad(channel, pad_size, mode='constant')
        result = np.zeros_like(channel)
        for i in range(channel.shape[0]):
            for j in range(channel.shape[1]):
                result[i,j] = np.sum(padded[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def edge_detection(image):
        """边缘检测"""
        if len(image.shape) == 3:
            image = ImageProcessor.to_grayscale(image)
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        grad_x = ImageProcessor._apply_kernel_single_channel(image, sobel_x)
        grad_y = ImageProcessor._apply_kernel_single_channel(image, sobel_y)
        return np.sqrt(grad_x**2 + grad_y**2).astype(np.uint8)

    @staticmethod
    def rotate_image(image, angle):
        """旋转图像"""
        if angle == 90:
            return np.rot90(image)
        elif angle == 180:
            return np.rot90(image, 2)
        elif angle == 270:
            return np.rot90(image, 3)
        return image

    @staticmethod
    def adjust_brightness_contrast(image, brightness=0, contrast=0):
        """调整亮度和对比度"""
        image = image.astype(np.float32)
        if brightness != 0:
            image = image + brightness
        if contrast != 0:
            factor = (259 * (contrast + 255)) / (255 * (259 - contrast))
            image = factor * (image - 128) + 128
        return np.clip(image, 0, 255).astype(np.uint8)

    @staticmethod
    def threshold(image, threshold=128, mode='binary'):
        """阈值处理"""
        if len(image.shape) == 3:
            image = ImageProcessor.to_grayscale(image)
        if mode == 'binary':
            return (image > threshold) * 255
        elif mode == 'binary_inv':
            return (image <= threshold) * 255
        elif mode == 'trunc':
            return np.where(image > threshold, threshold, image)
        elif mode == 'tozero':
            return np.where(image <= threshold, 0, image)
        elif mode == 'tozero_inv':
            return np.where(image > threshold, 0, image)
        return image

    @staticmethod
    def morphological_transform(image, operation, kernel_size=3):
        """形态学变换"""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if operation == 'erosion':
            return ImageProcessor._erode(image, kernel)
        elif operation == 'dilation':
            return ImageProcessor._dilate(image, kernel)
        elif operation == 'opening':
            return ImageProcessor._dilate(ImageProcessor._erode(image, kernel), kernel)
        elif operation == 'closing':
            return ImageProcessor._erode(ImageProcessor._dilate(image, kernel), kernel)
        return image

    @staticmethod
    def _erode(image, kernel):
        """腐蚀运算"""
        pad_size = kernel.shape[0] // 2
        padded = np.pad(image, pad_size, mode='constant')
        result = np.zeros_like(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                result[i,j] = np.min(padded[i:i+kernel.shape[0], j:j+kernel.shape[1]][kernel == 1])
        return result

    @staticmethod
    def _dilate(image, kernel):
        """膨胀运算"""
        pad_size = kernel.shape[0] // 2
        padded = np.pad(image, pad_size, mode='constant')
        result = np.zeros_like(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                result[i,j] = np.max(padded[i:i+kernel.shape[0], j:j+kernel.shape[1]][kernel == 1])
        return result

    @staticmethod
    def flip_image(image, direction='horizontal'):
        """翻转图像"""
        if direction == 'horizontal':
            return image[:, ::-1]
        elif direction == 'vertical':
            return image[::-1, :]
        return image

    @staticmethod
    def image_segmentation(image, method='threshold'):
        """图像分割"""
        if method == 'threshold':
            gray = ImageProcessor.to_grayscale(image) if len(image.shape) == 3 else image
            return (gray > 128) * 255
        return image

    @staticmethod
    def image_filter(image, low_threshold=0, high_threshold=255):
        """图像滤波"""
        if len(image.shape) == 3:
            gray = ImageProcessor.to_grayscale(image)
        else:
            gray = image
        return np.where((gray >= low_threshold) & (gray <= high_threshold), gray, 0)

class HistoryManager:
    def __init__(self):
        self.undo_stack = []
        self.redo_stack = []

    def push_state(self, state):
        self.undo_stack.append(state.copy())
        self.redo_stack = []

    def undo(self):
        if len(self.undo_stack) > 1:
            state = self.undo_stack.pop()
            self.redo_stack.append(state)
            return self.undo_stack[-1].copy()
        return None

    def redo(self):
        if self.redo_stack:
            state = self.redo_stack.pop()
            self.undo_stack.append(state)
            return state.copy()
        return None

class ImageEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("图像处理软件")
        
        # 初始化变量
        self.original_image = None
        self.processed_image = None
        self.crop_start = None
        self.crop_rect = None
        self.history_manager = HistoryManager()
        
        # 创建GUI
        self.create_widgets()
        
    def create_widgets(self):
        """创建GUI组件"""
        # 图像显示区域
        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        # 原始图像标签
        self.original_label = tk.Label(self.image_frame, width=400, height=400, relief=tk.SUNKEN)
        self.original_label.pack(side=tk.LEFT)
        
        # 处理后图像标签
        self.processed_label = tk.Label(self.image_frame, width=400, height=400, relief=tk.SUNKEN)
        self.processed_label.pack(side=tk.LEFT)
        
        # 控制面板
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(side=tk.RIGHT, padx=10, pady=10)
        
        # 文件操作按钮
        self.file_frame = tk.LabelFrame(self.control_frame, text="文件操作")
        self.file_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(self.file_frame, text="打开图像", command=self.open_image).pack(side=tk.LEFT, padx=5)
        tk.Button(self.file_frame, text="保存图像", command=self.save_image).pack(side=tk.LEFT, padx=5)
        
        # 图像处理按钮
        self.process_frame = tk.LabelFrame(self.control_frame, text="图像处理")
        self.process_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(self.process_frame, text="灰度转换", command=lambda: self.apply_processing(ImageProcessor.to_grayscale)).pack(fill=tk.X, pady=2)
        tk.Button(self.process_frame, text="高斯模糊", command=lambda: self.apply_processing(ImageProcessor.gaussian_blur, 3)).pack(fill=tk.X, pady=2)
        tk.Button(self.process_frame, text="边缘检测", command=lambda: self.apply_processing(ImageProcessor.edge_detection)).pack(fill=tk.X, pady=2)
        
        # 旋转选项
        self.rotate_frame = tk.Frame(self.process_frame)
        self.rotate_frame.pack(fill=tk.X, pady=2)
        tk.Button(self.rotate_frame, text="旋转图像", command=self.rotate_image).pack(side=tk.LEFT)
        self.rotate_var = tk.StringVar(value="90")
        tk.OptionMenu(self.rotate_frame, self.rotate_var, "90", "180", "270").pack(side=tk.LEFT)
        
        tk.Button(self.process_frame, text="裁剪图像", command=self.start_crop).pack(fill=tk.X, pady=2)
        tk.Button(self.process_frame, text="显示直方图", command=self.show_histogram).pack(fill=tk.X, pady=2)
        
        # 亮度和对比度滑块
        self.bc_frame = tk.LabelFrame(self.control_frame, text="亮度和对比度")
        self.bc_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(self.bc_frame, text="亮度").pack()
        self.brightness_var = tk.IntVar(value=0)
        self.brightness_slider = tk.Scale(self.bc_frame, from_=-100, to=100, orient=tk.HORIZONTAL, 
                                        variable=self.brightness_var, command=self.update_brightness_contrast)
        self.brightness_slider.pack(fill=tk.X)
        
        tk.Label(self.bc_frame, text="对比度").pack()
        self.contrast_var = tk.IntVar(value=0)
        self.contrast_slider = tk.Scale(self.bc_frame, from_=-100, to=100, orient=tk.HORIZONTAL, 
                                      variable=self.contrast_var, command=self.update_brightness_contrast)
        self.contrast_slider.pack(fill=tk.X)
        
        # 阈值处理
        tk.Button(self.process_frame, text="阈值处理", command=self.apply_threshold).pack(fill=tk.X, pady=2)
        self.threshold_var = tk.StringVar(value="binary")
        tk.OptionMenu(self.process_frame, self.threshold_var, "binary", "binary_inv", "trunc", "tozero", "tozero_inv").pack(fill=tk.X)
        
        # 形态学变换
        self.morph_frame = tk.Frame(self.process_frame)
        self.morph_frame.pack(fill=tk.X, pady=2)
        tk.Button(self.morph_frame, text="形态学变换", command=self.apply_morphology).pack(side=tk.LEFT)
        self.morph_op_var = tk.StringVar(value="erosion")
        tk.OptionMenu(self.morph_frame, self.morph_op_var, "erosion", "dilation", "opening", "closing").pack(side=tk.LEFT)
        self.morph_size_var = tk.StringVar(value="3")
        tk.OptionMenu(self.morph_frame, self.morph_size_var, "3", "5", "7").pack(side=tk.LEFT)
        
        # 翻转图像
        self.flip_frame = tk.Frame(self.process_frame)
        self.flip_frame.pack(fill=tk.X, pady=2)
        tk.Button(self.flip_frame, text="翻转图像", command=self.flip_image).pack(side=tk.LEFT)
        self.flip_var = tk.StringVar(value="horizontal")
        tk.OptionMenu(self.flip_frame, self.flip_var, "horizontal", "vertical").pack(side=tk.LEFT)
        
        # 图像分割
        tk.Button(self.process_frame, text="图像分割", command=lambda: self.apply_processing(ImageProcessor.image_segmentation)).pack(fill=tk.X, pady=2)
        
        # 图像滤波
        self.filter_frame = tk.LabelFrame(self.process_frame, text="图像滤波")
        self.filter_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(self.filter_frame, text="低频阈值").pack()
        self.low_threshold_var = tk.IntVar(value=0)
        tk.Scale(self.filter_frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.low_threshold_var).pack(fill=tk.X)
        
        tk.Label(self.filter_frame, text="高频阈值").pack()
        self.high_threshold_var = tk.IntVar(value=255)
        tk.Scale(self.filter_frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.high_threshold_var).pack(fill=tk.X)
        
        tk.Button(self.filter_frame, text="应用滤波", command=lambda: self.apply_processing(
            ImageProcessor.image_filter, self.low_threshold_var.get(), self.high_threshold_var.get())).pack(fill=tk.X)
        
        # 撤销/重做按钮
        self.undo_frame = tk.Frame(self.control_frame)
        self.undo_frame.pack(fill=tk.X, pady=5)
        tk.Button(self.undo_frame, text="撤销", command=self.undo).pack(side=tk.LEFT, padx=5)
        tk.Button(self.undo_frame, text="重做", command=self.redo).pack(side=tk.LEFT, padx=5)
        
        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)
    
    def open_image(self):
        """打开图像文件"""
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            try:
                image = Image.open(file_path)
                self.original_image = np.array(image)
                self.processed_image = self.original_image.copy()
                self.history_manager.push_state(self.processed_image)
                self.update_display()
                self.status_var.set(f"已加载: {file_path}")
            except Exception as e:
                self.status_var.set(f"错误: {str(e)}")
    
    def save_image(self):
        """保存处理后的图像"""
        if self.processed_image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", 
                                                    filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")])
            if file_path:
                try:
                    Image.fromarray(self.processed_image).save(file_path)
                    self.status_var.set(f"已保存: {file_path}")
                except Exception as e:
                    self.status_var.set(f"错误: {str(e)}")
    
    def update_display(self):
        """更新图像显示"""
        if self.original_image is not None:
            img = Image.fromarray(self.original_image)
            img = img.resize((400, 400), Image.LANCZOS)
            self.original_photo = ImageTk.PhotoImage(img)
            self.original_label.config(image=self.original_photo)
        
        if self.processed_image is not None:
            img = Image.fromarray(self.processed_image)
            img = img.resize((400, 400), Image.LANCZOS)
            self.processed_photo = ImageTk.PhotoImage(img)
            self.processed_label.config(image=self.processed_photo)
    
    def apply_processing(self, func, *args):
        """应用图像处理函数"""
        if self.processed_image is not None:
            self.processed_image = func(self.processed_image, *args)
            self.history_manager.push_state(self.processed_image)
            self.update_display()
    
    def rotate_image(self):
        """旋转图像"""
        if self.processed_image is not None:
            angle = int(self.rotate_var.get())
            self.apply_processing(ImageProcessor.rotate_image, angle)
