"""
极简封口质检演示工具（手动模式，支持模型切换 + 阈值调节）
依赖: tkinter, opencv-python, pillow, ultralytics
"""

import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO

# ========== 全局配置 ==========
MODEL_OPTIONS = {
    "YOLOv8 (yolov8_i.pt)": "models/yolo/weights/yolov8_3.pt",
    "YOLOv11 (yolov11_i.pt)": "models/yolo/weights/yolov11_1.pt",
}
DEFAULT_MODEL_KEY = "YOLOv8 (yolov8_i.pt)"
DEFAULT_CONF = 0.25
IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

# 默认阈值
DEFAULT_SILVER_THRESH = 0.3
DEFAULT_FOREIGN_THRESH = 0.6


class InspectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("封口质检工具 - 银边+异物检测")
        self.root.geometry("800x650")  # 增加高度容纳阈值控件

        # 状态变量
        self.image_list = []
        self.current_idx = -1
        self.model = None
        self.current_annotated = None
        self.current_model_path = None

        # 阈值变量
        self.silver_thresh = tk.DoubleVar(value=DEFAULT_SILVER_THRESH)
        self.foreign_thresh = tk.DoubleVar(value=DEFAULT_FOREIGN_THRESH)

        # 界面组件
        self.create_widgets()

        # 加载默认模型
        self.load_model_async(DEFAULT_MODEL_KEY)

    def create_widgets(self):
        """创建界面布局"""
        # 0. 模型选择区域（带浏览按钮）
        model_frame = tk.Frame(self.root)
        model_frame.pack(pady=5)

        tk.Label(model_frame, text="选择模型:").pack(side=tk.LEFT, padx=5)
        self.model_var = tk.StringVar(value=DEFAULT_MODEL_KEY)
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var,
                                        values=list(MODEL_OPTIONS.keys()), state="readonly", width=20)
        self.model_combo.pack(side=tk.LEFT, padx=5)
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_changed)

        self.btn_browse = tk.Button(model_frame, text="浏览...", command=self.select_custom_weight)
        self.btn_browse.pack(side=tk.LEFT, padx=5)

        # 0.5 置信度阈值调节区域
        thresh_frame = tk.Frame(self.root)
        thresh_frame.pack(pady=5, fill=tk.X, padx=10)

        # 银边阈值
        tk.Label(thresh_frame, text="银边置信度阈值:").pack(side=tk.LEFT, padx=5)
        self.silver_slider = tk.Scale(thresh_frame, from_=0.0, to=1.0, resolution=0.01,
                                      orient=tk.HORIZONTAL, variable=self.silver_thresh,
                                      length=150, command=self.on_thresh_changed)
        self.silver_slider.pack(side=tk.LEFT, padx=5)
        self.silver_label = tk.Label(thresh_frame, text=f"{self.silver_thresh.get():.2f}", width=5)
        self.silver_label.pack(side=tk.LEFT, padx=2)

        # 异物阈值
        tk.Label(thresh_frame, text="    异物置信度阈值:").pack(side=tk.LEFT, padx=5)
        self.foreign_slider = tk.Scale(thresh_frame, from_=0.0, to=1.0, resolution=0.01,
                                       orient=tk.HORIZONTAL, variable=self.foreign_thresh,
                                       length=150, command=self.on_thresh_changed)
        self.foreign_slider.pack(side=tk.LEFT, padx=5)
        self.foreign_label = tk.Label(thresh_frame, text=f"{self.foreign_thresh.get():.2f}", width=5)
        self.foreign_label.pack(side=tk.LEFT, padx=2)

        # 1. 控制按钮区域
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        self.btn_open = tk.Button(btn_frame, text="选择文件夹", command=self.select_folder, width=12)
        self.btn_open.pack(side=tk.LEFT, padx=5)

        self.btn_prev = tk.Button(btn_frame, text="上一张", command=self.prev_image, state=tk.DISABLED, width=8)
        self.btn_prev.pack(side=tk.LEFT, padx=5)

        self.btn_next = tk.Button(btn_frame, text="下一张", command=self.next_image, state=tk.DISABLED, width=8)
        self.btn_next.pack(side=tk.LEFT, padx=5)

        self.btn_detect = tk.Button(btn_frame, text="检测当前图片", command=self.detect_current, state=tk.DISABLED, width=12)
        self.btn_detect.pack(side=tk.LEFT, padx=5)

        self.btn_quit = tk.Button(btn_frame, text="退出", command=self.root.quit, width=8)
        self.btn_quit.pack(side=tk.LEFT, padx=5)

        # 2. 图像显示区域
        img_frame = tk.Frame(self.root, bd=2, relief=tk.SUNKEN)
        img_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.canvas = tk.Canvas(img_frame, bg='gray')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 3. 日志区域
        log_frame = tk.Frame(self.root)
        log_frame.pack(fill=tk.X, padx=10, pady=5)

        self.log_text = tk.Text(log_frame, height=8, state=tk.DISABLED, wrap=tk.WORD)
        scrollbar = tk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 4. 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪。请选择图片文件夹")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def on_thresh_changed(self, event=None):
        """阈值滑动条回调，更新显示标签"""
        self.silver_label.config(text=f"{self.silver_thresh.get():.2f}")
        self.foreign_label.config(text=f"{self.foreign_thresh.get():.2f}")

    def select_custom_weight(self):
        """选择自定义权重文件"""
        file_path = filedialog.askopenfilename(
            title="选择YOLO权重文件 (.pt)",
            filetypes=[("PyTorch模型", "*.pt"), ("所有文件", "*.*")]
        )
        if not file_path:
            return
        self.load_model_by_path(file_path)

    def load_model_by_path(self, model_path):
        """直接加载指定路径的模型"""
        def load():
            if self.current_model_path == model_path:
                self.log(f"模型已在使用中: {model_path}")
                return
            self.log(f"正在加载自定义模型: {model_path} ...")
            self.root.config(cursor="watch")
            try:
                new_model = YOLO(model_path)
                self.model = new_model
                self.current_model_path = model_path
                self.log(f"模型加载成功: {model_path}")
                self.status_var.set(f"已加载自定义模型: {os.path.basename(model_path)}")
                if self.image_list:
                    self.btn_detect.config(state=tk.NORMAL)
            except Exception as e:
                self.log(f"模型加载失败: {e}")
                messagebox.showerror("错误", f"模型加载失败\n请检查路径: {model_path}")
            finally:
                self.root.config(cursor="")
        threading.Thread(target=load, daemon=True).start()

    def on_model_changed(self, event=None):
        key = self.model_var.get()
        self.load_model_async(key)

    def load_model_async(self, model_key):
        def load():
            model_path = MODEL_OPTIONS[model_key]
            if self.current_model_path == model_path:
                return
            self.log(f"正在加载模型: {model_key} ...")
            self.root.config(cursor="watch")
            try:
                new_model = YOLO(model_path)
                self.model = new_model
                self.current_model_path = model_path
                self.log(f"模型加载成功: {model_key}")
                self.status_var.set(f"已加载 {model_key}")
                if self.image_list:
                    self.btn_detect.config(state=tk.NORMAL)
            except Exception as e:
                self.log(f"模型加载失败: {e}")
                messagebox.showerror("错误", f"模型加载失败\n请检查路径: {model_path}")
            finally:
                self.root.config(cursor="")
        threading.Thread(target=load, daemon=True).start()

    def select_folder(self):
        folder = filedialog.askdirectory(title="选择包含图片的文件夹")
        if not folder:
            return
        folder_path = Path(folder)
        self.image_list = sorted([str(p) for p in folder_path.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES])
        if not self.image_list:
            messagebox.showwarning("警告", "所选文件夹中没有支持的图片文件")
            return
        self.current_idx = 0
        self.current_annotated = None
        self.btn_prev.config(state=tk.NORMAL if len(self.image_list) > 1 else tk.DISABLED)
        self.btn_next.config(state=tk.NORMAL if len(self.image_list) > 1 else tk.DISABLED)
        self.btn_detect.config(state=tk.NORMAL if self.model else tk.DISABLED)
        self.status_var.set(f"已加载 {len(self.image_list)} 张图片")
        self.log(f"打开文件夹: {folder}，共 {len(self.image_list)} 张图片")
        self.show_current_image()

    def show_current_image(self, annotated_img_bgr=None):
        if not self.image_list or self.current_idx < 0 or self.current_idx >= len(self.image_list):
            return
        if annotated_img_bgr is None:
            img_original = cv2.imread(self.image_list[self.current_idx])
            if img_original is None:
                self.log(f"警告：无法读取图片 {self.image_list[self.current_idx]}")
                return
            cropped = img_original[500:1700, 200:2400]
            processed = cv2.resize(cropped, (1184, 640), interpolation=cv2.INTER_LINEAR)
            img_bgr = processed
        else:
            img_bgr = annotated_img_bgr
            self.current_annotated = annotated_img_bgr

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        canvas_width = self.canvas.winfo_width() if self.canvas.winfo_width() > 10 else 600
        canvas_height = self.canvas.winfo_height() if self.canvas.winfo_height() > 10 else 400
        img_pil.thumbnail((canvas_width, canvas_height), Image.LANCZOS)

        self.tk_img = ImageTk.PhotoImage(img_pil)
        self.canvas.delete("all")
        x = (canvas_width - self.tk_img.width()) // 2
        y = (canvas_height - self.tk_img.height()) // 2
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.tk_img)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def prev_image(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.current_annotated = None
            self.show_current_image()
            self.update_status()

    def next_image(self):
        if self.current_idx < len(self.image_list) - 1:
            self.current_idx += 1
            self.current_annotated = None
            self.show_current_image()
            self.update_status()

    def update_status(self):
        if self.image_list:
            self.status_var.set(f"当前图片: {self.current_idx+1}/{len(self.image_list)}")
        else:
            self.status_var.set("无图片")

    def detect_current(self):
        if not self.image_list:
            messagebox.showwarning("警告", "请先选择文件夹")
            return
        if self.model is None:
            messagebox.showwarning("警告", "模型尚未加载完成，请稍候")
            return

        img_path = self.image_list[self.current_idx]
        self.status_var.set(f"正在检测: {os.path.basename(img_path)}")
        self.root.update()

        try:
            annotated_img, desc = self.detect_and_show(img_path)
            self.show_current_image(annotated_img)
            self.log(f"[{self.current_idx+1}/{len(self.image_list)}] {os.path.basename(img_path)} : {desc}")
        except Exception as e:
            self.log(f"检测异常: {e}")
        finally:
            self.update_status()

    def detect_and_show(self, img_path):
        try:
            img_original = cv2.imread(img_path)
            if img_original is None:
                return None, "无法读取图片"
            cropped = img_original[500:1700, 200:2400]
            processed = cv2.resize(cropped, (1184, 640), interpolation=cv2.INTER_LINEAR)

            # 使用当前界面设置的阈值
            silver_th = self.silver_thresh.get()
            foreign_th = self.foreign_thresh.get()

            results = self.model.predict(processed, conf=0.4, iou=0.3, verbose=False)
            result = results[0]
            boxes = result.boxes
            annotated = processed.copy()

            if boxes is not None and len(boxes) > 0:
                cls_ids = boxes.cls.tolist()
                confs = boxes.conf.tolist()
                names = result.names

                keep_indices = []
                for i, (cls_id, conf) in enumerate(zip(cls_ids, confs)):
                    cls_id_int = int(cls_id)
                    if cls_id_int == 0 and conf >= silver_th:      # 银边
                        keep_indices.append(i)
                    elif cls_id_int == 1 and conf >= foreign_th:   # 异物
                        keep_indices.append(i)

                if keep_indices:
                    detected_classes = []
                    confs_kept = []
                    for i in keep_indices:
                        box = boxes.xyxy[i].tolist()
                        x1, y1, x2, y2 = map(int, box)
                        cls_id = int(cls_ids[i])
                        conf = confs[i]
                        class_name = names[cls_id]
                        detected_classes.append(class_name)
                        confs_kept.append(conf)
                        color = (0, 255, 0) if class_name == 'silver_edge' else (0, 0, 255)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        label = f"{class_name} {conf:.2f}"
                        cv2.putText(annotated, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    unique = set(detected_classes)
                    if 'silver_edge' in unique and 'foreign_object' in unique:
                        desc = "检测到银边和异物"
                    elif 'silver_edge' in unique:
                        desc = "检测到银边"
                    elif 'foreign_object' in unique:
                        desc = "检测到异物"
                    else:
                        desc = "无缺陷"
                    if confs_kept:
                        desc += f" (最高置信度: {max(confs_kept):.2f})"
                else:
                    desc = "无缺陷"
            else:
                desc = "无缺陷"

            return annotated, desc

        except Exception as e:
            self.log(f"检测出错 {img_path}: {e}")
            try:
                img_original = cv2.imread(img_path)
                cropped = img_original[500:1700, 200:2400]
                processed = cv2.resize(cropped, (1184, 640))
                return processed, "检测失败"
            except:
                return None, "检测失败"

    def log(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    app = InspectorApp(root)
    root.mainloop()