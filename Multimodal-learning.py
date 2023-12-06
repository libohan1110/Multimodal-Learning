
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as K
import cv2
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import messagebox
import threading
import lifelines
def browse_path1():
    global path1
    path1 = filedialog.askopenfilename()
    print("Path to image 1:", path1)

def browse_path2():
    global path2
    path2 = filedialog.askopenfilename()
    print("Path to image 2:", path2)
def analyze_images():
    global path1, path2

    if path1 is None or path2 is None:
        messagebox.showerror("Error", "Please select both images.")
        return
    else:
        progress_bar.start()
        for i in range(101):
            progress_bar["value"] = i
            progress_bar.update()

        t = threading.Thread(target=analyze_images_thread)
        t.start()

def analyze_images():
    clinicaldata = []
    for entry in entry_objects:
        data = entry.get().strip()  # 去除前后空格
        if data:
            try:
                data = float(data)
                clinicaldata.append(data)
            except ValueError:
                print(f"Invalid input: {data}")

    clinicaldata_array = np.array(clinicaldata)

    # 导入图像
    img_1 = image.load_img(path1, target_size=(336, 336))
    img_1 = np.array(img_1)
    img_2 = image.load_img(path2, target_size=(336, 336))
    img_2 = np.array(img_2)
    img = np.hstack((img_1, img_2))
    img = img.reshape(1, 336, 672, 3)
    train_x_ = img.astype("float32") * 1 / 255
    train_x_ = train_x_[:, ::2, ::2]

    # 载入模型
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    model = keras.models.load_model('mobileV3-1.h5')
    input = train_x_
    last_conv_layer = model.get_layer('dense')
    iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
    model_out, last_conv_layer = iterate(input)
    outcomes = np.array(last_conv_layer)
    combined_array = np.concatenate([clinicaldata_array, outcomes.flatten()])
    combined = pd.DataFrame(data=combined_array.reshape(1, -1))

    conv_base = keras.models.load_model('mobileV3conv_base.h5')
    with tf.GradientTape() as tape:
        last_conv_layer = conv_base.get_layer('multiply_11')
        iterate = tf.keras.models.Model([conv_base.inputs], [conv_base.output, last_conv_layer.output])
        model_out, last_conv_layer = iterate(input)
        grads = tape.gradient(model_out, last_conv_layer)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = 3 * heatmap
    heatmap[heatmap > 1] = 1
    heatmap = heatmap.reshape((11, 21))

    INTENSITY = 0.5

    raw = train_x_ * 255
    # raw = raw.resize(168, 336)
    heatmap = cv2.resize(heatmap, (336, 168))
    if model.predict(input) > 0.5:
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_WINTER)
    else:
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    img_combine = ((heatmap + np.array(raw)) * INTENSITY) / 255

    # 比例风险
    X_train = np.load('traintotal.npy')
    Y_train = np.load('T_train.npy')
    E_train = np.load('E_train.npy')
    X_train = pd.DataFrame(X_train)
    Y_train = pd.DataFrame(Y_train)
    E_train = pd.DataFrame(E_train)

    cols_standardize = X_train.columns
    X_ct = ColumnTransformer([('standardizer', StandardScaler(), cols_standardize)])
    X_ct.fit(X_train[cols_standardize])

    X_train[cols_standardize] = X_ct.transform(X_train[cols_standardize])
    combined[cols_standardize] = X_ct.transform(combined[cols_standardize])

    xgb_model = xgb.XGBRegressor()
    booster = xgb.Booster()
    booster.load_model('model.xgb')
    xgb_model._Booster = booster

    Y_pred_train2 = np.log(xgb_model.predict(X_train))
    time = np.array(Y_train.iloc[:, 0])
    event = np.array(E_train.iloc[:, 0])
    # 创建 Cox 比例风险模型并拟合数据
    cox_model = lifelines.CoxPHFitter()
    cox_model.fit(pd.DataFrame({
        'time': time,
        'event': event,
        'Y_pred_train': Y_pred_train2
    }), duration_col='time', event_col='event', show_progress=True)
    # 预测当时间为6，Y_pred_test=0.8 时发生事件的概率
    t2 = 24
    Y_pred_test = np.log(xgb_model.predict(combined))  # 假设 Y_pred_test 为 0.8
    surv_func = cox_model.predict_survival_function(Y_pred_test)
    prob_event2 = 1 - surv_func.iloc[t2][0]
    Y_pred_test = round(prob_event2, 3)

    img_combine = img_combine.reshape(168, 336, 3)

    words2 = "Fertility predicted score: " + str(Y_pred_test)
    words3 = "*>0.76:Low-risk; 0.13-0.76: Mid-risk; <0.13: High-risk"
    img_combine = cv2.resize(img_combine, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    if Y_pred_test >0.76:
        words = "Low risk: Hopeful for natural conception"
        cv2.putText(img_combine, words, (10, 250), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
    elif Y_pred_test >0.13:
        words = "Mid-risk: Recommended ART intervention"
        cv2.putText(img_combine, words, (10, 250), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2, cv2.LINE_AA)
    else:
        words = "High risk: Re-adhesiolysis & ART"
        cv2.putText(img_combine, words, (10, 250), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.putText(img_combine, words2, (10, 280), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img_combine, words3, (10, 300), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("predict result", img_combine)
    progress_bar.stop()
    cv2.waitKey()




# 创建一个主窗口
root = tk.Tk()
root.title("Multimodal learning application")
root.geometry("1000x800")

# 创建一个框架用于放置EMR信息的标签和文本框
emr_frame = tk.Frame(root)
emr_frame.pack(pady=20)

# 创建一个标签
emr_label = tk.Label(emr_frame, text="Please select the EMR information to analyze:", font=("Helvetica", 20))
emr_label.pack()

# 创建一个标签和文本框的框架
input_frame = tk.Frame(emr_frame)
input_frame.pack(pady=20)

# 添加所有的标签和文本框到输入框架中
entries_labels = [
    ("BMI", "Age"),
    ("Symptom duration", "Uterine volume"),
    ("Endometrial thickness", "CSGE"),
    ("Gravidity", "Artificial abortion"),
    ("Age at menarche", "Uterine cavity depth"),
    ("Fallopian tube orifice (Postoperation)", "Reduction of flow"),
    ("Blood supply", "Increase in flow"),
    ("Missed abortion", "Uterine cavity shape")
]

entry_objects = []  # 创建一个列表来保存所有的Entry对象

for row in range(8):
    for col in range(2):
        label_text = entries_labels[row][col]
        label = tk.Label(input_frame, text=label_text, font=("Helvetica", 12))
        label.grid(row=row, column=2*col, padx=5, pady=5, sticky="e")

        entry = tk.Entry(input_frame)
        entry.grid(row=row, column=2*col+1, padx=5, pady=5, sticky="w")

        entry_objects.append(entry)  # 将Entry对象添加到列表中

# 创建一个框架用于放置图片选择的标签和按钮
image_frame = tk.Frame(root)
image_frame.pack(pady=20)

# 创建一个标签
image_label = tk.Label(image_frame, text="Please select the images to analyze:", font=("Helvetica", 20))
image_label.pack()

# 创建一个框架用于放置按钮
button_frame = tk.Frame(image_frame)
button_frame.pack()

# 创建两个按钮，用于选择图片
button1 = tk.Button(button_frame, text="Enter the image of the uterine cavity", font=("Helvetica", 20, "bold"), command=browse_path1)
button1.pack(pady=10)

button2 = tk.Button(button_frame, text="Enter the image of the uterine corner", font=("Helvetica", 20, "bold"), command=browse_path2)
button2.pack(pady=10)

# 创建一个按钮，用于分析所选的两张图片
analyze_button = tk.Button(button_frame, text="Analyze", font=("Helvetica", 20, "bold"), command=analyze_images)
analyze_button.pack(pady=20)

progress_bar = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=400, mode='determinate')
progress_bar.pack(pady=20)

# 运行主循环
root.mainloop()









