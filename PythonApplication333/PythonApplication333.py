import cv2
import numpy as np
import streamlit as st
from PIL import Image

# إعدادات الصفحة
st.set_page_config(
    page_title="نظام كشف المركبات والأشخاص",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗🚶‍♂️ نظام كشف المركبات والأشخاص")
st.write("مشروع رؤية حاسوبية باستخدام نموذج YOLOv4-tiny لكشف الأشخاص والمركبات في الصور.")

# القائمة الجانبية لإعدادات التصفية
st.sidebar.header("إعدادات التحكم")
confidence_threshold = st.sidebar.slider("عتبة الثقة (Confidence)", 0.1, 1.0, 0.3)
nms_threshold = st.sidebar.slider("عتبة NMS (تداخل المربعات)", 0.1, 1.0, 0.4)

# أسماء الفئات الست المستهدفة حسب ترتيب فئات COCO
TARGET_CLASSES = {
    0: "Person",
    1: "Bicycle",
    2: "Car",
    3: "Motorbike",
    5: "Bus",
    7: "Truck"
}

# تحميل نموذج YOLO وتخزينه في الذاكرة لتسريع الأداء
@st.cache_resource
def load_yolo():
    net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

# رفع الصورة من القائمة الجانبية
uploaded_file = st.sidebar.file_uploader("اختر صورة لرفعها...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. قراءة وتحويل الصورة إلى RGB لضمان عدم حدوث خطأ الأبعاد
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    height, width, _ = img_array.shape

    # 2. تحميل النموذج وتجهيز الصورة
    net, output_layers = load_yolo()
    blob = cv2.dnn.blobFromImage(
        img_array, 1 / 255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False
    )
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    # 3. معالجة مخرجات الكشف
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # تصفية الفئات لإبقاء الفئات الست المستهدفة فقط وتجاوز عتبة الثقة
            if class_id in TARGET_CLASSES and confidence > confidence_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # 4. تطبيق Non-Max Suppression لإزالة المربعات المكررة
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    # 5. رسم المربعات والتسميات
    result_img = img_array.copy()
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(TARGET_CLASSES[class_ids[i]])
            confidence_score = confidences[i]

            # رسم المربع
            color = (0, 255, 0)
            cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)
            
            # كتابة اسم الفئة مع نسبة الثقة
            text = f"{label} {confidence_score:.2f}"
            cv2.putText(
                result_img,
                text,
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    # عرض النتيجة
    st.subheader("النتيجة:")
    st.image(result_img, caption="الصورة بعد الكشف", use_column_width=True)
else:
    st.info("👈 يرجى رفع صورة من القائمة الجانبية لبدء عملية الكشف.")
