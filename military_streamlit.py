import streamlit as st
from PIL import Image
from ultralytics import YOLO
import tempfile
import cv2
import pandas as pd


# Load the YOLO model
model = YOLO("C:/Users/Appu/Desktop/data science/python/military_detection/best.pt")

# Set page config
st.set_page_config(page_title="Military Object Detection", layout="wide")
st.title("🎖️ Military Object Detection using YOLO")
st.markdown("Detect military tanks, trucks, weapons, and soldiers from images or videos.")

# Sidebar for choosing mode
mode = st.sidebar.radio("Choose Detection Mode", ["Image", "Video", "Model comparison","Project Flowchart"])

if mode == "Image":
    uploaded_img = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_img:
        img = Image.open(uploaded_img)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button("Run Detection"):
            st.info("Detecting objects...")
            results = model.predict(img, imgsz=640, conf=0.25)
            result_img = results[0].plot()

            # Display result
            st.image(result_img, caption="Detected Objects", use_column_width=True)

            # Save and download
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)).save(tmp.name)
                st.download_button("📥 Download Result", data=open(tmp.name, "rb"), file_name="detection_result.jpg")
                tmp.close()

            st.session_state["results"] = results

elif mode == "Video":
    uploaded_vid = st.file_uploader("Choose a video", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_vid:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_vid.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        st.info("⏳ Processing video...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, imgsz=640, conf=0.25)
            annotated_frame = results[0].plot()
            stframe.image(annotated_frame, channels="BGR", use_column_width=True)

        cap.release()
        st.success("✅ Video processing complete.")
        st.session_state["results"] = results

elif mode == "Model comparison":
    data = {
        "Run Name": [
            "military_detection_finetune",
            "military_detection_full_finetune",
            "military_detection_continue",
            "military_detection4"
        ],
        "Epochs": [20, 15, 30, 30],
        "mAP50": [0.91510, 0.89977, 0.83542, 0.81523],
        "mAP50-95": [0.68654, 0.65999, 0.58832, 0.56661]
    }

    compare_df = pd.DataFrame(data)

    st.subheader("📊 Model Comparison Summary")
    st.dataframe(compare_df.sort_values("mAP50-95", ascending=False), use_container_width=True)
 
elif mode == "Project Flowchart":
    st.header("Project Flowchart")
    # Load and display the image
    image_path = "C:\\Users\\Appu\\Desktop\\data science\\python\\military_detection\\flowchart.png"
    image = Image.open(image_path)
    st.image(image, caption='project flowchart', use_column_width=True)
    

# Confidence and label summary
if st.sidebar.checkbox("🔢 Show Confidence Scores and Class Counts"):
    st.subheader("📊 Detection Summary")

    if "results" in st.session_state:
        results = st.session_state["results"]
        classes = results[0].boxes.cls.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        names = results[0].names

        class_counts = {names[int(c)]: 0 for c in classes}
        for c in classes:
            class_counts[names[int(c)]] += 1

        st.write("**Class Counts:**")
        st.json(class_counts)

        st.write("**Confidence Scores:**")
        for i, score in enumerate(scores):
            st.write(f"{names[int(classes[i])]}: {score:.2f}")
    else:
        st.warning("⚠️ Run detection first to see class scores and confidence.")
