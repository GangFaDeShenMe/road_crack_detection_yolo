from ultralytics import YOLO

# Build a YOLOv9c model from pretrained weight
model = YOLO("best.pt")

# Display model information (optional)
model.info()

# Perform object detection on an image
results0 = model("image0.jpg")
results1 = model("image1.jpg")
results2 = model("image2.jpeg")
results3 = model("image3.jpeg")

# Display the results
#results0[0].show()
#results1[0].show()
#results2[0].show()
results3[0].show()
