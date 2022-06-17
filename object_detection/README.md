## Object Detection with Torchvision

This code snippet accompanied my presentation at the Snowflake Summit '22, 
where I briefly mentioned a possible use case for detecting foot and car 
traffic in a parking garage.

Data scientists can use object detection models to turn unstructured 
data (images, text, audio) into either a) structured embeddings that can 
be used as features across different downstreams, or b) information by 
extracting meaning from the unstructured input.

### Running the example
```
python obj_detection_example.py your_image_url 
```
This example prints the detected objects in an image and the model's 
confidence in each prediction. Thresholding on the confidence helps filter 
out noisy predictions that are likely to be incorrect. 
