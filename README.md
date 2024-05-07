# drowsy_detection
drowsiness detection with ESP32-S3 T-SIMCAM 
1. Setting up a camera from ESP32-S3 T-SIMCAM to monitor a stream for faces. Capture every 1 second.
2. Apply facial landmark localization to extract the eye regions from the face.
The facial landmark detector produces 68 (x, y)-coordinates that map to specific facial structures.
The 68 point mappings were obtained by training a shape predictor on the labeled iBUG 300-W dataset.
3. Compute the eye aspect ratio (EAR) to determine if the eyes are closed.
3.1 Each eye is represented by 6 (x, y) -coordinates, starting at the left-corner of the eye, and then working clockwise around the remainder of the region.
![Screenshot (982)](https://github.com/pitijit/drowsy_detection/assets/85090124/928101a5-57f6-4b2e-8088-aab85e11795e)
3.2 Using the eye aspect ratio (EAR) equation that reflects the aforementioned relation, we can avoid image processing techniques and simply rely on the ratio of eye landmark distances to determine if a person is blinking.
The eye aspect ratio is constant, then rapidly drops close to zero, then increases again, indicating a single blink has taken place.
![Screenshot (983)](https://github.com/pitijit/drowsy_detection/assets/85090124/111de501-95f2-47e1-a9d0-b854736a7fef)


4. If the eye aspect ratio indicates that the eyes have been closed for a sufficiently long enough amount of time, we’ll sound an alarm to wake up the driver.
