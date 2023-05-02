from keras.models import model_from_json
import operator
import cv2


# Loading the model
json_file = open("model1.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
# load weights into new model
loaded_model.load_weights("model1.h5")
print("Loaded model from disk")

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    # Flip the image horizontally for a selfie-view display.
    frame = cv2.flip(frame, 1)

    # Coordinates of the ROI on frame
    x1 = 400
    y1 = 100
    x2 = 620
    y2 = 320
    # Drawing the ROI on frame. The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0) ,2)


    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, (128, 128)) 
    
    
    ## Convert to grayscale
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    roi = cv2.GaussianBlur(roi,(7,7),0)
    
    #roi = cv2.adaptiveThreshold(roi,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,3)
    _,roi = cv2.threshold(roi, 10, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    # Converting the image to 3 channels
    roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)

   
    cv2.imshow("Prediction", roi)
   
    result = loaded_model.predict(roi.reshape(1,128,128,3))
    prediction = {'Zero': result[0][0], 
                  'One': result[0][1], 
                  'Two': result[0][2],
                  'Three': result[0][3],
                  'Four': result[0][4],
                  'Five': result[0][5],
                  'A': result[0][6], 
                  'B': result[0][7], 
                  'C': result[0][8],
                  'D': result[0][9],
                  'E': result[0][10],
                  'F': result[0][11],
                  'G': result[0][12],
                  'H': result[0][13],
                  'I': result[0][14],
                  'J': result[0][15],
                  'Null': result[0][16]
                  }
    # Sorting based on top prediction
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    
    # Displaying the predictions
    cv2.putText(frame, prediction[0][0], (10, 120), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 3)    
    cv2.imshow("Frame: Prediction", frame)
    
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break
        
 
cap.release()
cv2.destroyAllWindows()