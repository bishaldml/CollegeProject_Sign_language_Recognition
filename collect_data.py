import cv2
import numpy as np
import os
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

if not os.path.exists("data"):
    os.makedirs("data")
    # train folder
    os.makedirs("data/train")
    os.makedirs("data/train/A")
    os.makedirs("data/train/B")
    os.makedirs("data/train/C")
    os.makedirs("data/train/D")
    os.makedirs("data/train/E")
    os.makedirs("data/train/F")
    os.makedirs("data/train/G")
    os.makedirs("data/train/H")
    os.makedirs("data/train/I")
    os.makedirs("data/train/J")
    os.makedirs("data/train/K")
    os.makedirs("data/train/L")
    os.makedirs("data/train/M")
    os.makedirs("data/train/N")
    os.makedirs("data/train/O")
    os.makedirs("data/train/P")
    os.makedirs("data/train/Q")
    os.makedirs("data/train/R")
    os.makedirs("data/train/S")
    os.makedirs("data/train/T")
    os.makedirs("data/train/U")
    os.makedirs("data/train/V")
    os.makedirs("data/train/W")
    os.makedirs("data/train/X")
    os.makedirs("data/train/Y")
    os.makedirs("data/train/Z")
    # test folder
    os.makedirs("data/test")
    os.makedirs("data/test/A")
    os.makedirs("data/test/B")
    os.makedirs("data/test/C")
    os.makedirs("data/test/D")
    os.makedirs("data/test/E")
    os.makedirs("data/test/F")
    os.makedirs("data/test/G")
    os.makedirs("data/test/H")
    os.makedirs("data/test/I")
    os.makedirs("data/test/J")
    os.makedirs("data/test/K")
    os.makedirs("data/test/L")
    os.makedirs("data/test/M")
    os.makedirs("data/test/N")
    os.makedirs("data/test/O")
    os.makedirs("data/test/P")
    os.makedirs("data/test/Q")
    os.makedirs("data/test/R")
    os.makedirs("data/test/S")
    os.makedirs("data/test/T")
    os.makedirs("data/test/U")
    os.makedirs("data/test/V")
    os.makedirs("data/test/W")
    os.makedirs("data/test/X")
    os.makedirs("data/test/Y")
    os.makedirs("data/test/Z")


mode = 'train'
directory = 'data/'+mode+'/'

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while True:
        _, frame = cap.read()
        # Flip the image horizontally for a selfie-view display.
        frame = cv2.flip(frame, 1)
    
        # Getting count of existing images
        count = {'A': len(os.listdir(directory+"/A")),
             'B': len(os.listdir(directory+"/B")),
             'C': len(os.listdir(directory+"/C")),
             'D': len(os.listdir(directory+"/D")),
             'E': len(os.listdir(directory+"/E")),
             'F': len(os.listdir(directory+"/F")),
             'G': len(os.listdir(directory+"/G")),
             'H': len(os.listdir(directory+"/H")),
             'I': len(os.listdir(directory+"/I")),
             'J': len(os.listdir(directory+"/J")),
             'K': len(os.listdir(directory+"/K")),
             'L': len(os.listdir(directory+"/L")),
             'M': len(os.listdir(directory+"/M")),
             'N': len(os.listdir(directory+"/N")),
             'O': len(os.listdir(directory+"/O")),
             'P': len(os.listdir(directory+"/P")),
             'Q': len(os.listdir(directory+"/Q")),
             'R': len(os.listdir(directory+"/R")),
             'S': len(os.listdir(directory+"/S")),
             'T': len(os.listdir(directory+"/T")),
             'U': len(os.listdir(directory+"/U")),
             'V': len(os.listdir(directory+"/V")),
             'W': len(os.listdir(directory+"/W")),
             'X': len(os.listdir(directory+"/X")),
             'Y': len(os.listdir(directory+"/Y")),
             'Z': len(os.listdir(directory+"/Z")),
            }
    
        # Printing the count in each set to the screen/frame
        cv2.putText(frame, "MODE : "+mode, (10, 10), cv2.FONT_HERSHEY_PLAIN, 0.9, (0,0,255), 1)
        cv2.putText(frame, "IMAGE COUNT", (10, 20), cv2.FONT_HERSHEY_PLAIN, 0.9, (0,0,255), 1)
        cv2.putText(frame, "A : "+str(count['A']), (10, 30), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,255,0), 1)
        cv2.putText(frame, "B : "+str(count['B']), (10, 45), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,255,0), 1)
        cv2.putText(frame, "C : "+str(count['C']), (10, 60), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,255,0), 1)
        cv2.putText(frame, "D : "+str(count['D']), (10, 75), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,255,0), 1)
        cv2.putText(frame, "E : "+str(count['E']), (10, 90), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,255,0), 1)
        cv2.putText(frame, "F : "+str(count['F']), (10, 105), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,255,0), 1)
        cv2.putText(frame, "G : "+str(count['G']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,255,0), 1)    
        cv2.putText(frame, "H : "+str(count['H']), (10, 135), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,255,0), 1)    
        cv2.putText(frame, "I : "+str(count['I']), (10, 150), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,255,0), 1)    
        cv2.putText(frame, "J : "+str(count['J']), (10, 165), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,255,0), 1)    
        cv2.putText(frame, "K : "+str(count['K']), (10, 180), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,255,0), 1)    
        cv2.putText(frame, "L : "+str(count['L']), (10, 195), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,255,0), 1)    
        cv2.putText(frame, "M : "+str(count['M']), (10, 210), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,255,0), 1)    
        cv2.putText(frame, "N : "+str(count['N']), (10, 225), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,255,0), 1)
        cv2.putText(frame, "O : "+str(count['O']), (10, 240), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,255,0), 1)
        cv2.putText(frame, "P : "+str(count['P']), (10, 255), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,255,0), 1)
        cv2.putText(frame, "Q : "+str(count['Q']), (10, 270), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,255,0), 1)
        cv2.putText(frame, "R : "+str(count['R']), (10, 285), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,255,0), 1)
        cv2.putText(frame, "S : "+str(count['S']), (10, 300), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,255,0), 1)
        cv2.putText(frame, "T : "+str(count['T']), (10, 315), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,255,0), 1)    
        cv2.putText(frame, "U : "+str(count['U']), (10, 330), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,255,0), 1)    
        cv2.putText(frame, "V : "+str(count['V']), (10, 345), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,255,0), 1)    
        cv2.putText(frame, "W : "+str(count['W']), (10, 360), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,255,0), 1)    
        cv2.putText(frame, "X : "+str(count['X']), (10, 375), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,255,0), 1)    
        cv2.putText(frame, "Y : "+str(count['Y']), (10, 390), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,255,0), 1)    
        cv2.putText(frame, "Z : "+str(count['Z']), (10, 405), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,255,0), 1)

         # Coordinates of the ROI
        x1 = int(0.5*frame.shape[1])
        y1 = 10
        x2 = frame.shape[1]-10
        y2 = int(0.5*frame.shape[1])


        # Drawing the ROI
        # The increment/decrement by 1 is to compensate for the bounding box
        cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,2)
        # Extracting the ROI
        roi = frame[y1:y2, x1:x2]
        roi = cv2.resize(roi, (128, 128)) 
        cv2.imshow("Frame: Data Collection", frame)


        # Convert to grayscale
        #roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        #_, roi = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
        ## Resize to (128, 128) while preserving aspect ratio
        #h, w = roi.shape[:2]
        #if h > w:
        #    new_h, new_w = 128, int(w / h * 128)
        #else:
        #    new_h, new_w = int(h / w * 128), 128
        #roi = cv2.resize(roi, (new_w, new_h))
#   
        ## Pad the image to size (128, 128)
        #pad_w = (128 - new_w) // 2
        #pad_h = (128 - new_h) // 2
        #roi = cv2.copyMakeBorder(roi, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT)

        roi.flags.writeable = False
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    
        results = hands.process(roi)

        # Draw the hand annotations on the image.
        roi.flags.writeable = True
        roi = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
          for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                roi,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())


        
        # Convert to grayscale
        img_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Resize to (128, 128) while preserving aspect ratio
        h, w = img_gray.shape[:2]
        if h > w:
            new_h, new_w = 128, int(w / h * 128)
        else:
            new_h, new_w = int(h / w * 128), 128
        img_gray_resized = cv2.resize(img_gray, (new_w, new_h))

        # Pad the image to size (128, 128)
        pad_w = (128 - new_w) // 2
        pad_h = (128 - new_h) // 2
        img_gray_padded = cv2.copyMakeBorder(img_gray_resized, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT)

        # Add a new axis to create a single-channel image with shape (height, width, 1)
        roi = np.expand_dims(img_gray_padded, axis=-1)
        #_, roi = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
        cv2.imshow("ROI", roi)

        interrupt = cv2.waitKey(10)
        if interrupt & 0xFF == 27: # esc key
            break
        if interrupt & 0xFF == ord('a'):
            cv2.imwrite(directory+'A/'+str(count['A'])+'.jpg', roi)
        if interrupt & 0xFF == ord('b'):
            cv2.imwrite(directory+'B/'+str(count['B'])+'.jpg', roi)
        if interrupt & 0xFF == ord('c'):
            cv2.imwrite(directory+'C/'+str(count['C'])+'.jpg', roi)
        if interrupt & 0xFF == ord('d'):
            cv2.imwrite(directory+'D/'+str(count['D'])+'.jpg', roi)
        if interrupt & 0xFF == ord('e'):
            cv2.imwrite(directory+'E/'+str(count['E'])+'.jpg', roi)
        if interrupt & 0xFF == ord('f'):
            cv2.imwrite(directory+'F/'+str(count['F'])+'.jpg', roi)
        if interrupt & 0xFF == ord('g'):
            cv2.imwrite(directory+'G/'+str(count['G'])+'.jpg', roi)
        if interrupt & 0xFF == ord('h'):
            cv2.imwrite(directory+'H/'+str(count['H'])+'.jpg', roi)
        if interrupt & 0xFF == ord('i'):
            cv2.imwrite(directory+'I/'+str(count['I'])+'.jpg', roi)
        if interrupt & 0xFF == ord('j'):
            cv2.imwrite(directory+'J/'+str(count['J'])+'.jpg', roi)
        if interrupt & 0xFF == ord('k'):
            cv2.imwrite(directory+'K/'+str(count['K'])+'.jpg', roi)
        if interrupt & 0xFF == ord('l'):
            cv2.imwrite(directory+'L/'+str(count['L'])+'.jpg', roi)
        if interrupt & 0xFF == ord('m'):
            cv2.imwrite(directory+'M/'+str(count['M'])+'.jpg', roi)
        if interrupt & 0xFF == ord('n'):
            cv2.imwrite(directory+'N/'+str(count['N'])+'.jpg', roi)
        if interrupt & 0xFF == ord('o'):
            cv2.imwrite(directory+'O/'+str(count['O'])+'.jpg', roi)
        if interrupt & 0xFF == ord('p'):
            cv2.imwrite(directory+'P/'+str(count['P'])+'.jpg', roi)
        if interrupt & 0xFF == ord('q'):
            cv2.imwrite(directory+'Q/'+str(count['Q'])+'.jpg', roi)
        if interrupt & 0xFF == ord('r'):
            cv2.imwrite(directory+'R/'+str(count['R'])+'.jpg', roi)
        if interrupt & 0xFF == ord('s'):
            cv2.imwrite(directory+'S/'+str(count['S'])+'.jpg', roi)
        if interrupt & 0xFF == ord('t'):
            cv2.imwrite(directory+'T/'+str(count['T'])+'.jpg', roi)
        if interrupt & 0xFF == ord('u'):
            cv2.imwrite(directory+'U/'+str(count['U'])+'.jpg', roi)
        if interrupt & 0xFF == ord('v'):
            cv2.imwrite(directory+'V/'+str(count['V'])+'.jpg', roi)
        if interrupt & 0xFF == ord('w'):
            cv2.imwrite(directory+'W/'+str(count['W'])+'.jpg', roi)
        if interrupt & 0xFF == ord('x'):
            cv2.imwrite(directory+'X/'+str(count['X'])+'.jpg', roi)
        if interrupt & 0xFF == ord('y'):
            cv2.imwrite(directory+'Y/'+str(count['Y'])+'.jpg', roi)
        if interrupt & 0xFF == ord('z'):
            cv2.imwrite(directory+'Z/'+str(count['Z'])+'.jpg', roi)
cap.release()
cv2.destroyAllWindows()