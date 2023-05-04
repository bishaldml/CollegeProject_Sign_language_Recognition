import os
import cv2

if not os.path.exists("data"):
    os.makedirs("data")
    # train folder
    os.makedirs("data/train")
    os.makedirs("data/train/0")
    os.makedirs("data/train/1")
    os.makedirs("data/train/2")
    os.makedirs("data/train/3")
    os.makedirs("data/train/4")
    os.makedirs("data/train/5")
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
    os.makedirs("data/train/N")

   
   
    # test folder
    os.makedirs("data/test")
    os.makedirs("data/test/0")
    os.makedirs("data/test/1")
    os.makedirs("data/test/2")
    os.makedirs("data/test/3")
    os.makedirs("data/test/4")
    os.makedirs("data/test/5")
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
    os.makedirs("data/test/N")


mode = 'test'
directory = 'data/'+mode+'/'

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    # Flip the image horizontally for a selfie-view display.
    frame = cv2.flip(frame, 1)

    # Getting count of existing images
    count = {'0': len(os.listdir(directory+"/0")),
         '1': len(os.listdir(directory+"/1")),
         '2': len(os.listdir(directory+"/2")),
         '3': len(os.listdir(directory+"/3")),
         '4': len(os.listdir(directory+"/4")),
         '5': len(os.listdir(directory+"/5")),
         'A': len(os.listdir(directory+"/A")),
         'B': len(os.listdir(directory+"/B")),
         'C': len(os.listdir(directory+"/C")),
         'D': len(os.listdir(directory+"/D")),
         'E': len(os.listdir(directory+"/E")),
         'F': len(os.listdir(directory+"/F")),
         'G': len(os.listdir(directory+"/G")),
         'H': len(os.listdir(directory+"/H")),
         'I': len(os.listdir(directory+"/I")),
         'J': len(os.listdir(directory+"/J")),
         'N': len(os.listdir(directory+"/N"))
         
        }

    # Printing the count in each set to the screen/frame
    cv2.putText(frame, "MODE : "+mode, (10, 10), cv2.FONT_HERSHEY_PLAIN, 0.9, (0,0,255), 1)
    cv2.putText(frame, "IMAGE COUNT", (10, 20), cv2.FONT_HERSHEY_PLAIN, 0.9, (0,0,255), 1)
    cv2.putText(frame, "Zero : "+str(count['0']), (10, 30), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,0,0), 1)
    cv2.putText(frame, "One : "+str(count['1']), (10, 45), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,0,0), 1)
    cv2.putText(frame, "Two : "+str(count['2']), (10, 60), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,0,0), 1)
    cv2.putText(frame, "Three : "+str(count['3']), (10, 75), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,0,0), 1)
    cv2.putText(frame, "Four : "+str(count['4']), (10, 90), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,0,0), 1)
    cv2.putText(frame, "Five : "+str(count['5']), (10, 105), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,0,0), 1)
    
    cv2.putText(frame, "A : "+str(count['A']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,0,0), 1)
    cv2.putText(frame, "B : "+str(count['B']), (10, 135), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,0,0), 1)
    cv2.putText(frame, "C : "+str(count['C']), (10, 150), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,0,0), 1)
    cv2.putText(frame, "D : "+str(count['D']), (10, 165), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,0,0), 1)
    cv2.putText(frame, "E : "+str(count['E']), (10, 180), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,0,0), 1)
    cv2.putText(frame, "F : "+str(count['F']), (10, 195), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,0,0), 1)
    cv2.putText(frame, "G : "+str(count['G']), (10, 210), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,0,0), 1)    
    cv2.putText(frame, "H : "+str(count['H']), (10, 225), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,0,0), 1)    
    cv2.putText(frame, "I : "+str(count['I']), (10, 240), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,0,0), 1)    
    cv2.putText(frame, "J : "+str(count['J']), (10, 255), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,0,0), 1)  
    cv2.putText(frame, "Null : "+str(count['N']), (10, 265), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,0,0), 1)  


    
    # Coordinates of the ROI on frame
    x1 = 400
    y1 = 100
    x2 = 620
    y2 = 320
    # Drawing the ROI on frame. The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0) ,2)
   
    
    cv2.imshow("Frame: Data Collection", frame)


    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, (128, 128)) 
   
    ## Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    gaussian = cv2.GaussianBlur(gray,(5,5),0)
    
    #thres = cv2.adaptiveThreshold(gaussian,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,3)
   
    _,thres = cv2.threshold(gaussian, 10, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
   
   
    cv2.imshow("ROI", thres)
    #print(thres.shape)

    
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break
    if interrupt & 0xFF == ord('0'):
        cv2.imwrite(directory+'0/'+str(count['0'])+'.jpg', thres)
    if interrupt & 0xFF == ord('1'):
        cv2.imwrite(directory+'1/'+str(count['1'])+'.jpg', thres)
    if interrupt & 0xFF == ord('2'):
        cv2.imwrite(directory+'2/'+str(count['2'])+'.jpg', thres)
    if interrupt & 0xFF == ord('3'):
        cv2.imwrite(directory+'3/'+str(count['3'])+'.jpg', thres)
    if interrupt & 0xFF == ord('4'):
        cv2.imwrite(directory+'4/'+str(count['4'])+'.jpg', thres)
    if interrupt & 0xFF == ord('5'):
        cv2.imwrite(directory+'5/'+str(count['5'])+'.jpg', thres)

    if interrupt & 0xFF == ord('a'):
        cv2.imwrite(directory+'A/'+str(count['A'])+'.jpg', thres)
    if interrupt & 0xFF == ord('b'):
        cv2.imwrite(directory+'B/'+str(count['B'])+'.jpg', thres)
    if interrupt & 0xFF == ord('c'):
        cv2.imwrite(directory+'C/'+str(count['C'])+'.jpg', thres)
    if interrupt & 0xFF == ord('d'):
        cv2.imwrite(directory+'D/'+str(count['D'])+'.jpg', thres)
    if interrupt & 0xFF == ord('e'):
        cv2.imwrite(directory+'E/'+str(count['E'])+'.jpg', thres)
    if interrupt & 0xFF == ord('f'):
        cv2.imwrite(directory+'F/'+str(count['F'])+'.jpg', thres)
    if interrupt & 0xFF == ord('g'):
        cv2.imwrite(directory+'G/'+str(count['G'])+'.jpg', thres)
    if interrupt & 0xFF == ord('h'):
        cv2.imwrite(directory+'H/'+str(count['H'])+'.jpg', thres)
    if interrupt & 0xFF == ord('i'):
        cv2.imwrite(directory+'I/'+str(count['I'])+'.jpg', thres)
    if interrupt & 0xFF == ord('j'):
        cv2.imwrite(directory+'J/'+str(count['J'])+'.jpg', thres)
    if interrupt & 0xFF == ord('n'):
        cv2.imwrite(directory+'N/'+str(count['N'])+'.jpg', thres)

cap.release()
cv2.destroyAllWindows()
