import cv2 as ComputerVision

Path_1 = 'D:\Studies\Project Motion detection\RandomWalking.mp4'
Path_2 = 'D:\Studies\Project Motion detection\Tracker.jpg'

Tracker = ComputerVision.imread(Path_2, ComputerVision.IMREAD_UNCHANGED)
Tracker_1 = ComputerVision.resize(Tracker, (500, 100))

camera = ComputerVision.VideoCapture(0)
_, Frame_1 = camera.read()
_, Frame_2 = camera.read()

Image_Count = 0

while camera.isOpened():

    Finding_difference = ComputerVision.absdiff(Frame_1, Frame_2)
    Convert_to_Grey = ComputerVision.cvtColor(Finding_difference, ComputerVision.COLOR_BGR2GRAY)
    Convert_to_Blur = ComputerVision.GaussianBlur(Convert_to_Grey, (5, 5), 0)
    Thresh_Value, Finding_Threshold = ComputerVision.threshold(Convert_to_Grey, 100, 255, ComputerVision.THRESH_BINARY)
    Dilated_Frame = ComputerVision.dilate(Finding_Threshold, None, iterations=3)
    Finding_Contours, Hierarchy = ComputerVision.findContours(Dilated_Frame, ComputerVision.RETR_EXTERNAL, ComputerVision.CHAIN_APPROX_SIMPLE)

    ComputerVision.putText(Frame_1, 'Status : ', (10, 40), ComputerVision.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)
    # ComputerVision.drawContours(Frame_1, Finding_Contours, -1, (255, 0, 0), 2)

    for Contour in Finding_Contours:

        if ComputerVision.contourArea(Contour) > 700:
            continue
        (x, y, w, h) = ComputerVision.boundingRect(Contour)
        Draw_Rectangle = ComputerVision.rectangle(Frame_1, (x, y), (x + w, y + h), (0, 255, 0), 2, ComputerVision.LINE_AA)
        ComputerVision.putText(Frame_1, 'Motion Detected', (80, 80), ComputerVision.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

        print('Image_' + str(Image_Count) + '_Saved')
        Image_Path = 'D:\Studies\Project Motion detection\Output_Images\Image_' + str(Image_Count) + '.jpeg'
        ComputerVision.imwrite(Image_Path, Frame_1)
        Image_Count += 1

    ComputerVision.imshow('Output', Frame_1)
    ComputerVision.imshow('Tracker', Tracker_1)
    Frame_1 = Frame_2
    _, Frame_2 = camera.read()
    key = ComputerVision.waitKey(1)
    if key == 27:
        break

camera.release()
ComputerVision.destroyAllWindows()
