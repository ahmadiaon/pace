import time
import datetime as dt
from datetime import datetime
import pygame
from pygame import mixer
import telepot
import cv2
import sys
import os
import numpy as np
from multiprocessing import Process

pygame.init()

now = datetime.now()
isSoundMask = 0
# 0 = no_body, 1 = isMask, 2 = noMask
sdThresh = 5
absolute_path = os.path.abspath(__file__)
dir_path = os.path.dirname(absolute_path)
directory = dir_path + "/Datasets/"
print(dir_path)
model_mask = cv2.CascadeClassifier(dir_path + '/Model/3900.xml')
no_mask_model = cv2.CascadeClassifier(dir_path + '/Model/3000nm.xml')
face_model = cv2.CascadeClassifier(dir_path + '/Model/haarcascades/haarcascade_frontalface_default.xml')
mouth_model = cv2.CascadeClassifier(dir_path + "/Opencv modules/Haarcascade_detection/Mouth.xml")
eyes_cascade = cv2.CascadeClassifier(dir_path + '/Model/haarcascades/haarcascade_eye.xml')

# SOUND
no_mask_sound = mixer.Sound(dir_path + '/assets/no_using_mask.wav')
mask_sound = mixer.Sound(dir_path + '/assets/masked.wav')
start_sound = mixer.Sound(dir_path + '/assets/start.wav')
test_sound = mixer.Sound(dir_path + '/assets/test_sound.wav')
goodby_sound = mixer.Sound(dir_path + '/assets/goodbye.wav')

bot = telepot.Bot('5416163618:AAHBBGGW22gxlbFixf06qlqVCIPqnDUKqtU')
chose_command = '\n - Sapa Bot -> /hi,\n - Test Suara -> /sound_test,\n - Lakukan Pengawasan -> /detect \n - Test Kamera -> /camera_test\n - Keluar -> /exit \n - Keluar Dan Matikan /poweroff'
chose_command_detect = 'break with :\n /break'
command = 'a'
count_handle = 1
count_handle_prev = 0
is_move = False
is_detect_object = True
end_time = dt.datetime.now()
is_motion = False
count = 0
n = 0
id_bot = 1693238711


def test_camera():
    global id_bot, dir_path
    print("Camera Take")
    vid = cv2.VideoCapture(0)

    while True:
        ret, frame = vid.read()

        # Display the resulting frame
        # cv2.imshow('frame', frame)
        t = time.localtime()
        current_time = time.strftime("%H%M%S", t)
        img_name = dir_path + "/Dokumen/test_" + current_time + ".jpg"
        cv2.imwrite(img_name, frame)
        bot.sendPhoto(id_bot, photo=open(img_name, 'rb'))
        time.sleep(1)
        break


def distance_map(frame_1, frame_2):
    """outputs pythagorean distance between two frames"""
    frame1_32 = np.float32(frame_1)
    frame2_32 = np.float32(frame_2)
    diff32 = frame1_32 - frame2_32
    norm32 = np.sqrt(diff32[:, :, 0] ** 2 + diff32[:, :, 1] ** 2 + diff32[:, :, 2] ** 2) / np.sqrt(
        255 ** 2 + 255 ** 2 + 255 ** 2)
    distance = np.uint8(norm32 * 255)
    return distance


def save_image_mask(images, group):
    print('on if')
    t = time.localtime()
    current_time = time.strftime("%H%M%S", t)
    img_name = group + current_time + ".jpg"
    cv2.imwrite(img_name, images)
    bot.sendPhoto(id_bot, photo=open(img_name, 'rb'))
    bot.sendMessage(id_bot, chose_command_detect)


def detection():
    global command, model_mask, no_mask_model, is_detect_object, end_time, count, dir_path, isSoundMask, mask
    is_detect_object = True
    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FPS, 5)
    print("Detection Start...")
    while True:
        if command == '/break':
            cv2.destroyAllWindows()
            break

        ret, img = cap.read()
        img = cv2.flip(img, 1)
        height, width = img.shape[0:2]

        # Garis Kiri kanan triger foto
        line_left = int(width * 0.3)
        line_right = int(width * 0.7)
        cv2.line(img, (int(width * 0.3), 0), (int(width * 0.3), height), (0, 0, 255), 2)
        cv2.line(img, (int(width * 0.7), 0), (int(width * 0.7), height), (0, 200, 255), 2)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mask = model_mask.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=9,
            minSize=(20, 20)
        )
        face = face_model.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(20, 20)
        )
        no_mask = no_mask_model.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=1,
            minSize=(20, 20)
        )
        eyes = eyes_cascade.detectMultiScale(gray, 1.1, 6)
        # eyes
        # for (ex, ey, ew, eh) in eyes:
        #     cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        # face
        if len(face) >= 1:
            for (x, y, w, h) in face:
                isSoundMask = 2
                object_left = int(x + w / 2)
                object_right = int(x + w / 2)
                cv2.rectangle(img, (x, y), (x + w, y + h), (90, 255, 0), 2)
                cv2.putText(img, 'Tanpa Masker', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 224, 0), 2)
                faces_without_mask = img[int(y * 0.8):y + int(h * 1.2), x:x + int(w * 1.2)]
                if line_left + 10 > object_left > line_left - 10:
                    count = count + 1
                    p = Process(target=save_image_mask,
                                args=(faces_without_mask, dir_path + '/Dokumen/without_mask/face_without_mask_',))
                    p.daemon = True
                    p.start()
                if line_right + 10 > object_right > line_right - 10:
                    count = count + 1
                    p = Process(target=save_image_mask,
                                args=(faces_without_mask, dir_path + '/Dokumen/without_mask/face_without_mask_',))
                    p.daemon = True
                    p.start()
        elif len(mask) >= 1:
            for (x, y, w, h) in mask:
                isSoundMask = 1
                object_left = int(x + w / 2)
                object_right = int(x + w / 2)
                cv2.rectangle(img, (x, y), (x + w, y + h), (225, 255, 0), 2)
                cv2.putText(img, 'Masker', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 224, 0), 2)
                faces_without_mask = img[int(y * 0.8):y + int(h * 1.2), x:x + int(w * 1.2)]
                if line_left + 10 > object_left > line_left - 10:
                    count = count + 1
                    p = Process(target=save_image_mask,
                                args=(faces_without_mask, dir_path + '/Dokumen/without_mask/mask_',))
                    p.daemon = True
                    p.start()
                if line_right + 10 > object_right > line_right - 10:
                    count = count + 1
                    p = Process(target=save_image_mask,
                                args=(faces_without_mask, dir_path + '/Dokumen/without_mask/mask_',))
                    p.daemon = True
                    p.start()

        cv2.imshow('LIVE', cv2.flip(img, 1))
        k = cv2.waitKey(30) & 0xff
        if k == 27:  # press 'ESC' to quit
            break

        # print(len(no_mask), len(mask))
        if isSoundMask == 1:
            if pygame.mixer.Channel(0).get_busy():
                print("masked")
            else:
                mask_sound.play()
        elif isSoundMask == 2:
            if pygame.mixer.Channel(0).get_busy():
                print("no mask")
            else:
                no_mask_sound.play()
        elif len(mask) == 0 and len(face) == 0:
            if is_detect_object:
                is_detect_object = False
                start_time = dt.datetime.now()
                end_time = start_time + dt.timedelta(seconds=5)
            elif not is_detect_object:
                print('No Faces')
                if dt.datetime.now() >= end_time:
                    print('is on return :', start_time)
                    print('is on return :', end_time)
                    is_no_detect = True
                    return is_no_detect
        isSoundMask = 0


def motion():
    global command, is_motion
    cap = cv2.VideoCapture(0)
    _, frame1 = cap.read()
    _, frame2 = cap.read()
    while True:
        if command == '/break':
            is_motion = False
            return is_motion
        _, frame3 = cap.read()
        rows, cols, _ = np.shape(frame3)
        dist = distance_map(frame1, frame3)

        frame1 = frame2
        frame2 = frame3

        # apply Gaussian smoothing
        mod = cv2.GaussianBlur(dist, (9, 9), 0)

        # apply thresholding
        _, thresh = cv2.threshold(mod, 100, 255, 0)

        # calculate st dev test
        _, st_dev = cv2.meanStdDev(mod)

        if st_dev > sdThresh:
            print("Motion detected.. Do something!!!")
            return True
        else:
            print('no movement')


def handle(msg):
    global command, count_handle, chose_command
    content_type, chat_type, chat_id = telepot.glance(msg)
    chat_id = msg['chat']['id']
    command = msg['text']

    count_handle += 1
    if content_type != 'text':
        bot.sendMessage(chat_id, "Wrong command,Type :" + chose_command)
    else:
        command = msg['text']
        print('Got command: %s', command)
        if command == '/exits':
            print('exit')

    return command


bot.sendMessage(id_bot, "System Mulai...")
start_sound.play()
bot.sendMessage(id_bot, "Chose your command," + chose_command)
bot.message_loop(handle)

while True:
    if count_handle == 1:
        print('I am Listening....')
        count_handle += 1
        count_handle_prev = 2
    elif count_handle_prev == count_handle:
        time.sleep(1)
    elif count_handle_prev < count_handle:
        print('new syntax')
        print(type(command), 'text :', command)
        if command == 'Hi' or command == '/hi':
            bot.sendMessage(id_bot, "Hello, Nice to meet you")
        elif command == '/detect':
            is_motion = True
            bot.sendMessage(id_bot, "Detection is begin...")
            while is_motion:
                is_move = motion()
                if is_move:
                    print('is movee')
                    is_detect = detection()
                if command == '/break':
                    break
        elif command == '/camera_test':
            bot.sendMessage(id_bot, "wait a minute, camera running....")
            test_camera()
            bot.sendMessage(id_bot, "Chose your command," + chose_command)
        elif command == '/sound_test':
            bot.sendMessage(id_bot, "Sound is Played....")
            test_sound.play()
            bot.sendMessage(id_bot, "Chose your command," + chose_command)
        elif command == '/exit':
            goodby_sound.play()

            bot.sendMessage(id_bot, "See You Next Time....")
            print('by')
            sys.exit()
        elif command == '/poweroff':
            bot.sendMessage(id_bot, "See you again...,")
            os.system("sudo shutdown -h now")
        else:
            bot.sendMessage(id_bot, "Sorry Command not found")
        count_handle_prev = count_handle