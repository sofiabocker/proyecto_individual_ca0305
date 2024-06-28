#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 19:08:40 2024

@author: sofiabocker
"""

import time
import random
from tkinter import *
from threading import Thread
from PIL import Image, ImageTk
from pynput.keyboard import Listener

import settings as st
import text as tt

class TypingTest:
    
    def __init__(self, root):
        # Window Settings
        self.window = root
        self.window.geometry(f"{st.width}x{st.height}")
        self.window.title('Typing Tester')
        self.window.resizable(width=False, height=False)
        self.window.configure(bg=st.color2)
        
        # Declaring some variables
        self.key = None
        self.typingStarted = False
        self.totalTime = st.totalTime
    
        # Text for using as a paragraph
        self.textList = [tt.text1, tt.text2, tt.text3, tt.text4, tt.text5]
    
        # Tkinter Frame
        self.frame = Frame(self.window, bg=st.color2, width=st.width, height=st.height)
        self.frame.place(x=0, y=0)
    
        # Calling the function, startWindow()
        self.startWindow()
        
    def startWindow(self):
        # A Tkinter Label to display the title
        titleLabel = Label(self.frame, text="Typing Speed Test", bg=st.color2, fg=st.color1 ,font=(st.font3, 35, "bold"))
        titleLabel.place(x=175, y=80)

        startButton = Button(self.frame, text="Start Here", border=0, cursor='hand2' ,fg=st.color2, bg=st.color1, font=(st.font4, 18), command=self.startTest)
        startButton.place(x=320, y=215)
        
    def startTest(self):
        # Clearing the previous screen
        self.clearScreen()

        # Getting the total time allocated for the test
        self.totalTime = st.totalTime

        # Choosing a random paragraph from the list of several choices
        self.paragraph = random.choice(self.textList)

        # A Label widget for showing the remaining time
        self.timeLabel = Label(self.frame, text="1:00", bg=st.color2, fg=st.color1, font=(st.font1, 15))
        self.timeLabel.place(x=20, y=15)

        # A Tkinter Text widget to display the paragraph 
        textBox = Text(self.frame, width=65, height=10, bg=st.color1, font=(st.color2, 13))
        textBox.place(x=40, y=80)
        
        # Inserting the text into the Text widget
        textBox.insert(END, self.paragraph)
        textBox.config(state='disabled')

        # A Tkinter Entry widget to get input from the user
        self.inputEntry = Entry(self.frame, fg=st.color2, bg=st.color1, width=35, font=(st.font4, 20))
        self.inputEntry.place(x=100, y=360)
        
        # Define the on press to capture key pressing
        ls = Listener(on_press=self.listener)
        # Starting a different thread for this task
        ls.start()
                
    def listener(self, key):
        self.key = str(key)

        # If any key is pressed, a different thread will create
        # to Count Down the remaining time.
        if self.key is not None:
            self.typingStarted = True
            self.multiThreading()
        
        # Returning False to stop the thread created to
        # capture key pressing.
        return False
                
    def multiThreading(self):
        x = Thread(target=self.countDown)
        x.start()
        
    def countDown(self):
        while self.totalTime > 0:
            # Updating the Time Label
            self.timeLabel.config(text=f"{0}:{self.totalTime}")
            time.sleep(1)
            self.totalTime -= 1

        self.timeLabel.config(text=f"{0}:{0}")
        # Calling the Function to Calculate the Final Result
        self.calculateResult()
        
    def backgroundImage(self, img):
        # Opening the image
        image = Image.open(img)
        # Resize the image to fit to the screen
        resizedImg = image.resize((st.width, st.height))
        # Creating an instance of PhotoImage class of ImageTk module
        self.img = ImageTk.PhotoImage(resizedImg)

        label = Label(self.frame, image=self.img)
        label.pack()
        
    def calculateResult(self):
        # Getting the text from the Tkinter Entry Widget and storing
        # it to a variable called text.
        text = self.inputEntry.get()
        # Splitting the text by choosing the SPACE as a delimiter and
        # storing the words in a list.
        text = text.split(" ")

        # Clearing the previous screen
        self.clearScreen()
        # Setting a Background image
        self.backgroundImage(st.resultImage)

        # wpm(word per minute)
        wpm = len(text)
        # A variable for counting the correct answer from the given
        correctAnswer = 0
        
        # Splitting the main paragraph by choosing the SPACE as a 
        # delimiter and storing the words in a list.
        mainText = self.paragraph.split(" ")
        # Excluding the word redundancy from the list by converting
        # it to a python set.
        mainText = set(mainText)
        
        # Iterate through the given text by the user
        for word in text:
            # If the word is present in the Main Paragraph
            # increment the correctAnswer variable by one.
            if word in mainText:
                correctAnswer += 1

        # Calculating the accuracy of the given text by the user
        accuracy = (correctAnswer / wpm) * 100
        # Calculating the net speed after applying the 
        # percentage of the accuracy
        netSpeed = (wpm / 100) * accuracy
        
        # Label to show the wpm(word per minute)
        wpmLabel = Label(self.frame, text=wpm, bg=st.color1, fg=st.color2, font=(st.font2, 24))
        wpmLabel.place(x=160, y=186)

        # Label to show the accuracy
        accuracyLabel = Label(self.frame, text=f"{int(accuracy)}%", bg=st.color1, fg=st.color2, font=(st.font2, 24))
        accuracyLabel.place(x=366, y=186)

        # Label to show the net speed
        netSpeedLabel = Label(self.frame, text=int(netSpeed), bg=st.color1, fg=st.color2, font=(st.font2, 24))
        netSpeedLabel.place(x=612, y=186)
        
        # A Button to start the test again
        resetButton = Button(self.frame, text="Test Again", border=0, cursor='hand2' ,fg=st.color2, bg=st.color1, font=(st.font4, 18), command=self.startTest)
        resetButton.place(x=320, y=360)
        
    def clearScreen(self):
        for widget in self.frame.winfo_children():
            widget.destroy()

# Main code block should be outside the class
if __name__ == "__main__":
    # Instance of Tk class
    root = Tk()
    # Object of TypingTest class
    obj = TypingTest(root)
    root.mainloop()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
