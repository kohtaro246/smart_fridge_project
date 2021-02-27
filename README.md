# smart_fridge_project

This is a program to manage expiration dates of items that are in a refrigerator using python.

# DEMO

Link to demo video: https://drive.google.com/file/d/1Yr306eEAU-251A7ezx_wUpb-QWhKtHqd/view?usp=sharing


# Features

This program has the following functions

1. Recognize expiration dates of products using OCR  
2. Recognize 5 types of plates  
3. Store data of the following:  
      * Picture of the product/plate  
      * Expiration dates  
      * Where it is stored  
4. Interface which allows users to do the following:  
      * Select from list of products/plates close to expiration date  
      * See picture of the products/plates close to expiration date  
      * See where the product/plate close to expiration is  
5. Detect where the new product/plate was placed using a camera  

# Requirement

* python 2.7.17
* mysqlclient 1.4.6
* pytesseract 0.3.6
* OpenCV 3.2.0
* Numpy 1.16.6
* PySimpleGUI27 2.4.1
* scikit-image 0.14.5
* imutils 0.5.3


# Usage

Clone to directory:
```bash
git clone https://github.com/kohtaro246/smart_fridge_project.git
```
Register items:
```bash
python final.py
```
Check and retrieve items:
```bash
python final_ret.py
```
# Performance
・Plate recognition accuracy：94%
・Plate recognition time：1 sec
・Expiration date recognition：< 5 sec
・Registration time（WITHOUT tracking）：10 sec/item
・Registration time（WITH tracking）：25 sec/item
・Check / Retrieval time：10 sec/item


# Author

* Kohtaro Tanaka
* University of Tokyo Mechano-Informatics


