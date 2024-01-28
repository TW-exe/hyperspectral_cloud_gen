#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 13:46:48 2023

L9 data products from 10 countries:
    - USA
    - Canada
    - Brazil
    - South Africa
    - Egypt
    - China
    - Australia
    - Russia
    - Japan 
    - UK
    
Preprocesses L9 data products by:
    - converting tif to numpy
    - converting DN to TOA reflectance using rescaling factors
    - correcting TOA reflectance for the sun angle
    - extracting cloudmasks from quality assessment band
    - saving cropped datacubes (8 bands) and cloudmasks as numpy arrays (this is used for labelling and training models)
Also,
    - saves thumbnails of full scene (uncropped) and cloudmask (uncropped) as png images 

@author: andrew
"""

from PIL import Image
import numpy as np
import pandas as pd
import os
import time
import xmltodict
import pprint

# data_dir = '/media/andrew/Expansion/Work-Machine-Backup-20.04/adversarial-cloud-detector-s2/datasets'

# ###############################################################################
# ############### CREATE A DATAFRAME OF NAMES OF L9 DATA PRODUCT ################
# ###############################################################################
# # create dataframe
# scene_df = pd.DataFrame(columns=['scene'])

# # define list to store name of scenes (name of data product w/o .SAFE)
# scene_folder = []

# # store name of scenes from 'products' folder (x70)
# for folder in sorted(os.listdir(f'{data_dir}/Landsat-9-Level-1/products')):
#     # print(folder)
#     head, sep, tail = folder.partition('.SAFE')
#     scene_folder.append(head)

# # store list of scenes in dataframe    
# scene_df['scene'] = scene_folder

# """
# output of 'scene_df['scene']': 
    
# 0     LC09_L1TP_014032_20230418_20230418_02_T1
# 1     LC09_L1TP_030042_20230418_20230418_02_T1
# 2     LC09_L1TP_039025_20230417_20230417_02_T1
# 3     LC09_L1TP_046022_20230418_20230418_02_T1
# 4     LC09_L1TP_049018_20230423_20230424_02_T1
# 5     LC09_L1TP_098085_20230415_20230415_02_T1
# 6     LC09_L1TP_101063_20230420_20230420_02_T1
# 7     LC09_L1TP_107034_20230414_20230414_02_T1
# 8     LC09_L1TP_107080_20230414_20230414_02_T1
# 9     LC09_L1TP_124038_20230421_20230421_02_T1
# 10    LC09_L1TP_132041_20230413_20230413_02_T1
# 11    LC09_L1TP_158073_20230419_20230420_02_T1
# 12    LC09_L1TP_175083_20230221_20230309_02_T1
# 13    LC09_L1TP_176022_20230417_20230417_02_T1
# 14    LC09_L1TP_177039_20230408_20230408_02_T1
# 15    LC09_L1TP_190025_20230419_20230420_02_T1
# 16    LC09_L1TP_199024_20230418_20230418_02_T1
# 17    LC09_L1TP_199025_20230418_20230418_02_T1
# 18    LC09_L1TP_215067_20230418_20230418_02_T1
# 19    LC09_L1TP_218070_20230423_20230423_02_T1
# Name: scene, dtype: object
# """

# ###############################################################################
# ########## CREATE A DATAFRAME OF BAND NAMES FOR EACH L9 DATA PRODUCT ##########
# ###############################################################################
# # create dataframe
# band_df = pd.DataFrame(columns=['band'])

# # define list of bands for each data product 
# band_folder = []

# # loop over number of scenes
# index = 0
# for index in range(0, len(scene_df)):
#     # store images names from 'bands' folder (14 images for each scene)
#     for folder in sorted(os.listdir(f'{data_dir}/Landsat-9-Level-1/products/' +  scene_df['scene'][index])):
#         # print(folder)
#         band_folder.append(folder)

# # store image name of bands in dataframe
# band_df['band'] = band_folder

# """
# output of 'band_df['band']': 

# 0       LC09_L1TP_014032_20230418_20230418_02_T1_ANG.txt
# 1        LC09_L1TP_014032_20230418_20230418_02_T1_B1.TIF
# 2       LC09_L1TP_014032_20230418_20230418_02_T1_B10.TIF
# 3       LC09_L1TP_014032_20230418_20230418_02_T1_B11.TIF
# 4        LC09_L1TP_014032_20230418_20230418_02_T1_B2.TIF
                       
# 395    LC09_L1TP_218070_20230423_20230423_02_T1_QA_RA...
# 396     LC09_L1TP_218070_20230423_20230423_02_T1_SAA.TIF
# 397     LC09_L1TP_218070_20230423_20230423_02_T1_SZA.TIF
# 398     LC09_L1TP_218070_20230423_20230423_02_T1_VAA.TIF
# 399     LC09_L1TP_218070_20230423_20230423_02_T1_VZA.TIF
# Name: band, Length: 400, dtype: object

# NOTE: 
# """

# ###############################################################################
# ######### CREATE NUMPY ARRAYS OF MULTISPECTRAL IMAGES AND CLOUD MASK ##########
# ###############################################################################
# # loop over number of scenes
# j = 0
# for i in range(len(scene_df)):
    
#     # print('i:', i)
#     # print('j:', j)
#     # j += 14
    
#     start_time = time.time()
    
#     print(scene_df['scene'][i])
    
#     ###########################################################################
#     ### BANDS ###
#     ###########################################################################
#     print('preprocessing bands of scene:', i)
 
#     # convert tif to numpy
#     print('convert tif to numpy')
#     b1 = np.asarray(Image.open(f'{data_dir}/Landsat-9-Level-1/products/' + scene_df['scene'][i] + '/' + band_df['band'][j+1]))
#     b2 = np.asarray(Image.open(f'{data_dir}/Landsat-9-Level-1/products/' + scene_df['scene'][i] + '/' + band_df['band'][j+4]))
#     b3 = np.asarray(Image.open(f'{data_dir}/Landsat-9-Level-1/products/' + scene_df['scene'][i] + '/' + band_df['band'][j+5]))
#     b4 = np.asarray(Image.open(f'{data_dir}/Landsat-9-Level-1/products/' + scene_df['scene'][i] + '/' + band_df['band'][j+6]))
#     b5 = np.asarray(Image.open(f'{data_dir}/Landsat-9-Level-1/products/' + scene_df['scene'][i] + '/' + band_df['band'][j+7]))
#     b6 = np.asarray(Image.open(f'{data_dir}/Landsat-9-Level-1/products/' + scene_df['scene'][i] + '/' + band_df['band'][j+8]))
#     b7 = np.asarray(Image.open(f'{data_dir}/Landsat-9-Level-1/products/' + scene_df['scene'][i] + '/' + band_df['band'][j+9]))
#     # b8 = np.asarray(Image.open(f'{data_dir}/Landsat-9-Level-1/products/' + scene_df['scene'][i] + '/' + band_df['band'][j+10]))
#     b9 = np.asarray(Image.open(f'{data_dir}/Landsat-9-Level-1/products/' + scene_df['scene'][i] + '/' + band_df['band'][j+11]))
#     # b10 = np.asarray(Image.open(f'{data_dir}/Landsat-9-Level-1/products/' + scene_df['scene'][i] + '/' + band_df['band'][j+2]))
#     # b11 = np.asarray(Image.open(f'{data_dir}/Landsat-9-Level-1/products/' + scene_df['scene'][i] + '/' + band_df['band'][j+3]))
#     qa = np.asarray(Image.open(f'{data_dir}/Landsat-9-Level-1/products/' + scene_df['scene'][i] + '/' + band_df['band'][j+14]))  

#     # ##################################
#     # print('.....STOP HERE.....')
#     # print('\n')
#     # from IPython import embed; embed()
#     # ##################################

#     # Convert metadata xml to dictionary
#     print('convert DN to TOA reflectance using rescaling factor')
#     with open(f'{data_dir}/Landsat-9-Level-1/products/' + scene_df['scene'][i] + '/' + band_df['band'][j+13], 'r', encoding='utf-8') as file:
#         xml = file.read()

#     xml_dict = xmltodict.parse(xml)

#     j += 20

#     # Multiplicative rescaling factors
#     mult_factor1 = float(xml_dict['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']['REFLECTANCE_MULT_BAND_1'])
#     mult_factor2 = float(xml_dict['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']['REFLECTANCE_MULT_BAND_2'])
#     mult_factor3 = float(xml_dict['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']['REFLECTANCE_MULT_BAND_3'])
#     mult_factor4 = float(xml_dict['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']['REFLECTANCE_MULT_BAND_4'])
#     mult_factor5 = float(xml_dict['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']['REFLECTANCE_MULT_BAND_5'])
#     mult_factor6 = float(xml_dict['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']['REFLECTANCE_MULT_BAND_6'])
#     mult_factor7 = float(xml_dict['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']['REFLECTANCE_MULT_BAND_7'])
#     mult_factor9 = float(xml_dict['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']['REFLECTANCE_MULT_BAND_9'])
    
#     # Additive rescaling factors
#     add_factor1 = float(xml_dict['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']['REFLECTANCE_ADD_BAND_1'])
#     add_factor2 = float(xml_dict['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']['REFLECTANCE_ADD_BAND_2'])
#     add_factor3 = float(xml_dict['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']['REFLECTANCE_ADD_BAND_3'])
#     add_factor4 = float(xml_dict['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']['REFLECTANCE_ADD_BAND_4'])
#     add_factor5 = float(xml_dict['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']['REFLECTANCE_ADD_BAND_5'])
#     add_factor6 = float(xml_dict['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']['REFLECTANCE_ADD_BAND_6'])
#     add_factor7 = float(xml_dict['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']['REFLECTANCE_ADD_BAND_7'])
#     add_factor9 = float(xml_dict['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']['REFLECTANCE_ADD_BAND_9'])
    
#     # Convert DN to TOA reflectance using rescaling factors (without correction for the sun angle) 
#     b1r = np.float32((mult_factor1 * b1) + add_factor1)
#     b2r = np.float32((mult_factor2 * b2) + add_factor2)
#     b3r = np.float32((mult_factor3 * b3) + add_factor3)
#     b4r = np.float32((mult_factor4 * b4) + add_factor4)
#     b5r = np.float32((mult_factor1 * b5) + add_factor5)
#     b6r = np.float32((mult_factor2 * b6) + add_factor6)
#     b7r = np.float32((mult_factor3 * b7) + add_factor7)
#     b9r = np.float32((mult_factor4 * b9) + add_factor9)
    
#     # TOA reflectance with correction for the sun angle
#     print('correct TOA reflectance for the sun angle')
#     sun_elevation = float(xml_dict['LANDSAT_METADATA_FILE']['IMAGE_ATTRIBUTES']['SUN_ELEVATION']) # in degrees
    
#     b1r = b1r / np.sin(sun_elevation * (np.pi/180))
#     b2r = b2r / np.sin(sun_elevation * (np.pi/180))
#     b3r = b3r / np.sin(sun_elevation * (np.pi/180))
#     b4r = b4r / np.sin(sun_elevation * (np.pi/180))
#     b5r = b5r / np.sin(sun_elevation * (np.pi/180))
#     b6r = b6r / np.sin(sun_elevation * (np.pi/180))
#     b7r = b7r / np.sin(sun_elevation * (np.pi/180))
#     b9r = b9r / np.sin(sun_elevation * (np.pi/180))

#     # Concatenate bands
#     print('concatenate bands together')
#     bands = np.stack((b1r, b2r, b3r, b4r, b5r, b6r, b7r, b9r))
    
#     # move axis 0 to axis 2 - 7911x7831x13
#     print('move axis of band')
#     bands = np.moveaxis(bands, 0, 2)
    
#     # Save RGB image
#     print('save RGB image')
#     band432 = np.stack((b4r, b3r, b2r))
#     band432 = np.moveaxis(band432, 0, 2)
#     image = Image.fromarray(np.uint8(band432*255))
#     # image.show()
#     image.save(f'{data_dir}/Landsat-9-Level-1/thumbnails/scenes/' + scene_df['scene'][i]+ '.png')
    
#     # Crop scene into 512x512 subsences and save in PNG and NUMPY formats
#     print('crop scene into 512x512 subsences and save in PNG and NUMPY formats')
#     filename = scene_df['scene'][i]
#     band_image = bands
    
#     # Crop numpy array [row, column, bands]
#     k = 0
#     for row in range((int(band_image.shape[0]/2))-2745, int((band_image.shape[0]/2))+2745, 549):
#         # print('k:', k)
#         print('row:', row)
#         for column in range((int(band_image.shape[1]/2))-2745, int((band_image.shape[1]/2))+2745, 549):
#             print('column:', column)
#             crop = band_image[row:row+512, column:column+512, :]
            
#             # save cropped (512x512x13) png images of band 4,3,2 (RGB)
#             crop432 = crop[..., [3, 2, 1]]
#             image = Image.fromarray(np.uint8(crop432*255))
#             image.save(f'{data_dir}/Landsat-9-Level-1/preprocessed/png/bands432/{filename}_{k}.png')
            
#             # save cropped (512x512x13) numpy arrays of 5490x5490x13
#             np.save(f'{data_dir}/Landsat-9-Level-1/preprocessed/numpy/images/{filename}_{k}', crop)
    
#             k += 1

#     ###########################################################################
#     ### CLOUD MASK ### 
#     ###########################################################################
#     # Extract cloud mask from quality assessment band
#     print('extract cloud mask from quality assessment band')
#     integer_mask = qa
    
#     binary_repr_v = np.vectorize(np.binary_repr)
#     binary_mask = binary_repr_v(integer_mask, 16)
    
#     cm = np.zeros(shape=(binary_mask.shape[0], binary_mask.shape[1]))
#     for row in range(cm.shape[0]):
#         for column in range(cm.shape[1]):
#             cm[row,column] = binary_mask[row,column][-4]
    
#     # Save cloud mask
#     image = Image.fromarray(np.uint8(cm*255))
#     # image.show()
#     image.save(f'{data_dir}/Landsat-9-Level-1/thumbnails/cloudmasks/' + scene_df['scene'][i]+ '.png')

#     # Crop numpy array [row, column]
#     k = 0
#     for row in range((int(cm.shape[0]/2))-2745, int((cm.shape[0]/2))+2745, 549):
#         # print('k:', k)
#         print('row:', row)
#         for column in range((int(cm.shape[1]/2))-2745, int((cm.shape[1]/2))+2745, 549):
#             print('column:', column)      
#             crop = cm[row:row+512, column:column+512]*255
            
#             # Save cropped (512x512) png images of cloud mask
#             image = Image.fromarray(np.uint8(crop))
#             # image.show()
#             image.save(f'{data_dir}/Landsat-9-Level-1/preprocessed/png/cloudmasks/{filename}_{k}.png')
            
#             # Save cropped (512x512) numpy arrays of cloud mask
#             np.save(f'{data_dir}/Landsat-9-Level-1/preprocessed/numpy/cloudmasks/{filename}_{k}', crop)
    
#             k += 1    
    
#     finish_time = time.time()
#     print('time taken to preprocess one scene:', finish_time - start_time)










# Landsat 9 product directory
product_dict = 'datasets/Landsat-9-Level-1/products'

# Convert TIFF to NPY
print('convert tif to numpy')
b1 = np.asarray(Image.open(f'{product_dict}/LC09_L1TP_114063_20240112_20240112_02_T1/LC09_L1TP_114063_20240112_20240112_02_T1_B1.TIF'))
b2 = np.asarray(Image.open(f'{product_dict}/LC09_L1TP_114063_20240112_20240112_02_T1/LC09_L1TP_114063_20240112_20240112_02_T1_B1.TIF'))
b3 = np.asarray(Image.open(f'{product_dict}/LC09_L1TP_114063_20240112_20240112_02_T1/LC09_L1TP_114063_20240112_20240112_02_T1_B1.TIF'))
b4 = np.asarray(Image.open(f'{product_dict}/LC09_L1TP_114063_20240112_20240112_02_T1/LC09_L1TP_114063_20240112_20240112_02_T1_B1.TIF'))
b5 = np.asarray(Image.open(f'{product_dict}/LC09_L1TP_114063_20240112_20240112_02_T1/LC09_L1TP_114063_20240112_20240112_02_T1_B1.TIF'))
b6 = np.asarray(Image.open(f'{product_dict}/LC09_L1TP_114063_20240112_20240112_02_T1/LC09_L1TP_114063_20240112_20240112_02_T1_B1.TIF'))
b7 = np.asarray(Image.open(f'{product_dict}/LC09_L1TP_114063_20240112_20240112_02_T1/LC09_L1TP_114063_20240112_20240112_02_T1_B1.TIF'))
b9 = np.asarray(Image.open(f'{product_dict}/LC09_L1TP_114063_20240112_20240112_02_T1/LC09_L1TP_114063_20240112_20240112_02_T1_B1.TIF'))
qa_pixel = np.asarray(Image.open(f'{product_dict}/LC09_L1TP_114063_20240112_20240112_02_T1/LC09_L1TP_114063_20240112_20240112_02_T1_QA_PIXEL.TIF'))

# Convert metadata xml to dictionary
print('convert DN to TOA reflectance using rescaling factor')
with open(f'{product_dict}/LC09_L1TP_114063_20240112_20240112_02_T1/LC09_L1TP_114063_20240112_20240112_02_T1_MTL.xml', 'r', encoding='utf-8') as file:
    xml = file.read()

xml_dict = xmltodict.parse(xml)

# Multiplicative rescaling factors
mult_factor1 = float(xml_dict['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']['REFLECTANCE_MULT_BAND_1'])
mult_factor2 = float(xml_dict['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']['REFLECTANCE_MULT_BAND_2'])
mult_factor3 = float(xml_dict['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']['REFLECTANCE_MULT_BAND_3'])
mult_factor4 = float(xml_dict['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']['REFLECTANCE_MULT_BAND_4'])
mult_factor5 = float(xml_dict['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']['REFLECTANCE_MULT_BAND_5'])
mult_factor6 = float(xml_dict['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']['REFLECTANCE_MULT_BAND_6'])
mult_factor7 = float(xml_dict['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']['REFLECTANCE_MULT_BAND_7'])
mult_factor9 = float(xml_dict['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']['REFLECTANCE_MULT_BAND_9'])

# Additive rescaling factors
add_factor1 = float(xml_dict['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']['REFLECTANCE_ADD_BAND_1'])
add_factor2 = float(xml_dict['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']['REFLECTANCE_ADD_BAND_2'])
add_factor3 = float(xml_dict['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']['REFLECTANCE_ADD_BAND_3'])
add_factor4 = float(xml_dict['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']['REFLECTANCE_ADD_BAND_4'])
add_factor5 = float(xml_dict['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']['REFLECTANCE_ADD_BAND_5'])
add_factor6 = float(xml_dict['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']['REFLECTANCE_ADD_BAND_6'])
add_factor7 = float(xml_dict['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']['REFLECTANCE_ADD_BAND_7'])
add_factor9 = float(xml_dict['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING']['REFLECTANCE_ADD_BAND_9'])

# Convert DN to TOA reflectance using rescaling factors (without correction for the sun angle) 
b1r = np.float32((mult_factor1 * b1))
b2r = np.float32((mult_factor2 * b2))
b3r = np.float32((mult_factor3 * b3))
b4r = np.float32((mult_factor4 * b4))
b5r = np.float32((mult_factor1 * b5))
b6r = np.float32((mult_factor2 * b6))
b7r = np.float32((mult_factor3 * b7))
b9r = np.float32((mult_factor4 * b9))

# TOA reflectance with correction for the sun angle
print('correct TOA reflectance for the sun angle')
sun_elevation = float(xml_dict['LANDSAT_METADATA_FILE']['IMAGE_ATTRIBUTES']['SUN_ELEVATION']) # degrees

b1r = b1r / np.cos(sun_elevation * (np.pi/180))
b2r = b2r / np.cos(sun_elevation * (np.pi/180))
b3r = b3r / np.cos(sun_elevation * (np.pi/180))
b4r = b4r / np.cos(sun_elevation * (np.pi/180))
b5r = b5 / np.cos(sun_elevation * (np.pi/180))
b6r = b6 / np.cos(sun_elevation * (np.pi/180))
b7r = b7r / np.cos(sun_elevation * (np.pi/180))
b9r = b9r / np.cos(sun_elevation * (np.pi/180))

# Concatenate bands
print('concatenate bands together')
bands = np.stack((b1r, b2r, b3r, b4r, b5r, b6r, b7r, b9r))

# move axis 0 to axis 2 - 7911x7831x13
print('move axis of band')
bands = np.moveaxis(bands, 0, 2)

# Save RGB image
band432 = np.stack((b4r, b3r, b2r))
band432 = np.moveaxis(band432, 0, 2)
image = Image.fromarray(np.uint8(band432*255))

# image.show()
savedir = "C:\Users\thoma\AIML2024Summer\preprocess_results"

# # crop numpy array [row, column, bands]
# k = 0
# for row in range((int(band_image.shape[0]/2))-2500, int((band_image.shape[0]/2))+2500, 549):
#     # print('k:', k)
#     print('row:', row)
#     for column in range((int(band_image.shape[0]/2))-2500, int((band_image.shape[0]/2))+2500, 549):
#         print('column:', column)      





############################################################################################################################
################ not neccesary as frame will be scaled down and cloud mask will be applied separately ######################
############################################################################################################################


# # image.show()
# savedir = "C:\Users\thoma\AIML2024Summer\preprocess_results"


# print('crop scene into 512x512 subsences and save in PNG and NUMPY formats')

# filename = "LC09_L1TP_039025_20230417_20230417_02_T1"

# #image.save(f'{savedir}/png/bands432/{filename}_{"1"}.png')


# band_image = bands

# # crop numpy array [row, column, bands]
# k = 0
# for row in range((int(band_image.shape[0]/2))-2500, int((band_image.shape[0]/2))+2500, 549):
#     # print('k:', k)
#     print('row:', row)
#     for column in range((int(band_image.shape[0]/2))-2500, int((band_image.shape[0]/2))+2500, 549):
#         print('column:', column)      
#         crop = band_image[row:row+512, column:column+512, :]
        
#         # save cropped (512x512x13) png images of band 4,3,2 (RGB)
#         crop432 = crop[..., [3, 2, 1]]
#         image = Image.fromarray(np.uint8(crop432*255))
#         image.save(f'{savedir}/png/bands432/{filename}_{k}.png')
        
#         # save cropped (512x512x13) numpy arrays of 5490x5490x13
#         np.save(f'{savedir}/numpy/images/{filename}_{k}', crop)

#         k += 1

# ###########################################################################
# ### CLOUD MASK ### 
# ###########################################################################
# # Extract cloud mask from quality assessment band
# print('extract cloud mask from quality assessment band')
# integer_mask = np.asarray(Image.open(f'{product_dict}/LC09_L1TP_114063_20240112_20240112_02_T1/LC09_L1TP_114063_20240112_20240112_02_T1_QA_PIXEL.TIF'))

# binary_repr_v = np.vectorize(np.binary_repr)
# binary_mask = binary_repr_v(integer_mask, 16)

# cloud_mask = np.zeros(shape=(binary_mask.shape[0], binary_mask.shape[1]))
# for i in range(cloud_mask.shape[0]):
#     for j in range(cloud_mask.shape[1]):
#         cloud_mask[i,j] = binary_mask[i,j][-4]

# image = Image.fromarray(np.uint8(cloud_mask*255))
# image.show()











