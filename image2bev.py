# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 11:42:18 2020

@author: gao
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

class _DictObjHolder(object):
    def __init__(self, dct):
        self.dct = dct

    def __getattr__(self, name):
        return self.dct[name]
    
def rotX(a):
    a = np.deg2rad(a)
    R = np.array([[1,0,0],
                  [0,np.cos(a),-np.sin(a)],
                  [0,np.sin(a),np.cos(a)]])
    return R

def rotY(a):
    a = np.deg2rad(a)
    R = np.array([[np.cos(a),0,np.sin(a)],
                  [0,1,0],
                  [-np.sin(a),0,np.cos(a)]])
    return R

def rotZ(a):
    a = np.deg2rad(a)
    R = np.array([[np.cos(a),-np.sin(a),0],
                  [np.sin(a),np.cos(a),0],
                  [0,0,1]])
    return R 

def birdseyeviewimage(image,IntrinsicMatrix,CameraPose,OutImgView,OutImgeize):
    Pitch = CameraPose.Pitch
    Yaw = CameraPose.Yaw
    Roll = CameraPose.Roll
    Height = CameraPose.Height
    
    distAheadOfSensor = OutImgView.distAheadOfSensor
    spaceToLeftSide = OutImgView.spaceToLeftSide 
    spaceToRightSide = OutImgView.spaceToRightSide
    bottomOffset = OutImgView.bottomOffset
    
    outView = np.array([bottomOffset,distAheadOfSensor,-spaceToLeftSide,spaceToRightSide])
    reqImgHW = OutImgeize
    worldHW  = np.abs([outView[1]-outView[0], outView[3]-outView[2]])
    
    rotation = np.linalg.multi_dot([rotY(180),rotZ(-90),rotZ(Yaw),rotX(90-Pitch),rotZ(Roll)])
    rotationMatrix = np.linalg.multi_dot([rotZ(Yaw),rotX(90-Pitch),rotZ(Roll)])
    sl = [0,0]
    translationInWorldUnits = [sl[1], sl[0], Height]
    translation = np.dot(translationInWorldUnits,rotationMatrix)
    camMatrix = np.dot(np.vstack([rotation,translation]),IntrinsicMatrix)
    tform2D = np.array([camMatrix[0,:], camMatrix[1,:], camMatrix[3,:]])
    ImageToVehicleTransform = np.linalg.inv(tform2D)
    vehicleHomography = ImageToVehicleTransform
    adjTform = np.array([[0, -1, 0],
                         [-1, 0, 0],
                         [0, 0, 1]])
    bevTform = np.dot(vehicleHomography,adjTform)
    
    nanIdxHW = np.isnan(reqImgHW)
    scale   = (reqImgHW[~nanIdxHW]-1)/worldHW[~nanIdxHW]
    scaleXY = np.hstack([scale, scale])
    worldDim = worldHW[nanIdxHW]
    outDimFrac = scale*worldDim
    outDim     = np.round(outDimFrac)+1
    outSize = reqImgHW
    outSize[nanIdxHW] = outDim
    OutputView = outView
    dYdXVehicle = np.array([OutputView[3], OutputView[1]])
    tXY         = scaleXY*dYdXVehicle
    viewMatrix = np.array([[scaleXY[0], 0, 0],
                           [0, scaleXY[1], 0],
                           [tXY[0]+1, tXY[1]+1, 1]])
    BirdsEyeViewTransform = np.transpose(np.dot(bevTform, viewMatrix))
    birdsEyeViewImage = cv2.warpPerspective(image,BirdsEyeViewTransform,tuple(np.int_(np.flipud(outSize))))
    return birdsEyeViewImage

# Camera Pose
Pitch = -1
Yaw = -3
Roll = 0
Height = 1.15
CameraPose = _DictObjHolder({
        "Pitch": Pitch,
        "Yaw": Yaw,
        "Roll": Roll,
        "Height": Height,
        })

# IntrinsicMatrix
f_x = 1121.2
f_y = 1131.9
c_x = 963.4416
c_y = 526.8709
IntrinsicMatrix = np.array([[f_x,0,0],[0,f_y,0],[c_x,c_y,1]])

# Out Image View
distAheadOfSensor = 32
spaceToLeftSide = 8    
spaceToRightSide = 8
bottomOffset = 3
OutImageView = _DictObjHolder({
        "distAheadOfSensor": distAheadOfSensor,
        "spaceToLeftSide": spaceToLeftSide,
        "spaceToRightSide": spaceToRightSide,
        "bottomOffset": bottomOffset,
        })

OutImgeize = np.array([np.nan, 500])  # image H, image W

image = cv2.imread('000_000100.png')
birdseyeview = birdseyeviewimage(image, IntrinsicMatrix, CameraPose, OutImageView, OutImgeize)
plt.imshow(birdseyeview)



