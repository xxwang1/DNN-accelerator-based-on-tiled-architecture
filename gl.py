import math
import numpy as np
import os
#global variables
data = {'input_activation':[], 'weight_array':[], 'output_activation':[]}
cb_size = (256, 64) #crossbar dimensions
(w_tot, w_int) = (16, 5)
crossbar_set = []
input_set=[]
mapping_mode = 'dual_array' # (ideal, dual_array, neg_act)
scale_mode = 'per_column' #(per_layer, per_block, per_column)
scale_number = '1' #(1,2,4,8,arbitrary)
onoff_ratio = 'infinite' #(infinite, 10, 100, 1000)
latency = []
energy = []

# 2 different types of rounding: exponential and linear
# when using one type of rounding, comment out the other type
def scale2bit(a): #Per-layer 8bit-quantization scale
    ##exponential round - returns a positive power of 2 (or 0 if a = 0)
    # if a > 0:
    #     return 2**round(math.log(a, 2))
    # elif a < 0:
    #     return -2**round(math.log(-a, 2))
    # else:
    return a

    ##linear round - returns an integer that is between 50 to 800 if a is postive and -50 to -800 if a is negative
    # if a > 800:
    #     return 800
    # elif a < -800:
    #     return -800
    # elif a > 0:
    #     return math.floor((a+75)/250)*250+50
    # elif a < 0:
    #     return math.ceil((a-75)/250)*250-50
    # else:
    #     return a

def array2bit(a): #Per-array of per-column scale
    if mapping_mode == 'dual_array':
        
        #execute different scenarios based on the value of onoff_ratio
        if onoff_ratio == '10':
            
            #execute different rounding methods based on the value of scale_number
            if scale_number == '4':
                #6600 7100 7600 8100
                a = np.round((a-6100) / 500) * 500 + 6100
                a[a>8100] = 8100
                a[a<6600] = 6600
                return a
            elif scale_number == '8':
                #6600 6800 7000 7200 7400 7600 7800 8000
                a = np.round((a-6400)/200)*200 +6400
                a[a>8000] = 8000
                a[a<6600] = 6600
                return a
            elif scale_number == 'arbitrary':
                return a
        elif onoff_ratio == '100':

            #execute different rounding methods based on the value of scale_number
            if scale_number == '1':
                return np.ones_like(a) * 2800
            elif scale_number == '2':#1400, 2800
                a = np.ceil(a/1400)
                a[a>2800] = 2800
                return a
            elif scale_number == '4':#700 1400 2100 2800
                a = np.round(a/700) * 700
                a[a>2800] = 2800
                return a
            elif scale_number == '8':#700 850 1000 1150 1300 1450 1600 1750
                a = np.round((a-550)/150)*150+550
                a[a>1750] = 1750
                return a
            elif scale_number == 'arbitrary':
                return a
        elif onoff_ratio == '1000':

            #execute different rounding methods based on the value of scale_number
            # if scale_number == '1':
            #     return np.ones_like(a) * 2800
            # elif scale_number == '2':#1400, 2800
            #     a = np.ceil(a/1400)
            #     a[a>2800] = 2800
            #     return a
            if scale_number == '4':
                #100 433 766 1099
                a = np.round((a+233)/333)*333-233
                a[a>1099] = 1099
                a[a<100] = 100
                return a
            elif scale_number == '8':
                #75 225 375 525 675 825 975 1125
                a = np.round((a+75)/150)*150-75
                a[a>1125] = 1125
                a[a<75] = 75
                return a
            elif scale_number == 'arbitrary':
                return a
        elif onoff_ratio == 'infinite':

            #execute different rounding methods based on the value of scale_number
            if scale_number == '1':
                return np.ones_like(a) * 1100
            elif scale_number == '2':#1000, 2000
                a = np.ceil(a/1000)
                a[a>2000] = 2000
                return a
            elif scale_number == '4':#62.5 375 687.5 1000
                # a = np.round((a+250)/312.5)*312.5-250
                # a[a>1000] = 1000
                # a[a<62.5] = 62.5
                # #35 357 679 1001
                # a = np.round((a+287)/322)*322-287
                # a[a>1001] = 1001
                #30 355 680 1005
                # a = np.round((a+295)/325)*325-295
                # a[a>1005] = 1005
                #30 320 610 900
                # a = np.round((a+260)/290)*290-260
                # a[a>900] = 900
                #30 387 744 1101
                # a = np.round((a+327)/357)*357-327
                # a[a>1101] = 1101
                #25 350 675 1000
                # a = np.round((a+300)/325)*325-300
                # a[a>1000] = 1000
                #50 370 690 1010
                # a = np.round((a+270)/320)*320-270
                # a[a>1010] = 1010
                # a[a<50] = 50
                #75 425 775 1125
                # a = np.round((a+275)/350)*350-275
                # a[a>1125] = 1125
                # a[a<75] = 75
                # #100 433 766 1099
                a = np.round((a+233)/333)*333-233
                a[a>1099] = 1099
                a[a<100] = 100
                #125 450 775 1100
                # a = np.round((a+200)/325)*325-200
                # a[a>1100] = 1100
                # a[a<125] = 125
                return a
            elif scale_number == '8':#30 197.5 332.5 467.5 602.5 737.5 872.5 1007.5
                # a = np.round((a+110)/140)*140-110
                # a[a>1010] = 1010
                #125 250 375 500 625 750 875 1000
                # a = np.round((a+125)/125)*125-125
                # a[a>1000] = 1000
                # a[a<125] = 125
                #50 200 350 500 650 800 950 1100
                # a = np.round((a+100)/150)*150-100
                # a[a>1100] = 1100
                # a[a<50] = 50
                #50 185 320 455 590 725 860 995
                # a = np.round((a+85)/135)*135-85
                # a[a>995] = 995
                # a[a<50] = 50
                #75 225 375 525 675 825 975 1125
                a = np.round((a+75)/150)*150-75
                a[a>1125] = 1125
                a[a<75] = 75
                return a
            elif scale_number == 'arbitrary':
                mask = np.abs(a) / a
                a = np.abs(a)
                a = np.power(2, np.round(np.log2(a)))
                return mask * a
                # return a
    elif mapping_mode == 'neg_act':
    #     if scale_number == "1":

    #     elif scale_number == '2':
        
        if scale_number == '4':
            #100 400 700 1000
            a = np.round((a+200)/300) * 300 - 200
            a[a>1000] = 1000
            a[a<100] = 100
            return a

        elif scale_number == '8':
            # #30 105 180 255 330 405 480 555
            # a = np.round((a+45)/75)*75-45
            # a[a>555] = 555
            # a[a<30] = 30
            # return a
            #45 120 195 270 345 420 495 570
            a = np.round((a+30)/75)*75-30
            a[a>570] = 570
            a[a<45] = 45
            return a
            #45 125 205 285 365 445 525 605
            # a = np.round((a+35)/80)*80-35
            # a[a>605] = 605
            # a[a<45] = 45
            # return a

        elif scale_number == 'arbitrary':
            return a
    ##exponential round
    # mask = np.abs(a) / a
    # a = np.abs(a)
    # a = np.power(2, np.round(np.log2(a)))
    # return mask * a
    
    #linear scale
    # mask = np.abs(a) / a
    # a = np.abs(a)
    #start with 25
    # a = np.round((a+100)/125)*125-100
    # a[a>400] = 400
    #start with 30
    # a = np.round((a+120)/150)*150-120
    # a[a>480] = 480
    # return mask * a
    #start with 500
    # a = np.round(a/625)*625
    # a[a>2500] = 2500
    # return mask * a
    # return a
    # return np.ones_like(a) * 250 * mask
    ##exponential ceil
    # mask = np.abs(a) / a
    # a = np.abs(a)
    # a = np.ceil(a/250) * 250
    # a[a>1000] = 1000
    # return mask * a

# print(array2bit([-500, -20, -358, -201, -400, -175, 175, 400, 201, 358, 20, 500]))