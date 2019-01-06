# Forward Information
import math
input_act=np.array(list(range(400))) #28*28
input_act=input_act.reshape(10,10,4)
input_act = np.moveaxis(input_act, -1, 0)

input_set=[]

def zero_extending(input_act, weight_row, weight_col, stride, duplicate):
    input_extend=input_act
    w_extend_row=weight_row+stride*(duplicate-1)
    stride_extend=stride*duplicate
    input_shape=input_act.shape
    
    left_row=(input_shape[1]-w_extend_row)%(stride_extend)
    left_col=(input_shape[2]-weight_col)%stride
    if left_row!=0:
        extend_row=int(stride_extend-left_row)
        input_extend=np.append(input_act, np.zeros((input_shape[0],extend_row,input_shape[2])),axis=1)
    input_extend_shape=input_extend.shape
    input_extend_row_cnt=input_extend_shape[1]
    if left_col!=0:
        extend_col=int(stride-left_col)
        input_extend=np.append(input_extend, np.zeros((input_shape[0],input_extend_row_cnt,extend_col)),axis=2)
    return input_extend


def act_fc_mapping(input_2D): # Get a 2D activation return a 2D input vector map
    #input_2D(patch_num, w_size)
    input_shape = input_2D.shape
    patch_num=input_shape[0]
    cb_array_col_cnt = math.ceil(input_shape[1]/cb_size) #number of mapped arrays
    
    input_mapped = np.empty([patch_num,cb_array_col_cnt])
    for row in range(patch_num):            
        for cb_col in range(cb_array_col_cnt):
            input_set.append(Activation())
            # Set the range of the array slice by defining staring and ending col
            col_strt = cb_col*cb_size
            if cb_col == cb_array_col_cnt-1:
                col_end = -1
            else:
                col_end = (cb_col+1)*cb_size
            
            # Create new array object and update the array stored with sliced activation array
            if col_end==-1:
                input_set[-1].update_array(input_2D[row][col_strt:])
            else:
                input_set[-1].update_array(input_2D[row][col_strt:(col_end)])
           
            new_cb_id = len(input_set)-1
            input_mapped[row][cb_col] = new_cb_id    
    return input_mapped

def act_conv_mapping(input_act,weight_4D, stride):
    #input_act(ch, row, col)  weight_4D(row, col, ch, NF)
    input_shape=input_act.shape
    input_row_cnt=input_shape[1]
    input_col_cnt=input_shape[2]
    input_ch_cnt=input_shape[0]
    input_list=[]
    
    w_shape=weight_4D.shape
    
    input_extend=zero_extending(input_act,w_shape[0],w_shape[1],stride,1)
    input_extend_shape=input_extend.shape
    print(input_extend_shape)
    input_extend_row_cnt=input_extend_shape[1]
    input_extend_col_cnt=input_extend_shape[2]
    w_size =(w_shape[0]*w_shape[1]*w_shape[2])
        
    for input_row in range(0,input_extend_row_cnt-w_shape[0]+1,stride):
        for input_col in range(0,input_extend_col_cnt-w_shape[1]+1,stride): 
            for j in range(w_shape[0]):
                for i in range(w_shape[1]):
                    for input_ch in range(input_ch_cnt):
                        input_list.append(input_extend[input_ch][input_row+j][input_col+i])
    input_array=np.array(input_list) #1D, w*size*patch_num
    print(input_array)
    
    input_size=len(input_array)
    patch_num=int(input_size/w_size)
    input_2D=input_array.reshape(patch_num,w_size)
    
    return act_fc_mapping(input_2D)

def act_dw_conv_mapping(input_act, weight_3D, stride, duplicate, time_multiplex):
    #input_act(ch, row, col)  weight_3D(row, col, NF)
    input_shape=input_act.shape
    input_row_cnt=int(input_shape[1])
    input_col_cnt=int(input_shape[2])
    input_ch_cnt=int(input_shape[0])
    input_list=[]
    
    w_shape=weight_3D.shape
    w_size =(w_shape[0]*w_shape[1])
    w_extend_row=w_shape[0]+stride*(duplicate-1)
    
    row_multiplex=math.ceil(w_shape[2]/time_multiplex)
    w_size_dw=(w_size+stride*w_shape[1]*(duplicate-1))*row_multiplex
    input_extend=zero_extending(input_act,w_shape[0], w_shape[1], stride, duplicate)
    input_extend_shape=input_extend.shape
    print(input_extend_shape)
    input_row_extend=int(input_extend_shape[1])
    input_col_extend=int(input_extend_shape[2])
    for input_ch in range(0, input_ch_cnt-input_ch_cnt%row_multiplex+1,row_multiplex):
        for input_col in range(0,input_col_extend-w_shape[1]+1,stride):
            for input_row in range(0,input_extend_shape[1]-w_extend_row+1,stride*duplicate):
                for k in range(row_multiplex):
                    for j in range(stride*(duplicate-1)+w_shape[0]):
                        for i in range(w_shape[1]):
                            if input_ch+k<input_ch_cnt:
                                input_list.append(input_extend[input_ch+k][input_row+j][input_col+i])
                            else:
                                input_list.append(0)
    input_array=np.array(input_list) #1D, w*size*patch_num
    input_size=len(input_array)
    patch_num=int(input_size/w_size_dw)
    input_2D=input_array.reshape(patch_num,w_size_dw)
    return act_fc_mapping(input_2D)

weight=weight.reshape(3,3,5)
input_act=np.array(list(range(400)))
input_act=input_act.reshape(10,10,4)
input_act=np.moveaxis(input_act, -1, 0)
#input_mapped=act_dw_conv_mapping(input_act,weight,1,2,2)
input_mapped=act_conv_mapping(input_act,np_final,1)
print(input_mapped)