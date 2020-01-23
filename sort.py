fp = open('vgg_neg_act/vgg_negact_4scale_10ratio_2var.txt',"r+")
linesText = fp.readlines()
linesText.sort()
for line in linesText:
    print(line)
fp.seek(0)
fp.close()
fp=open('vgg_neg_act/vgg_negact_4scale_10ratio_2var.txt','w')
lines_to_write = []
lines_to_write.append(linesText[0])
for i in range(len(linesText)):
    if i>0:
        if linesText[i].split()[0] != linesText[i-1].split()[0]:
            lines_to_write.append(linesText[i])
fp.writelines(lines_to_write)
fp.close()