label = []
for label_text in open('new_label_1.txt'):
	label.append(int(label_text.split()[-1]))
total=0
accuracy=0
for image in open('vgg_energy.txt'):
	index = int(image.split()[0][15:-5])
	
	total +=1
	if int(image.split()[-1]) == label[index-1]-1:
		accuracy +=1
				#nf.write(image.split()[0]+' '+image.split()[-1]+' '+label[44:]+ '\n')
				#nf.close()

print(total)
print(accuracy)
f_accuracy=accuracy/total
print(f_accuracy)