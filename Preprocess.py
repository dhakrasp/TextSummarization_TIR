
with open('dailymail-validation','r') as f:
	text=f.readlines()
i=2
j=1
art=['']*4
#entities=[0]*200
entities=''
while j<=3:
	
	article=''
	while i<len(text) and text[i]!='\n' :
		if j==1:	
			words=text[i].split()
			if words[-1]=='1' or words[-1]=='2':
				#print(words[-1])
				st=" ".join(words[:-1])
				st.replace('\t',';')
				article=article+' '+st
		if j==2:
			lords=text[i].split()
			lords=" ".join(lords)
			lords.replace('\n',' ')
			article=article+' '+lords

		if j==3:
			entities=entities+' '+text[i]
			"""
			partentity=text[i].split(':')
			position=partentity[0].index('y')
			number=''
			for j in range(position+1,len(partentity[0])):
				number=number+partentity[0][j]
			number=int(number)
			print(number)
			entities[number]=partentity[1]
			"""		
		i+=1
	art[j]=article
	i=i+1
	j+=1
with open('preprocesseddailymailvalid.txt','w') as f:
	f.write(art[1]+'\n')
	f.write(art[2]+'\n')
	f.write(entities)
print(art[1])
print(art[2])
print(entities)
