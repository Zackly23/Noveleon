data = {'Romance':0, 'Horor':0, 'Fantasi':0, 'Sejarah':0}

i,j = 0, 2

tups = list(data.items())
tups[i], tups[j] = tups[j], tups[i]
res = dict(tups)

print(res)
