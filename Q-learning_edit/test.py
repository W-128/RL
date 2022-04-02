list1 = [1, 2, 3, 4, 5, 6]
list2 = []
list2.append(list1[2])
for element in list1:
    if element == 2:
        del element
print(list1)
print(list2)
