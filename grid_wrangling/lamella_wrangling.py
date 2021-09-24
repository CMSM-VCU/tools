lower = list(range(3,21,1))
upper = list(range(21,39,1))


output = ""
count = 0

for i in range(len(lower)):
    for j in range(len(lower)):
        if j != i and j != i-1 and j != i+1:
            count += 1
            output += "\t" + str(lower[i]) + "\t" + str(lower[j]) + "\n"

for i in range(len(upper)):
    for j in range(len(upper)):
        if j != i and j != i-1 and j != i+1:
            count += 1
            output += "\t" + str(upper[i]) + "\t" + str(upper[j]) + "\n"

print("Disconnect" + "\n\t" + str(count) + "\n" + output)


for i in range(len(lower)):
        if i != 0:
            print("intergace_force_coef_" + str(lower[i]) + "_" + str(lower[i-1]))
            print("\t1.0")
            print("intergace_strength_coef_" + str(lower[i]) + "_" + str(lower[i-1]))
            print("\t1.0")
        print("intergace_force_coef_" + str(lower[i]) + "_1" )
        print("\t1.0")
        print("intergace_strength_coef_" + str(lower[i]) + "_1" )
        print("\t1.0")
        print("intergace_force_coef_" + str(lower[i]) + "_2")
        print("\t1.0")
        print("intergace_strength_coef_" + str(lower[i]) + "_2")
        print("\t1.0")


for i in range(len(upper)):
        if i != 0:
            print("intergace_force_coef_" + str(upper[i]) + "_" + str(upper[i-1]))
            print("\t1.0")
            print("intergace_strength_coef_" + str(upper[i]) + "_" + str(upper[i-1]))
            print("\t1.0")
        print("intergace_force_coef_" + str(upper[i]) + "_1" )
        print("\t1.0")
        print("intergace_strength_coef_" + str(upper[i]) + "_1" )
        print("\t1.0")
        print("intergace_force_coef_" + str(upper[i]) + "_2")
        print("\t1.0")
        print("intergace_strength_coef_" + str(upper[i]) + "_2")
        print("\t1.0")
