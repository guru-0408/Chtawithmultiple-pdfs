num = int(input("Enter a number:"))

if num > 2:
    for i in range(2,int(num/2)+1):
        if num%i == 0:
            print(num, "this is not a prime number")
            break
        else:
            print(num,"This is a prime number")
            break
else:
    print(num,"This is not a prime number")