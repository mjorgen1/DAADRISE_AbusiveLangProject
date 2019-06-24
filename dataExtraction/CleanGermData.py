# deletes all the usernames from the original text file and saves the comments and labels to a new text file
# the new text file has an empty line at the end, so delete if you'd like

def main():
    # opens the text file to extract data from, and this needs to be in the same folder as this python script
    f1 = open('germeval2018.test.txt','r')

    # make a new text file and open it
    f2 = open("cleanGermanDataTest.txt","w+")

    # i is for the purposes of testing
    i = 1

    list_spaces = [" ", "  ", "   ", "    ", "     ", "      ", "       ", "        "]

    # iterates through the text file line by line
    for line in f1.readlines():
        # ^ for the above we know the total num of chars in the string
        delete = ""
        if(line.find('@') != -1):
            for char in line:
                if(char == "@"):
                    delete += char
                    print(char)
                    start = line.find(char)+1
                    for x in range(start, len(line)):
                        if(not (line[x] in list_spaces)):
                            delete += line[x]
                        else:
                            line = line.replace(delete, "").lstrip().rstrip()
                            delete = ""
                            break
        i += 1
        print(i)
        f2.write(line)

    f1.close()
    f2.close()

if __name__ == '__main__':
    main()