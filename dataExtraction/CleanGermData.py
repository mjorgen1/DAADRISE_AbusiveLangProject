# deletes all the usernames from the original text file and saves the comments and labels to a new text file

def main():
    # opens the text file
    f1 = open('germeval2018.training.txt','r')

    # make a new text file and open it
    f2 = open("cleanGermanDataTraining.txt","w+")

    # i is for the purposes of testing
    i = 1

    # iterates through the text file line by line
    for line in f1.readlines():
        # TODO: go through the line looking for @ sign and save to a str until a space is encountered with a non@ following
        # ^ for the above we know the total num of chars in the string
        # TODO: delete the word(s?)
        # these should be within a for loop going through the line and it should have a break statement
        #word = "@"
       # line = line.replace(word, "")

        f2.write("open")

        # for the purposes of testing below should do one iteration
        i += 1
        if i == 2:
            break

    f1.close()
    f2.close()

if __name__ == '__main__':
    main()